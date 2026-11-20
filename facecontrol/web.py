"""Veb-panel (FastAPI): jonli oqim, odamlar bazasi, tarix, boshqaruv."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .db import Database

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Engine:
    """Recognizer'ni bitta qulf ostida saqlaydi (kamera oqimi va yuklangan rasmlar uchun umumiy)."""

    def __init__(self, cfg: Config, db: Database):
        self.cfg, self.db = cfg, db
        self._rec = None
        self.lock = threading.Lock()

    @property
    def rec(self):
        if self._rec is None:
            from .recognizer import Recognizer

            self._rec = Recognizer(self.cfg, self.db)
        return self._rec


class CameraWorker(threading.Thread):
    def __init__(self, engine: Engine, source: str, name: str):
        super().__init__(daemon=True)
        self.engine, self.source, self.name = engine, source, name
        self.jpeg: bytes | None = None
        self.fps = 0.0
        self.running = True
        self.status = "ishga tushmoqda"
        self.cond = threading.Condition()

    def run(self):
        from .recognizer import annotate

        src = int(self.source) if self.source.isdigit() else self.source
        is_file = isinstance(src, str) and Path(src).exists()
        while self.running:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.status = "manba ochilmadi — qayta urinish..."
                time.sleep(3)
                continue
            self.status = "ishlamoqda"
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
            t_last = time.time()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self.engine.lock:
                    res = self.engine.rec.process(frame, source=self.name, video=True)
                vis = annotate(frame, res)
                now = time.time()
                inst = 1.0 / max(now - t_last, 1e-3)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
                t_last = now
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.cond:
                    self.jpeg = buf.tobytes()
                    self.cond.notify_all()
                if is_file:
                    time.sleep(max(0.0, 1 / fps_src - (time.time() - now)))
            cap.release()
            with self.engine.lock:
                self.engine.rec.end_stream(self.name)
            if is_file:
                cap = None
                self.status = "video tugadi — qaytadan"
            else:
                self.status = "aloqa uzildi — qayta ulanish..."
                time.sleep(2)


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Rasmni o'qib bo'lmadi")
    return img


def create_app(cfg: Config, source: str | None = None, source_name: str | None = None) -> FastAPI:
    db = Database(cfg.db_path, cfg.crops_dir)
    engine = Engine(cfg, db)
    worker: CameraWorker | None = None
    if source is not None:
        worker = CameraWorker(engine, source, source_name or f"kamera-{source}")

    @asynccontextmanager
    async def lifespan(_app):
        if worker is not None:
            worker.start()
        yield
        if worker is not None:
            worker.running = False

    app = FastAPI(title="Face Control", lifespan=lifespan)
    app.mount("/crops", StaticFiles(directory=cfg.crops_dir), name="crops")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/stats")
    def stats():
        s = db.stats()
        s["live"] = None if worker is None else {"status": worker.status, "fps": round(worker.fps, 1),
                                                 "name": worker.name}
        return s

    @app.get("/api/persons")
    def persons(q: str | None = None):
        return db.persons(q)

    @app.get("/api/persons/{pid}")
    def person(pid: int):
        p = db.person(pid)
        if not p:
            raise HTTPException(404)
        strip = lambda rows: [{k: v for k, v in r.items() if k != "vec"} for r in rows]  # noqa: E731
        p["face_samples"] = strip(db.samples("face", pid))
        p["body_samples"] = strip(db.samples("body", pid))
        p["sightings"] = db.sightings(pid, limit=300)
        return p

    @app.post("/api/persons/{pid}/rename")
    def rename(pid: int, name: str = Form(""), notes: str = Form("")):
        db.rename(pid, name.strip() or None, notes)
        return {"ok": True}

    @app.post("/api/persons/{pid}/merge")
    def merge(pid: int, other: int = Form(...)):
        try:
            db.merge(pid, other)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.delete("/api/persons/{pid}")
    def delete(pid: int):
        db.delete_person(pid)
        return {"ok": True}

    @app.delete("/api/samples/{kind}/{sid}")
    def delete_sample(kind: str, sid: int):
        if kind not in ("face", "body"):
            raise HTTPException(400)
        db.delete_sample(kind, sid)
        db.bump_version()
        return {"ok": True}

    @app.get("/api/sightings")
    def sightings(limit: int = 50):
        return db.sightings(limit=limit)

    @app.get("/api/suggestions")
    def suggestions():
        from .gallery import Gallery

        g = Gallery(cfg, db)
        out = []
        for a, b, s in g.merge_suggestions()[:30]:
            pa, pb = db.person(a), db.person(b)
            if pa and pb:
                out.append({"a": pa, "b": pb, "sim": round(s, 3)})
        return out

    @app.post("/api/recognize")
    async def recognize(file: UploadFile = File(...)):
        import base64

        from .recognizer import annotate

        img = _decode(await file.read())
        with engine.lock:
            res = engine.rec.process(img, source=f"yuklash:{file.filename}", video=False)
        _, buf = cv2.imencode(".jpg", annotate(img, res), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
            "results": [{"person_id": r.person_id, "name": r.name, "score": round(r.score, 3),
                         "confidence": round(r.confidence, 1),
                         "method": r.method, "is_new": r.is_new, "age": r.age, "gender": r.gender}
                        for r in res],
        }

    @app.post("/api/enroll")
    async def enroll(name: str = Form(...), files: list[UploadFile] = File(...), person_id: int | None = Form(None)):
        imgs = [_decode(await f.read()) for f in files]
        with engine.lock:
            try:
                pid, n = engine.rec.enroll(name.strip(), imgs, person_id=person_id)
            except ValueError as e:
                raise HTTPException(400, str(e))
        return {"person_id": pid, "accepted": n, "total": len(imgs)}

    @app.get("/stream.mjpg")
    def stream():
        if worker is None:
            raise HTTPException(404, "Jonli manba ulanmagan (serve --source ...)")

        def gen():
            while True:
                with worker.cond:
                    worker.cond.wait(timeout=2)
                    frame = worker.jpeg
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    return app
