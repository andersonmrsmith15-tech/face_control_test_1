"""Buyruq qatori interfeysi:  python -m facecontrol <buyruq> ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def _fmt_ts(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"


def _open_source(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Manbani ochib bo'lmadi: {src}")
    return cap


def cmd_run(a, cfg):
    from .recognizer import Recognizer, annotate

    rec = Recognizer(cfg)
    cap = _open_source(a.source)
    is_file = not a.source.isdigit() and Path(a.source).exists()
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    t0, n, frame_i = time.time(), 0, 0
    start_wall = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            if a.skip > 1 and frame_i % a.skip:
                continue
            # video faylda vaqt — kadr vaqti bo'yicha (tez qayta ishlansa ham treklar to'g'ri ishlaydi)
            ts = start_wall + frame_i / fps_src if is_file else time.time()
            results = rec.process(frame, source=a.name or a.source, video=True, ts=ts)
            n += 1
            vis = annotate(frame, results)
            fps = n / max(time.time() - t0, 1e-6)
            cv2.putText(vis, f"{fps:.1f} FPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if a.save:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_src / max(a.skip, 1), (w, h))
                writer.write(vis)
            if not a.no_show:
                cv2.imshow("Face Control", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        rec.end_stream(a.name or a.source)
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"{n} kadr qayta ishlandi. Bazada: {rec.db.stats()}")


def cmd_scan(a, cfg):
    from .recognizer import Recognizer, annotate

    rec = Recognizer(cfg, use_body=not a.no_body)
    files = _images(a.paths)
    if not files:
        sys.exit("Rasm topilmadi")
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  o'qib bo'lmadi: {f}")
            continue
        res = rec.process(img, source=str(f), video=False)
        names = ", ".join(
            f"{'YANGI ' if r.is_new else ''}{r.name}({r.score:.2f})" for r in res if r.person_id is not None
        )
        unk = sum(1 for r in res if r.person_id is None)
        print(f"{f.name}: {names or '-'}" + (f"  [+{unk} aniqlanmagan]" if unk else ""))
        if out_dir:
            cv2.imwrite(str(out_dir / f.name), annotate(img, res))
    print("Bazada:", rec.db.stats())


def cmd_enroll(a, cfg):
    from .recognizer import Recognizer

    rec = Recognizer(cfg)
    files = _images(a.paths)
    imgs = [i for i in (cv2.imread(str(f)) for f in files) if i is not None]
    pid, n = rec.enroll(a.name, imgs, person_id=a.id)
    print(f"'{a.name}' -> #{pid}: {n}/{len(imgs)} rasm qabul qilindi")


def _db(cfg):
    from .db import Database

    return Database(cfg.db_path, cfg.crops_dir)


def cmd_list(a, cfg):
    db = _db(cfg)
    rows = db.persons(a.q)
    print(f"{'ID':>4}  {'Ism':<22} {'Ko`rildi':>8} {'Yuz':>4} {'Tana':>4} {'Yosh':>5} {'Jins':<6} Oxirgi marta")
    for p in rows:
        print(f"{p['id']:>4}  {p['display_name'][:22]:<22} {p['seen_count']:>8} {p['n_face']:>4} {p['n_body']:>4} "
              f"{p['age'] or '-':>5} {p['gender'] or '-':<6} {_fmt_ts(p['last_seen'])}")
    print(db.stats())


def cmd_show(a, cfg):
    db = _db(cfg)
    p = db.person(a.id)
    if not p:
        sys.exit("Topilmadi")
    for k in ("id", "display_name", "seen_count", "age", "gender", "gender_conf", "attrs", "notes", "last_source"):
        print(f"{k:>14}: {p[k]}")
    print(f"{'first_seen':>14}: {_fmt_ts(p['first_seen'])}\n{'last_seen':>14}: {_fmt_ts(p['last_seen'])}")
    print(f"{'yuz namunalar':>14}: {len(db.samples('face', a.id))}   tana: {len(db.samples('body', a.id))}")
    print("\nOxirgi ko'rinishlar:")
    for s in db.sightings(a.id, limit=a.limit):
        dur = s["ts_end"] - s["ts_start"]
        print(f"  {_fmt_ts(s['ts_start'])}  {dur:6.1f}s  {s['frames']:>5} kadr  {s['method']:<11} {s['score']:.2f}  {s['source']}")


def cmd_rename(a, cfg):
    _db(cfg).rename(a.id, a.name, a.notes)
    print("OK")


def cmd_merge(a, cfg):
    db = _db(cfg)
    for d in a.drop:
        db.merge(a.keep, d)
    print("OK")


def cmd_delete(a, cfg):
    _db(cfg).delete_person(a.id)
    print("O'chirildi")


def cmd_dedupe(a, cfg):
    from .gallery import Gallery

    db = _db(cfg)
    g = Gallery(cfg, db)
    sugg = g.merge_suggestions()
    if not sugg:
        print("Dublikat topilmadi")
        return
    merged = set()
    for x, y, s in sugg:
        px, py = db.person(x), db.person(y)
        if not px or not py or x in merged or y in merged:
            continue
        print(f"#{x} {px['display_name']}  <->  #{y} {py['display_name']}   o'xshashlik={s:.3f}")
        if a.auto and s >= a.min_sim:
            keep, drop = (x, y) if (px["name"] or not py["name"]) else (y, x)
            db.merge(keep, drop)
            merged.add(drop)
            print(f"   -> #{drop} #{keep} ga birlashtirildi")


def cmd_export(a, cfg):
    db = _db(cfg)
    data = []
    for p in db.persons():
        p = {k: v for k, v in p.items()}
        p["sightings"] = db.sightings(p["id"], limit=100000)
        data.append(p)
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    print(f"{len(data)} odam -> {a.out}")


def cmd_serve(a, cfg):
    import uvicorn

    from .web import create_app

    app = create_app(cfg, source=a.source, source_name=a.name)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


def main(argv=None):
    from .config import Config

    ap = argparse.ArgumentParser(prog="facecontrol", description="Yuz va tana bo'yicha tanish tizimi")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="kamera / video / RTSP oqimni real vaqtda tanish")
    s.add_argument("--source", default="0", help="0 (veb-kamera), video.mp4 yoki rtsp://...")
    s.add_argument("--name", default=None, help="manba nomi (masalan 'kirish-eshik')")
    s.add_argument("--no-show", action="store_true")
    s.add_argument("--save", default=None, help="natijani video faylga yozish")
    s.add_argument("--skip", type=int, default=1, help="har N-kadrni qayta ishlash")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("scan", help="rasmlar papkasini qayta ishlash va bazaga yig'ish")
    s.add_argument("paths", nargs="+")
    s.add_argument("--out", default=None, help="belgilangan rasmlarni saqlash papkasi")
    s.add_argument("--no-body", action="store_true")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("enroll", help="odamni ism bilan rasmlardan ro'yxatga olish")
    s.add_argument("name")
    s.add_argument("paths", nargs="+")
    s.add_argument("--id", type=int, default=None, help="mavjud odamga qo'shish")
    s.set_defaults(fn=cmd_enroll)

    s = sub.add_parser("list", help="odamlar ro'yxati")
    s.add_argument("-q", default=None)
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("show", help="odam haqida batafsil")
    s.add_argument("id", type=int)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("rename", help="odamga ism berish")
    s.add_argument("id", type=int)
    s.add_argument("name")
    s.add_argument("--notes", default=None)
    s.set_defaults(fn=cmd_rename)

    s = sub.add_parser("merge", help="dublikatlarni birlashtirish")
    s.add_argument("keep", type=int)
    s.add_argument("drop", type=int, nargs="+")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("delete", help="odamni va barcha ma'lumotlarini o'chirish")
    s.add_argument("id", type=int)
    s.set_defaults(fn=cmd_delete)

    s = sub.add_parser("dedupe", help="dublikat bo'lishi mumkin bo'lgan odamlarni topish")
    s.add_argument("--auto", action="store_true", help="avtomatik birlashtirish")
    s.add_argument("--min-sim", type=float, default=0.62)
    s.set_defaults(fn=cmd_dedupe)

    s = sub.add_parser("export", help="barcha ma'lumotni JSON ga eksport")
    s.add_argument("out")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("serve", help="veb-panel (+ ixtiyoriy jonli kamera)")
    s.add_argument("--source", default=None, help="kamera/video/RTSP; berilmasa faqat panel")
    s.add_argument("--name", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("ultralytics", "boxmot", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load(a.config)
    a.fn(a, cfg)


if __name__ == "__main__":
    main()
