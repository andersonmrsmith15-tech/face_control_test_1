"""Asosiy tanish konveyeri: yuz + tana birlashmasi, trek bo'yicha ovoz berish, avtomatik ro'yxatga olish
va har bir kadrda ma'lumotlarni yangilash."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import attributes as A
from .config import Config
from .db import Database
from .gallery import Gallery
from .models import BodyDet, BodyEngine, FaceDet, FaceEngine, dominant_colors, estimate_height_cm

log = logging.getLogger(__name__)


@dataclass
class TrackState:
    key: str
    first_ts: float
    last_ts: float
    person_id: int | None = None
    votes: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    face_votes: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    pending_faces: list[tuple[FaceDet, str | None]] = field(default_factory=list)
    pending_bodies: list[np.ndarray] = field(default_factory=list)
    frames: int = 0
    sighting_id: int | None = None
    sighting_flush_ts: float = 0.0
    last_face_sample: float = 0.0
    last_body_sample: float = 0.0
    last_face_crop: float = 0.0
    last_body_crop: float = 0.0
    last_attr_ts: float = 0.0
    best_crop_q: float = 0.0
    best_score: float = 0.0
    last_height_ts: float = 0.0
    appear: dict = field(default_factory=dict)
    last_appear_ts: float = 0.0
    last_appear_db_ts: float = 0.0
    height: float | None = None
    conf: float = 0.0                 # joriy aniqlik foizi (0..100), kadrlar bo'yicha silliqlangan
    cur_method: str = ""
    method: str = ""
    face_path: str | None = None
    body_path: str | None = None
    last_bbox: np.ndarray | None = None


@dataclass
class Result:
    person_id: int | None
    name: str
    score: float
    method: str
    body_bbox: np.ndarray | None
    face_bbox: np.ndarray | None
    track: str
    age: float | None = None
    gender: str | None = None
    is_new: bool = False
    confidence: float = 0.0           # aniqlik foizi 0..100
    height: float | None = None       # taxminiy bo'y, sm
    appearance: list[str] = field(default_factory=list)   # masalan ["ko'zoynak", "kepka"]


class Recognizer:
    def __init__(self, cfg: Config | None = None, db: Database | None = None, use_body: bool = True):
        self.cfg = cfg or Config.load()
        self.db = db or Database(self.cfg.db_path, self.cfg.crops_dir)
        log.info("Yuz modeli yuklanmoqda...")
        self.face = FaceEngine(self.cfg)
        self.body = None
        if use_body:
            log.info("Tana modellari yuklanmoqda...")
            self.body = BodyEngine(self.cfg)
        self.attrs = None
        if self.cfg.detect_appearance:
            log.info("Tashqi ko'rinish modeli (CLIP) yuklanmoqda...")
            try:
                self.attrs = A.AttributeEngine(self.cfg)
            except Exception as e:  # pragma: no cover
                log.warning("Ko'rinish modeli yuklanmadi (%s) — belgilar o'chirildi", e)
        self.gallery = Gallery(self.cfg, self.db)
        self.tracks: dict[str, TrackState] = {}
        self._face_only_seq = 0
        self._names: dict[int, str] = {}
        self._heights: dict[int, float | None] = {}
        self._names_ts = 0.0
        self._reload_ts = 0.0

    # ================================================================ public
    def process(self, img: np.ndarray, source: str = "camera", video: bool = True,
                ts: float | None = None) -> list[Result]:
        ts = ts or time.time()
        if ts - self._reload_ts > 2.0:
            self._sync_gallery()
            self._reload_ts = ts
        bodies = self.body.detect(img, track=video) if self.body else []
        faces = self.face.detect(img)
        subjects = self._associate(bodies, faces)

        proposals = []
        for body, face in subjects:
            st = self._state_for(body, face, source, video, ts)
            st.last_ts = ts
            st.frames += 1
            st.last_bbox = body.bbox if body else face.bbox
            pid, score, method, learn_face = self._evidence(st, body, face, img, video)
            proposals.append((st, body, face, pid, score, method, learn_face))

        # bitta kadrda bitta odam ikki joyda bo'la olmaydi
        owner: dict[int, tuple[float, TrackState]] = {}
        for st, *_ in proposals:
            if st.person_id is not None:
                strength = st.votes.get(st.person_id, 0.0)
                if st.person_id not in owner or strength > owner[st.person_id][0]:
                    owner[st.person_id] = (strength, st)

        results = []
        for st, body, face, pid, score, method, learn_face in proposals:
            is_new = False
            if st.person_id is not None and owner.get(st.person_id, (0, st))[1] is not st:
                st.votes.pop(st.person_id, None)
                st.person_id = None
            if st.person_id is None:
                is_new = self._maybe_create(st, img, video)
            if st.person_id is not None:
                self._update_person(st, body, face, pid, score, method, learn_face, img, source, ts)
            if st.person_id is not None:
                if is_new:
                    st.conf, st.cur_method = 100.0, "yangi"
                elif pid == st.person_id and score > 0:
                    p = confidence_pct(score, method)
                    st.conf = p if st.conf == 0 else 0.6 * st.conf + 0.4 * p
                    st.cur_method = method
            else:
                st.conf = 0.0
            results.append(self._result(st, body, face, is_new))

        if video:
            self._expire(ts, source)
        else:
            for st, *_ in proposals:
                self._finish(st, source)
        return results

    def _sync_gallery(self) -> None:
        """Boshqa jarayon odamni birlashtirgan/o'chirgan bo'lsa — galereya va treklarni moslaymiz."""
        before = self.gallery.version
        self.gallery.maybe_reload()
        if self.gallery.version == before:
            return
        alive = {p["id"] for p in self.db.persons()}
        for st in self.tracks.values():
            if st.person_id is not None and st.person_id not in alive:
                st.person_id, st.sighting_id = None, None
                st.votes.clear()
                st.face_votes.clear()
        self._names_ts = 0.0

    def end_stream(self, source: str = "camera") -> None:
        for st in list(self.tracks.values()):
            self._finish(st, source)
        self.tracks.clear()
        if self.body:
            self.body.reset_tracker()

    def name_of(self, pid: int | None) -> str:
        if pid is None:
            return "?"
        now = time.time()
        if now - self._names_ts > 3 or pid not in self._names:
            persons = self.db.persons()
            self._names = {p["id"]: p["display_name"] for p in persons}
            self._heights = {p["id"]: p["attrs"].get("height_cm") for p in persons}
            self._names_ts = now
        return self._names.get(pid, f"#{pid}")

    def enroll(self, name: str, images: list[np.ndarray], person_id: int | None = None) -> tuple[int, int]:
        """Nomi ma'lum odamni rasmlar orqali qo'lda ro'yxatga olish. (person_id, qabul qilingan rasmlar)."""
        cands: list[tuple[FaceDet, np.ndarray, BodyDet | None]] = []
        for img in images:
            faces = [f for f in self.face.detect_any(img) if f.quality > 0.15]
            if not faces:
                continue
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            body = None
            if self.body:
                for b, ff in self._associate(self.body.detect(img, track=False), [f]):
                    if ff is f and b is not None:
                        body = b
            cands.append((f, img, body))
        if not cands:
            raise ValueError("Rasmlarda yaroqli yuz topilmadi")
        # izchillik: medianaga o'xshamagan (boshqa odamga tegishli) yuzlarni tashlab yuboramiz
        m = np.stack([c[0].embedding for c in cands])
        center = np.median(m, 0)
        center /= np.linalg.norm(center)
        cands = [c for c in cands if float(c[0].embedding @ center) > 0.35] or cands[:1]

        if person_id is None:
            for p in self.db.persons():
                if p["name"] and p["name"].lower() == name.lower():
                    person_id = p["id"]
            if person_id is None:
                hits = self.gallery.match_face(center.astype(np.float32))
                if hits and hits[0][1] >= self.cfg.face_match and not self.db.person(hits[0][0])["name"]:
                    person_id = hits[0][0]   # avtomatik yaratilgan noma'lum odam — unga nom beramiz
                    log.info("Bu odam allaqachon #%d sifatida mavjud — nom beriladi", person_id)
                else:
                    person_id = self.db.create_person(name)
        self.db.rename(person_id, name)
        for f, img, body in cands:
            crop = self.gallery.learn("face", person_id, f.embedding, max(f.quality, 0.5),
                                      lambda: self._save_crop(person_id, "faces", img, f.bbox, 1.6))
            if isinstance(crop, str):
                self.db.set_best_face(person_id, f.quality, crop)
            self.db.record_observation(person_id, time.time(), "enroll", f.age, f.gender, False)
            if body is not None and body.embedding is not None and body.quality > 0.3:
                bcrop = self.gallery.learn("body", person_id, body.embedding, body.quality,
                                           lambda: self._save_crop(person_id, "bodies", img, body.bbox, 1.0))
                if isinstance(bcrop, str):
                    self.db.set_last_body(person_id, bcrop)
        self.db.bump_version()
        self.gallery.version = self.db.gallery_version()
        return person_id, len(cands)

    # ============================================================ internals
    def _associate(self, bodies: list[BodyDet], faces: list[FaceDet]):
        """Yuzni tegishli tanaga bog'laydi: yuz markazi tananing yuqori qismida bo'lishi kerak."""
        pairs = []
        for fi, f in enumerate(faces):
            fcx, fcy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
            fw = f.bbox[2] - f.bbox[0]
            for bi, b in enumerate(bodies):
                bw, bh = b.bbox[2] - b.bbox[0], b.bbox[3] - b.bbox[1]
                if not (b.bbox[0] <= fcx <= b.bbox[2] and b.bbox[1] - 0.1 * bh <= fcy <= b.bbox[1] + 0.5 * bh):
                    continue
                if fw > bw * 1.1:
                    continue
                dx = abs(fcx - (b.bbox[0] + b.bbox[2]) / 2) / max(bw, 1)
                dy = (fcy - b.bbox[1]) / max(bh, 1)
                pairs.append((dx + 0.5 * dy, fi, bi))
        pairs.sort()
        used_f, used_b, out = set(), set(), []
        for _, fi, bi in pairs:
            if fi in used_f or bi in used_b:
                continue
            used_f.add(fi)
            used_b.add(bi)
            out.append((bodies[bi], faces[fi]))
        out += [(b, None) for i, b in enumerate(bodies) if i not in used_b]
        out += [(None, f) for i, f in enumerate(faces) if i not in used_f]
        return out

    def _state_for(self, body: BodyDet | None, face: FaceDet | None, source: str, video: bool,
                   ts: float) -> TrackState:
        if not video:
            self._face_only_seq += 1
            return TrackState(key=f"img:{self._face_only_seq}", first_ts=ts, last_ts=ts)
        if body is not None and body.track_id is not None:
            key = f"{source}:t{body.track_id}"
        else:
            # trek ID yo'q (yuz tanasiz yoki tana modeli o'chiq) — IoU bo'yicha oddiy kuzatuv
            box = face.bbox if face is not None else body.bbox
            key = None
            best = 0.3
            for k, st in self.tracks.items():
                if k.startswith(f"{source}:f") and st.last_bbox is not None and ts - st.last_ts < 1.0:
                    iou = _iou(box, st.last_bbox)
                    if iou > best:
                        best, key = iou, k
            if key is None:
                self._face_only_seq += 1
                key = f"{source}:f{self._face_only_seq}"
        if key not in self.tracks:
            self.tracks[key] = TrackState(key=key, first_ts=ts, last_ts=ts)
        return self.tracks[key]

    def _evidence(self, st: TrackState, body: BodyDet | None, face: FaceDet | None, img: np.ndarray,
                  video: bool):
        """Shu kadrdagi dalillarni yig'adi va trekning ovozlarini yangilaydi."""
        pid, score, method, learn_face = None, 0.0, "", False
        bvec = body.embedding if body is not None and body.quality > 0.2 else None

        if face is not None and face.quality > 0.1:
            hits = self.gallery.match_face(face.embedding)
            top_pid, top = hits[0] if hits else (None, -1.0)
            second = hits[1][1] if len(hits) > 1 else -1.0
            if top >= self.cfg.face_match and top - second > 0.03:
                pid, score, method = top_pid, top, "face"
            elif top >= self.cfg.face_maybe and bvec is not None and self._body_agrees(top_pid, bvec):
                pid, score, method = top_pid, top, "face+body"
            elif top >= self.cfg.face_maybe and st.person_id == top_pid and \
                    st.face_votes.get(top_pid, 0) >= 1.0:
                # trek allaqachon yuz bilan tasdiqlangan — yangi rakursni (masalan profil) o'rganamiz
                pid, score, method = top_pid, top, "face(track)"
            elif top < self.cfg.face_new and face.quality >= self.cfg.min_quality_enroll and st.person_id is None:
                crop = self._crop_img(img, face.bbox, 1.6)
                st.pending_faces.append((face, crop))
            if pid is not None:
                w = 1.0 if method == "face" else 0.7
                st.votes[pid] += score * w
                st.face_votes[pid] += score * w
                learn_face = face.quality >= self.cfg.min_quality_enroll and (
                    score >= self.cfg.face_update or method == "face(track)" or
                    (method == "face+body" and score >= self.cfg.face_maybe + 0.05))
        if pid is None and bvec is not None:
            hits = self.gallery.match_body(bvec)
            if hits:
                top = hits[0][1]
                margin = top - (hits[1][1] if len(hits) > 1 else 0.0)
                c = self.cfg
                if (top >= c.body_strong and margin >= c.body_strong_margin) or \
                        (top >= c.body_match and margin >= c.body_margin):
                    pid, score, method = hits[0][0], top, "body"
                    st.votes[pid] += score * 0.5 * (1.0 - body.occlusion)
        if bvec is not None and st.person_id is None and body.quality > 0.3:
            st.pending_bodies = (st.pending_bodies + [bvec])[-10:]

        self._resolve(st, video)
        return pid, score, method, learn_face

    def _body_agrees(self, pid: int, bvec: np.ndarray) -> bool:
        """Tana ham aynan shu odamni eng yaxshi nomzod deb hisoblaydimi (aniq farq bilan)?"""
        hits = self.gallery.match_body(bvec)
        if not hits or hits[0][0] != pid or hits[0][1] < self.cfg.body_support:
            return False
        return len(hits) == 1 or hits[0][1] - hits[1][1] >= 0.05

    def _resolve(self, st: TrackState, video: bool) -> None:
        if not st.votes:
            return
        best_pid = max(st.votes, key=st.votes.get)
        best = st.votes[best_pid]
        if st.person_id is None:
            face_ok = st.face_votes.get(best_pid, 0) >= self.cfg.face_maybe
            body_ok = best >= (1.1 if video else 0.37)  # tana: videoda ~3 kadr ketma-ket
            if face_ok or body_ok:
                st.person_id = best_pid
        elif best_pid != st.person_id:
            cur = st.votes.get(st.person_id, 0.0)
            if best > 1.5 * cur + 1.0:      # ID almashuvi (trekda odam almashgan) — tuzatamiz
                log.info("Trek %s: #%s -> #%s", st.key, st.person_id, best_pid)
                st.person_id = best_pid
                st.pending_faces.clear()

    def _maybe_create(self, st: TrackState, img: np.ndarray, video: bool) -> bool:
        need = self.cfg.new_person_min_frames if video else 1
        if len(st.pending_faces) < need or any(v >= 0.5 for v in st.votes.values()):
            return False
        faces = st.pending_faces
        # trek ichida yuzlar bir-biriga mos ekanini tekshiramiz (trek almashib ketmagan bo'lsin)
        ref = max(faces, key=lambda x: x[0].quality)[0].embedding
        good = [(f, c) for f, c in faces if float(f.embedding @ ref) >= 0.45]
        if len(good) < need:
            st.pending_faces = good
            return False
        # oxirgi tekshiruv: galereyada allaqachon paydo bo'lgan bo'lishi mumkin
        hits = self.gallery.match_face(ref)
        if hits and hits[0][1] >= self.cfg.face_new:
            st.pending_faces.clear()
            return False
        pid = self.db.create_person()
        st.person_id = pid
        best_q = 0.0
        for f, crop in sorted(good, key=lambda x: -x[0].quality)[: self.cfg.max_face_samples]:
            path = self.gallery.learn("face", pid, f.embedding, f.quality,
                                      lambda c=crop: self._write_crop(pid, "faces", c))
            if isinstance(path, str) and f.quality > best_q:
                best_q = f.quality
                self.db.set_best_face(pid, f.quality, path)
            st.votes[pid] += 0.5
            st.face_votes[pid] += 0.5
        for bv in st.pending_bodies[-3:]:
            self.gallery.learn("body", pid, bv, 0.5, None)
        st.pending_faces.clear()
        st.pending_bodies.clear()
        st.last_face_sample = st.last_ts
        st.best_score, st.method = 1.0, "yangi"
        self.gallery.version = self.db.gallery_version()
        log.info("Yangi odam ro'yxatga olindi: #%d", pid)
        return True

    def _update_person(self, st, body, face, pid, score, method, learn_face, img, source, ts) -> None:
        cfg, P = self.cfg, st.person_id
        agree = pid == P
        new_visit = st.sighting_id is None
        if agree:
            st.best_score = max(st.best_score, score)
            st.method = method if not st.method or method.startswith("face") else st.method

        face_path = body_path = None
        # --- yuz namunalari va rasmlari
        if face is not None and agree and method.startswith("face"):
            if learn_face and ts - st.last_face_sample >= cfg.sample_interval_s:
                saved = self.gallery.learn("face", P, face.embedding, face.quality,
                                           lambda: self._save_crop(P, "faces", img, face.bbox, 1.6))
                if saved:
                    st.last_face_sample = ts
                    if isinstance(saved, str):
                        face_path = saved
            if face_path or face.quality > st.best_crop_q * 1.1 or \
                    ts - st.last_face_crop >= cfg.crop_save_interval_s:
                face_path = face_path or self._save_crop(P, "faces", img, face.bbox, 1.6)
                st.last_face_crop = ts
                st.best_crop_q = max(st.best_crop_q, face.quality)
                self.db.set_best_face(P, face.quality, face_path)
        # --- tana namunalari: faqat yuz bilan tasdiqlangan trekdan (aks holda xato tarqaladi)
        face_confirmed = st.face_votes.get(P, 0) >= cfg.face_match
        if body is not None and body.embedding is not None and body.quality > 0.35:
            if face_confirmed and (agree or pid is None) and ts - st.last_body_sample >= cfg.sample_interval_s:
                saved = self.gallery.learn("body", P, body.embedding, body.quality,
                                           lambda: self._save_crop(P, "bodies", img, body.bbox, 1.0))
                if saved:
                    st.last_body_sample = ts
                    if isinstance(saved, str):
                        body_path = saved
                for bv in st.pending_bodies:
                    self.gallery.learn("body", P, bv, 0.4, None)
                st.pending_bodies.clear()
            if body_path or ts - st.last_body_crop >= cfg.crop_save_interval_s:
                body_path = body_path or self._save_crop(P, "bodies", img, body.bbox, 1.0)
                st.last_body_crop = ts
                self.db.set_last_body(P, body_path)
            if ts - st.last_attr_ts >= 30:
                colors = dominant_colors(img, body.bbox)
                if colors:
                    self.db.update_attrs(P, {"clothes": colors, "clothes_ts": ts})
                st.last_attr_ts = ts
        # --- taxminiy bo'y: yuz (masshtab) + to'liq skelet kerak
        if cfg.estimate_height and face is not None and body is not None and agree and \
                ts - st.last_height_ts >= cfg.height_interval_s:
            ipd = cfg.ipd_mm_female if face.gender == 0 else cfg.ipd_mm_male
            cm = estimate_height_cm(face, body, ipd)
            if cm is not None:
                st.last_height_ts = ts
                st.height = self.db.add_height(P, cm)
                self._heights[P] = st.height
        # --- tashqi ko'rinish: ko'zoynak, soqol, bosh kiyim
        if self.attrs is not None and face is not None and agree and face.quality >= 0.3 and \
                abs(face.yaw) < 45 and ts - st.last_appear_ts >= cfg.appearance_interval_s:
            probs = self.attrs.predict(img, face.bbox)
            if probs:
                st.appear = A.smooth(st.appear, probs)
                st.last_appear_ts = ts
                if ts - st.last_appear_db_ts >= 3.0:
                    self.db.update_appearance(P, A.labels(st.appear, cfg.appearance_min_p), ts)
                    st.last_appear_db_ts = ts
        st.face_path = face_path or st.face_path
        st.body_path = body_path or st.body_path

        # --- yig'ma ma'lumot (yosh, jins) — faqat ishonchli yuzdan
        if face is not None and agree and face.quality >= 0.3 and (new_visit or face_path):
            self.db.record_observation(P, ts, source, face.age, face.gender, new_visit)
        elif new_visit:
            self.db.record_observation(P, ts, source, None, None, True)

        # --- ko'rinishlar jurnali
        if st.sighting_id is None:
            st.sighting_id = self.db.add_sighting(
                P, st.first_ts, source, st.method or method or "track", st.best_score,
                st.face_path, st.body_path,
                face.age if face is not None else None,
                None if face is None or face.gender is None else ("erkak" if face.gender == 1 else "ayol"),
            )
            st.sighting_flush_ts = ts
        elif ts - st.sighting_flush_ts >= 1.0:
            self._flush_sighting(st)
            st.sighting_flush_ts = ts
            self.db.record_observation(P, ts, source, None, None, False)

    def _flush_sighting(self, st: TrackState) -> None:
        if st.sighting_id is not None:
            self.db.extend_sighting(st.sighting_id, st.last_ts, st.frames, st.best_score,
                                    st.method or "track", st.face_path, st.body_path, st.person_id)

    def _expire(self, ts: float, source: str) -> None:
        for k in [k for k, st in self.tracks.items() if ts - st.last_ts > self.cfg.track_lost_s]:
            self._finish(self.tracks.pop(k), source)

    def _finish(self, st: TrackState, source: str) -> None:
        if st.person_id is not None:
            self._flush_sighting(st)
            self.db.record_observation(st.person_id, st.last_ts, source, None, None, False)
        elif self.cfg.create_body_only_persons and st.frames >= 15 and len(st.pending_bodies) >= 3:
            pid = self.db.create_person()
            for bv in st.pending_bodies:
                self.gallery.learn("body", pid, bv, 0.4, None)
            self.db.record_observation(pid, st.last_ts, source, None, None, True)
            self.db.add_sighting(pid, st.first_ts, source, "body-only", 0.0, None, None, None, None)

    def _result(self, st: TrackState, body, face, is_new: bool) -> Result:
        return Result(
            person_id=st.person_id,
            name=self.name_of(st.person_id) if st.person_id is not None else "Aniqlanmoqda...",
            score=st.best_score,
            confidence=st.conf,
            appearance=[n for n, _ in A.labels(st.appear, self.cfg.appearance_min_p).values() if n != A.NONE]
            if st.person_id is not None else [],
            height=(self._heights.get(st.person_id) or st.height) if st.person_id is not None else None,
            method=st.cur_method or st.method,
            body_bbox=None if body is None else body.bbox,
            face_bbox=None if face is None else face.bbox,
            track=st.key,
            age=None if face is None else face.age,
            gender=None if face is None or face.gender is None else ("erkak" if face.gender == 1 else "ayol"),
            is_new=is_new,
        )

    # ---------------------------------------------------------------- crops
    @staticmethod
    def _crop_img(img: np.ndarray, bbox: np.ndarray, expand: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * expand, (y2 - y1) * expand
        H, W = img.shape[:2]
        a, b = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
        c, d = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
        return img[b:d, a:c].copy()

    def _write_crop(self, pid: int, kind: str, crop: np.ndarray | None) -> str | None:
        if crop is None or crop.size == 0:
            return None
        if kind == "faces" and crop.shape[0] > 256:
            crop = cv2.resize(crop, (int(crop.shape[1] * 256 / crop.shape[0]), 256))
        if kind == "bodies" and crop.shape[0] > 384:
            crop = cv2.resize(crop, (int(crop.shape[1] * 384 / crop.shape[0]), 384))
        rel = Path(str(pid)) / kind / f"{time.time():.3f}.jpg"
        out = self.cfg.crops_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return rel.as_posix()

    def _save_crop(self, pid: int, kind: str, img: np.ndarray, bbox: np.ndarray, expand: float) -> str | None:
        return self._write_crop(pid, kind, self._crop_img(img, bbox, expand))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _color(pid: int | None) -> tuple[int, int, int]:
    if pid is None:
        return (160, 160, 160)
    rng = np.random.default_rng(pid * 7919)
    return tuple(int(x) for x in rng.integers(60, 255, 3))


def confidence_pct(score: float, method: str) -> float:
    """Kosinus o'xshashlikni tushunarli aniqlik foiziga aylantiradi (empirik logistik moslash).
    Yuz: 0.33 -> 50%, 0.45 -> ~90%, 0.60 -> ~99%.  Tana: 0.70 -> 50%, 0.80 -> ~88%, 0.90 -> ~98%."""
    if method == "body":
        z = (score - 0.70) * 20
    else:
        z = (score - 0.33) * 18
    return float(100.0 / (1.0 + np.exp(-z)))


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_FONTS: dict[int, object] = {}


def _font(size: int):
    from PIL import ImageFont

    if size not in _FONTS:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                _FONTS[size] = ImageFont.truetype(path, size)
                break
        else:
            _FONTS[size] = ImageFont.load_default(size)
    return _FONTS[size]


_METHOD_UZ = {"face": "yuz", "face(track)": "yuz", "face+body": "yuz+tana", "body": "tana", "yangi": "yangi"}


def _conf_color(p: float) -> tuple[int, int, int]:
    """RGB: yashil (ishonchli) / sariq (o'rtacha) / to'q sariq (past)."""
    if p >= 85:
        return (40, 200, 90)
    if p >= 65:
        return (240, 200, 40)
    return (250, 130, 40)


def annotate(img: np.ndarray, results: list[Result]) -> np.ndarray:
    from PIL import Image, ImageDraw

    out = img.copy()
    H = out.shape[0]
    big = max(22, int(H / 20))           # ism shrifti (720p da ~36px)
    small = max(15, int(big * 0.55))
    labels = []
    for r in results:
        col = _color(r.person_id)
        thick = max(2, H // 360)
        if r.body_bbox is not None:
            x1, y1, x2, y2 = r.body_bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        if r.face_bbox is not None:
            x1, y1, x2, y2 = r.face_bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        anchor = r.face_bbox if r.face_bbox is not None else r.body_bbox
        name = ("YANGI: " if r.is_new else "") + r.name
        extra = ", ".join(x for x in (r.gender, f"~{r.age:.0f} yosh" if r.age else None,
                                      f"bo'y ~{r.height:.0f} sm" if r.height else None) if x)
        parts = [] if r.person_id is None else [f"{r.confidence:.0f}%", _METHOD_UZ.get(r.method, r.method)]
        sub = "  ·  ".join(parts + ([extra] if extra else [])) or " "
        labels.append((anchor, name, sub, col[::-1], r))

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fb, fs = _font(big), _font(small)
    W = pil.width
    placed: list[tuple[int, int, int, int]] = []
    # tanilganlar oxirida chiziladi (ustda bo'ladi), noma'lumlar avval
    labels.sort(key=lambda t: (t[4].person_id is not None, t[4].confidence))
    for anchor, name, sub, col, r in labels:
        nb = draw.textbbox((0, 0), name, font=fb)
        sb = draw.textbbox((0, 0), sub, font=fs)
        app = "  ·  ".join(r.appearance)
        ab = draw.textbbox((0, 0), app, font=fs) if app else (0, 0, 0, 0)
        pad = max(6, big // 5)
        line_gap = pad // 2
        bw = max(nb[2], sb[2], ab[2]) + 2 * pad
        bh = (nb[3] - nb[1]) + (sb[3] - sb[1]) + 3 * pad + ((sb[3] - sb[1]) + line_gap if app else 0)
        x = int(min(max(0, anchor[0]), W - bw))
        # nomzod joylar: ramka tepasi, ramka osti, so'ng yuqoriga/pastga siljitib — boshqa yorliqni yopmasin
        above, below = int(anchor[1]) - bh - 4, int(anchor[3]) + 4
        cands = [above, below] + [above - k * (bh + 4) for k in range(1, 6)] + \
                [below + k * (bh + 4) for k in range(1, 6)]
        cands = [c for c in cands if 0 <= c <= H - bh] or [max(0, min(above, H - bh))]
        y = next((c for c in cands if not any(
            x < px2 and x + bw > px1 and c < py2 and c + bh > py1 for px1, py1, px2, py2 in placed)), cands[0])
        placed.append((x, y, x + bw, y + bh))
        draw.rounded_rectangle((x, y, x + bw, y + bh), radius=pad, fill=(15, 17, 22))
        draw.rectangle((x, y, x + max(4, pad // 2), y + bh), fill=col)
        draw.text((x + pad + 2, y + pad - nb[1]), name, font=fb, fill=(255, 255, 255))
        sy = y + 2 * pad + (nb[3] - nb[1]) - sb[1]
        sub_col = (170, 175, 185) if r.person_id is None else _conf_color(r.confidence)
        draw.text((x + pad + 2, sy), sub, font=fs, fill=sub_col)
        if app:
            draw.text((x + pad + 2, sy + (sb[3] - sb[1]) + line_gap), app, font=fs, fill=(120, 190, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
