"""Xotiradagi galereya: tezkor vektor qidiruv va har bir tanishda o'zini yangilab boruvchi namunalar."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import Config
from .db import Database


@dataclass
class Sample:
    id: int
    person_id: int
    vec: np.ndarray
    quality: float
    created_at: float


class _Index:
    """Bitta modalnost (yuz yoki tana) uchun vektor indeksi."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.samples: dict[int, list[Sample]] = {}
        self._dirty = True
        self._mat = np.zeros((0, dim), np.float32)
        self._owner = np.zeros(0, np.int64)
        self._ts = np.zeros(0, np.float64)
        self._protos: dict[int, np.ndarray] = {}

    def add(self, s: Sample) -> None:
        self.samples.setdefault(s.person_id, []).append(s)
        self._dirty = True

    def remove(self, person_id: int, sample_id: int) -> None:
        lst = self.samples.get(person_id, [])
        self.samples[person_id] = [s for s in lst if s.id != sample_id]
        self._dirty = True

    def drop_person(self, pid: int) -> None:
        self.samples.pop(pid, None)
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        vecs, owners, ts = [], [], []
        self._protos = {}
        for pid, lst in self.samples.items():
            if not lst:
                continue
            m = np.stack([s.vec for s in lst])
            w = np.array([max(s.quality, 0.05) for s in lst], np.float32)
            p = (m * w[:, None]).sum(0)
            self._protos[pid] = p / (np.linalg.norm(p) + 1e-9)
            vecs.append(m)
            owners += [pid] * len(lst)
            ts += [s.created_at for s in lst]
        self._mat = np.concatenate(vecs) if vecs else np.zeros((0, self.dim), np.float32)
        self._owner = np.asarray(owners, np.int64)
        self._ts = np.asarray(ts, np.float64)
        self._dirty = False

    def search(self, vec: np.ndarray, min_ts: float = 0.0, exclude: set[int] | None = None,
               top: int = 5) -> list[tuple[int, float]]:
        """Har bir odam uchun ball: 0.7 * eng yaqin namuna + 0.3 * prototip (o'rtacha) bilan o'xshashlik."""
        self._rebuild()
        if len(self._owner) == 0:
            return []
        sims = self._mat @ vec
        if min_ts:
            sims = np.where(self._ts >= min_ts, sims, -1.0)
        best: dict[int, float] = {}
        for pid, s in zip(self._owner, sims):
            if s > best.get(pid, -2.0):
                best[pid] = float(s)
        out = []
        for pid, mx in best.items():
            if mx <= -1.0 or (exclude and pid in exclude):
                continue
            proto = float(self._protos[pid] @ vec)
            out.append((int(pid), 0.7 * mx + 0.3 * proto))
        out.sort(key=lambda x: -x[1])
        return out[:top]

    def prototypes(self) -> dict[int, np.ndarray]:
        self._rebuild()
        return dict(self._protos)


class Gallery:
    def __init__(self, cfg: Config, db: Database):
        self.cfg = cfg
        self.db = db
        self.lock = threading.RLock()
        self.version = -1
        self.reload()

    def reload(self) -> None:
        with self.lock:
            self.face = _Index()
            self.body = _Index()
            for kind, idx in (("face", self.face), ("body", self.body)):
                for r in self.db.samples(kind):
                    idx.add(Sample(r["id"], r["person_id"], r["vec"], r["quality"], r["created_at"]))
            self.version = self.db.gallery_version()

    def maybe_reload(self) -> None:
        """Boshqa jarayon (masalan veb-panel) birlashtirish/o'chirish qilsa — qayta yuklaymiz."""
        if self.db.gallery_version() != self.version:
            self.reload()

    # ---------------------------------------------------------------- search
    def match_face(self, vec: np.ndarray, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        with self.lock:
            return self.face.search(vec, exclude=exclude)

    def match_body(self, vec: np.ndarray, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        with self.lock:
            min_ts = time.time() - self.cfg.body_ttl_hours * 3600
            return self.body.search(vec, min_ts=min_ts, exclude=exclude)

    def body_sim(self, pid: int, vec: np.ndarray) -> float:
        with self.lock:
            min_ts = time.time() - self.cfg.body_ttl_hours * 3600
            lst = [s for s in self.body.samples.get(pid, []) if s.created_at >= min_ts]
            if not lst:
                return -1.0
            return float(max(s.vec @ vec for s in lst))

    def has_person(self, pid: int) -> bool:
        return pid in self.face.samples or pid in self.body.samples

    # ---------------------------------------------------------------- update
    def learn(self, kind: str, pid: int, vec: np.ndarray, quality: float,
              crop: str | None | Callable[[], str | None]) -> str | bool:
        """Yangi namunani galereyaga qo'shadi. Siyosat:
        - deyarli bir xil namuna bo'lsa: sifati yaxshiroq bo'lsagina eskisini almashtiradi;
        - galereya to'lsa: yuzda — eng ortiqcha (boshqalarga eng o'xshash) va past sifatli
          namuna o'chiriladi; tanada — eng eskisi (kiyim o'zgaradi).
        """
        idx = self.face if kind == "face" else self.body
        cap = self.cfg.max_face_samples if kind == "face" else self.cfg.max_body_samples
        dedup = self.cfg.face_dedup_sim if kind == "face" else self.cfg.body_dedup_sim
        with self.lock:
            lst = idx.samples.get(pid, [])
            if lst:
                m = np.stack([s.vec for s in lst])
                sims = m @ vec
                j = int(np.argmax(sims))
                if sims[j] >= dedup:
                    if kind == "body":
                        # tana uchun yangilik muhim: eski o'xshash namunani yangisiga almashtiramiz
                        if time.time() - lst[j].created_at < 600 and quality <= lst[j].quality:
                            return False
                    elif quality <= lst[j].quality * 1.05:
                        return False
                    self._remove(kind, idx, lst[j])
                    lst = idx.samples.get(pid, [])
                if len(lst) >= cap:
                    self._evict(kind, idx, lst)
            if callable(crop):          # rasm faqat namuna qabul qilinganda saqlanadi
                crop = crop()
            sid = self.db.add_sample(kind, pid, vec, quality, crop)
            idx.add(Sample(sid, pid, vec.astype(np.float32), quality, time.time()))
            return crop or True

    def _remove(self, kind: str, idx: _Index, s: Sample) -> None:
        idx.remove(s.person_id, s.id)
        self.db.delete_sample(kind, s.id)

    def _evict(self, kind: str, idx: _Index, lst: list[Sample]) -> None:
        if kind == "body":
            victim = min(lst, key=lambda s: s.created_at)
        else:
            m = np.stack([s.vec for s in lst])
            redundancy = (m @ m.T).mean(1)
            q = np.array([s.quality for s in lst])
            score = redundancy - 0.5 * q          # ortiqcha + past sifatli => chiqarib tashlanadi
            victim = lst[int(np.argmax(score))]
        self._remove(kind, idx, victim)

    def forget_person(self, pid: int) -> None:
        with self.lock:
            self.face.drop_person(pid)
            self.body.drop_person(pid)

    def merge_suggestions(self) -> list[tuple[int, int, float]]:
        """Bir odam ikki marta ro'yxatga olingan bo'lishi mumkin bo'lgan juftliklar."""
        with self.lock:
            protos = self.face.prototypes()
        ids = sorted(protos)
        if len(ids) < 2:
            return []
        m = np.stack([protos[i] for i in ids])
        s = m @ m.T
        out = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if s[a, b] >= self.cfg.merge_suggest_sim:
                    out.append((ids[a], ids[b], float(s[a, b])))
        return sorted(out, key=lambda x: -x[2])
