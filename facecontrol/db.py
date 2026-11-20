"""SQLite ombori: odamlar, yuz/tana namunalari va ko'rinishlar (sightings) tarixi."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

# numpy skalyarlari aks holda SQLite'ga BLOB bo'lib yoziladi
for _t in (np.float16, np.float32, np.float64):
    sqlite3.register_adapter(_t, float)
for _t in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
    sqlite3.register_adapter(_t, int)
sqlite3.register_adapter(np.bool_, bool)

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    created_at      REAL NOT NULL,
    first_seen      REAL,
    last_seen       REAL,
    seen_count      INTEGER NOT NULL DEFAULT 0,
    age_sum         REAL NOT NULL DEFAULT 0,
    age_n           INTEGER NOT NULL DEFAULT 0,
    male_votes      INTEGER NOT NULL DEFAULT 0,
    female_votes    INTEGER NOT NULL DEFAULT 0,
    best_face_q     REAL NOT NULL DEFAULT 0,
    best_face_path  TEXT,
    last_body_path  TEXT,
    last_source     TEXT,
    notes           TEXT,
    attrs           TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS face_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vec         BLOB NOT NULL,
    quality     REAL NOT NULL,
    created_at  REAL NOT NULL,
    crop_path   TEXT
);
CREATE TABLE IF NOT EXISTS body_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vec         BLOB NOT NULL,
    quality     REAL NOT NULL,
    created_at  REAL NOT NULL,
    crop_path   TEXT
);
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    ts_start    REAL NOT NULL,
    ts_end      REAL NOT NULL,
    frames      INTEGER NOT NULL DEFAULT 1,
    source      TEXT,
    method      TEXT,
    score       REAL,
    face_path   TEXT,
    body_path   TEXT,
    age         REAL,
    gender      TEXT,
    extra       TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS ix_face_person ON face_samples(person_id);
CREATE INDEX IF NOT EXISTS ix_body_person ON body_samples(person_id);
CREATE INDEX IF NOT EXISTS ix_sight_person ON sightings(person_id, ts_start);
CREATE INDEX IF NOT EXISTS ix_sight_ts ON sightings(ts_start);
"""


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


class Database:
    """Thread-safe SQLite o'rami. Bir nechta jarayon (veb + kamera) WAL rejimida birga ishlaydi."""

    def __init__(self, path: Path, crops_dir: Path):
        self.path = path
        self.crops_dir = crops_dir
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ meta
    def gallery_version(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='gallery_version'").fetchone()
        return int(row[0]) if row else 0

    def bump_version(self) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('gallery_version','1') "
                "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1"
            )
            self.conn.commit()

    # --------------------------------------------------------------- persons
    def create_person(self, name: str | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO persons(name, created_at) VALUES(?, ?)", (name, time.time())
            )
            self.conn.commit()
            pid = cur.lastrowid
        self.bump_version()
        return pid

    def person(self, pid: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM persons WHERE id=?", (pid,)).fetchone()
        return self._person_dict(row) if row else None

    def persons(self, q: str | None = None) -> list[dict]:
        sql = (
            "SELECT p.*, (SELECT COUNT(*) FROM face_samples f WHERE f.person_id=p.id) AS n_face,"
            " (SELECT COUNT(*) FROM body_samples b WHERE b.person_id=p.id) AS n_body FROM persons p"
        )
        args: tuple = ()
        if q:
            sql += " WHERE p.name LIKE ? OR CAST(p.id AS TEXT)=?"
            args = (f"%{q}%", q)
        sql += " ORDER BY COALESCE(p.last_seen, p.created_at) DESC"
        return [self._person_dict(r) for r in self.conn.execute(sql, args)]

    @staticmethod
    def _person_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["display_name"] = d["name"] or f"Noma'lum #{d['id']}"
        d["age"] = round(d["age_sum"] / d["age_n"], 1) if d["age_n"] else None
        mv, fv = d["male_votes"], d["female_votes"]
        d["gender"] = None if mv + fv == 0 else ("erkak" if mv >= fv else "ayol")
        d["gender_conf"] = None if mv + fv == 0 else round(max(mv, fv) / (mv + fv), 2)
        d["attrs"] = json.loads(d.get("attrs") or "{}")
        return d

    def rename(self, pid: int, name: str | None, notes: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE persons SET name=?, notes=COALESCE(?, notes) WHERE id=?", (name or None, notes, pid)
            )
            self.conn.commit()

    def update_attrs(self, pid: int, attrs: dict) -> None:
        with self._lock:
            row = self.conn.execute("SELECT attrs FROM persons WHERE id=?", (pid,)).fetchone()
            if not row:
                return
            cur = json.loads(row[0] or "{}")
            cur.update(attrs)
            self.conn.execute("UPDATE persons SET attrs=? WHERE id=?", (json.dumps(cur), pid))
            self.conn.commit()

    def add_height(self, pid: int, cm: float, keep: int = 60) -> float | None:
        """Bo'y o'lchovini qo'shadi va mediana (chetga chiqqanlarga chidamli) qiymatini qaytaradi."""
        with self._lock:
            row = self.conn.execute("SELECT attrs FROM persons WHERE id=?", (pid,)).fetchone()
            if not row:
                return None
            a = json.loads(row[0] or "{}")
            hs = (a.get("height_samples", []) + [round(float(cm), 1)])[-keep:]
            a["height_samples"] = hs
            a["height_cm"] = round(float(np.median(hs)), 1)
            a["height_n"] = len(hs)
            self.conn.execute("UPDATE persons SET attrs=? WHERE id=?", (json.dumps(a), pid))
            self.conn.commit()
            return a["height_cm"]

    def record_observation(
        self,
        pid: int,
        ts: float,
        source: str,
        age: float | None,
        gender: int | None,
        new_visit: bool,
    ) -> None:
        """Odam haqidagi yig'ma ma'lumotni yangilaydi (yosh, jins, oxirgi ko'rinish, ...)."""
        with self._lock:
            self.conn.execute(
                """UPDATE persons SET
                    first_seen = COALESCE(first_seen, ?),
                    last_seen  = MAX(COALESCE(last_seen, 0), ?),
                    seen_count = seen_count + ?,
                    age_sum    = age_sum + ?,
                    age_n      = age_n + ?,
                    male_votes = male_votes + ?,
                    female_votes = female_votes + ?,
                    last_source = ?
                   WHERE id=?""",
                (
                    ts, ts, 1 if new_visit else 0,
                    age or 0.0, 1 if age is not None else 0,
                    1 if gender == 1 else 0, 1 if gender == 0 else 0,
                    source, pid,
                ),
            )
            self.conn.commit()

    def set_best_face(self, pid: int, quality: float, path: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE persons SET best_face_q=?, best_face_path=? WHERE id=? AND best_face_q < ?",
                (quality, path, pid, quality),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def set_last_body(self, pid: int, path: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE persons SET last_body_path=? WHERE id=?", (path, pid))
            self.conn.commit()

    def delete_person(self, pid: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM persons WHERE id=?", (pid,))
            self.conn.commit()
        shutil.rmtree(self.crops_dir / str(pid), ignore_errors=True)
        self.bump_version()

    def merge(self, keep: int, drop: int) -> None:
        """`drop` odamni `keep` ga qo'shib yuboradi (barcha namunalar va tarix bilan)."""
        if keep == drop:
            return
        with self._lock:
            a, b = self.person(keep), self.person(drop)
            if not a or not b:
                raise ValueError("Bunday odam topilmadi")
            c = self.conn
            for t in ("face_samples", "body_samples", "sightings"):
                c.execute(f"UPDATE {t} SET person_id=? WHERE person_id=?", (keep, drop))
            best_q, best_p = (a["best_face_q"], a["best_face_path"])
            if b["best_face_q"] > best_q:
                best_q, best_p = b["best_face_q"], b["best_face_path"]
            first = min(x for x in (a["first_seen"], b["first_seen"], a["created_at"]) if x)
            last = max(x or 0 for x in (a["last_seen"], b["last_seen"]))
            c.execute(
                """UPDATE persons SET name=COALESCE(name, ?), first_seen=?, last_seen=?,
                    seen_count=seen_count+?, age_sum=age_sum+?, age_n=age_n+?,
                    male_votes=male_votes+?, female_votes=female_votes+?,
                    best_face_q=?, best_face_path=?, last_body_path=COALESCE(?, last_body_path)
                   WHERE id=?""",
                (
                    b["name"], first, last, b["seen_count"], b["age_sum"], b["age_n"],
                    b["male_votes"], b["female_votes"], best_q, best_p,
                    b["last_body_path"] if (b["last_seen"] or 0) > (a["last_seen"] or 0) else None,
                    keep,
                ),
            )
            c.execute("DELETE FROM persons WHERE id=?", (drop,))
            c.commit()
        # rasmlarni ko'chiramiz (yo'llar bazada nisbiy saqlanadi: "<pid>/...")
        src, dst = self.crops_dir / str(drop), self.crops_dir / str(keep)
        if src.exists():
            for f in src.rglob("*"):
                if f.is_file():
                    target = dst / f.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), target)
            shutil.rmtree(src, ignore_errors=True)
            with self._lock:
                for t, cols in (
                    ("face_samples", ["crop_path"]),
                    ("body_samples", ["crop_path"]),
                    ("sightings", ["face_path", "body_path"]),
                    ("persons", ["best_face_path", "last_body_path"]),
                ):
                    for col in cols:
                        c.execute(
                            f"UPDATE {t} SET {col}=? || SUBSTR({col}, ?) WHERE {col} LIKE ?",
                            (f"{keep}/", len(f"{drop}/") + 1, f"{drop}/%"),
                        )
                c.commit()
        self.bump_version()

    # --------------------------------------------------------------- samples
    def add_sample(self, kind: str, pid: int, vec: np.ndarray, quality: float, crop: str | None) -> int:
        with self._lock:
            cur = self.conn.execute(
                f"INSERT INTO {kind}_samples(person_id, vec, quality, created_at, crop_path) VALUES(?,?,?,?,?)",
                (pid, _blob(vec), quality, time.time(), crop),
            )
            self.conn.commit()
            return cur.lastrowid

    def delete_sample(self, kind: str, sid: int) -> None:
        with self._lock:
            self.conn.execute(f"DELETE FROM {kind}_samples WHERE id=?", (sid,))
            self.conn.commit()

    def samples(self, kind: str, pid: int | None = None) -> list[dict]:
        sql = f"SELECT * FROM {kind}_samples"
        args: tuple = ()
        if pid is not None:
            sql += " WHERE person_id=?"
            args = (pid,)
        out = []
        for r in self.conn.execute(sql + " ORDER BY id", args):
            d = dict(r)
            d["vec"] = _vec(d["vec"])
            out.append(d)
        return out

    # ------------------------------------------------------------- sightings
    def add_sighting(self, pid: int, ts: float, source: str, method: str, score: float,
                     face_path: str | None, body_path: str | None, age: float | None,
                     gender: str | None, extra: dict | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO sightings(person_id, ts_start, ts_end, source, method, score,
                     face_path, body_path, age, gender, extra) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (pid, ts, ts, source, method, score, face_path, body_path, age, gender,
                 json.dumps(extra or {})),
            )
            self.conn.commit()
            return cur.lastrowid

    def extend_sighting(self, sid: int, ts: float, frames: int, score: float, method: str,
                        face_path: str | None, body_path: str | None, pid: int | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE sightings SET ts_end=?, frames=?, score=MAX(score, ?), method=?,
                     face_path=COALESCE(?, face_path), body_path=COALESCE(?, body_path),
                     person_id=COALESCE(?, person_id) WHERE id=?""",
                (ts, frames, score, method, face_path, body_path, pid, sid),
            )
            self.conn.commit()

    def sightings(self, pid: int | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT s.*, p.name FROM sightings s JOIN persons p ON p.id=s.person_id"
        args: tuple = ()
        if pid is not None:
            sql += " WHERE s.person_id=?"
            args = (pid,)
        sql += " ORDER BY s.ts_start DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(sql, args + (limit,))]

    def stats(self) -> dict:
        q = lambda s: self.conn.execute(s).fetchone()[0]  # noqa: E731
        day = time.time() - 86400
        return {
            "persons": q("SELECT COUNT(*) FROM persons"),
            "named": q("SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"),
            "face_samples": q("SELECT COUNT(*) FROM face_samples"),
            "body_samples": q("SELECT COUNT(*) FROM body_samples"),
            "sightings": q("SELECT COUNT(*) FROM sightings"),
            "seen_24h": self.conn.execute(
                "SELECT COUNT(DISTINCT person_id) FROM sightings WHERE ts_end>?", (day,)
            ).fetchone()[0],
        }
