"""Tezkor tekshiruv:  .venv/bin/python -m pytest tests -q   (yoki to'g'ridan-to'g'ri python bilan)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import insightface

from facecontrol.config import Config
from facecontrol.recognizer import Recognizer

SAMPLE = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"


def _recognizer(tmp: str) -> Recognizer:
    cfg = Config.load(Path("/nonexistent.toml"))
    cfg.data_dir = Path(tmp)
    cfg.crops_dir.mkdir(parents=True, exist_ok=True)
    return Recognizer(cfg)


def test_auto_enroll_and_reidentify():
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recognizer(tmp)
        img = cv2.imread(str(SAMPLE))
        first = rec.process(img, source="t", video=False)
        created = {r.person_id for r in first if r.is_new}
        assert len(created) >= 4, "guruh rasmida kamida 4 odam avtomatik ro'yxatga olinishi kerak"

        # o'zgartirilgan nusxa: aks ettirilgan + kichraytirilgan + qorong'i
        variant = cv2.convertScaleAbs(cv2.resize(cv2.flip(img, 1), None, fx=0.8, fy=0.8), alpha=0.75, beta=-15)
        second = rec.process(variant, source="t", video=False)
        matched = {r.person_id for r in second if r.person_id is not None}
        assert created <= matched, "barcha odamlar qayta tanilishi kerak"
        assert not any(r.is_new for r in second), "dublikat yaratilmasligi kerak"
        assert rec.db.stats()["persons"] == len(created)


def test_enroll_and_merge():
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recognizer(tmp)
        img = cv2.imread(str(SAMPLE))
        pid, n = rec.enroll("Test Odam", [img])
        assert n == 1 and rec.db.person(pid)["name"] == "Test Odam"
        other = rec.db.create_person()
        rec.db.merge(pid, other)
        assert rec.db.person(other) is None


if __name__ == "__main__":
    test_auto_enroll_and_reidentify()
    test_enroll_and_merge()
    print("OK")
