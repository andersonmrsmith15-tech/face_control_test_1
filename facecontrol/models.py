"""Neyron tarmoq modellari: yuz (InsightFace) va tana (YOLO + OSNet ReID)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class FaceDet:
    bbox: np.ndarray            # x1,y1,x2,y2
    kps: np.ndarray             # 5 nuqta
    det_score: float
    embedding: np.ndarray       # L2-normallangan 512-d
    age: float | None
    gender: int | None          # 1 = erkak, 0 = ayol
    yaw: float
    pitch: float
    quality: float = 0.0


@dataclass
class BodyDet:
    bbox: np.ndarray
    conf: float
    track_id: int | None
    embedding: np.ndarray | None = None
    quality: float = 0.0
    occlusion: float = 0.0
    faces: list[FaceDet] = field(default_factory=list)


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class FaceEngine:
    def __init__(self, cfg: Config):
        from insightface.app import FaceAnalysis

        self.cfg = cfg
        os.environ.setdefault("INSIGHTFACE_HOME", str(cfg.models_dir / "insightface"))
        self.app = FaceAnalysis(
            name=cfg.face_model,
            root=str(cfg.models_dir / "insightface"),
            providers=cfg.resolved_providers(),
            allowed_modules=["detection", "recognition", "genderage", "landmark_3d_68"],
        )
        self.app.prepare(ctx_id=0, det_size=(cfg.face_det_size, cfg.face_det_size))

    def quality(self, f: FaceDet) -> float:
        """0..1 sifat bahosi: o'lcham, detektor ishonchi, yuz burilishi va xiralik."""
        w, h = f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]
        size = min(w, h)
        if size < self.cfg.min_face_px or f.det_score < self.cfg.min_face_det_score:
            return 0.0
        size_q = min(1.0, (size - self.cfg.min_face_px) / 80.0 + 0.4)
        yaw_q = max(0.0, 1.0 - abs(f.yaw) / self.cfg.max_yaw)
        pitch_q = max(0.0, 1.0 - abs(f.pitch) / self.cfg.max_pitch)
        pose_q = yaw_q * 0.7 + pitch_q * 0.3
        return float(np.clip(f.det_score, 0, 1) * size_q * (0.25 + 0.75 * pose_q))

    @staticmethod
    def sharpness(img: np.ndarray, bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0
        g = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    def detect_any(self, img: np.ndarray) -> list[FaceDet]:
        """Ro'yxatga olish uchun: kesilgan/kichik yuz rasmlarida ham yuzni topishga harakat qiladi."""
        faces = self.detect(img)
        if faces:
            return faces
        h, w = img.shape[:2]
        scale = max(1.0, 320 / max(h, w))
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else img
        ph, pw = big.shape[0] // 2, big.shape[1] // 2
        padded = cv2.copyMakeBorder(big, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = self.detect(padded)
        for f in faces:                        # koordinatalarni asl rasmga qaytaramiz
            f.bbox = ((f.bbox - np.array([pw, ph, pw, ph], np.float32)) / scale).astype(np.float32)
            f.kps = (f.kps - np.array([pw, ph], np.float32)) / scale
            f.quality = max(f.quality, min(0.6, f.det_score))   # sun'iy kichiklik jarimasi yo'q
        return faces

    def detect(self, img: np.ndarray) -> list[FaceDet]:
        out = []
        for f in self.app.get(img):
            pitch, yaw = 0.0, 0.0
            if getattr(f, "pose", None) is not None:
                pitch, yaw = float(f.pose[0]), float(f.pose[1])
            fd = FaceDet(
                bbox=f.bbox.astype(np.float32),
                kps=f.kps,
                det_score=float(f.det_score),
                embedding=_l2(f.embedding.astype(np.float32)),
                age=float(f.age) if getattr(f, "age", None) is not None else None,
                gender=int(f.gender) if getattr(f, "gender", None) is not None else None,
                yaw=yaw,
                pitch=pitch,
            )
            q = self.quality(fd)
            sharp = self.sharpness(img, fd.bbox)
            if sharp < 40:               # juda xira yuz
                q *= max(0.3, sharp / 40)
            fd.quality = q
            out.append(fd)
        return out


class BodyEngine:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = cfg.resolved_device()
        weights = cfg.models_dir / cfg.body_model
        if not weights.exists():
            YOLO(cfg.body_model)                  # ultralytics joriy papkaga yuklab oladi
            if os.path.exists(cfg.body_model):
                os.replace(cfg.body_model, weights)
        src = str(weights) if weights.exists() else cfg.body_model
        self.yolo = YOLO(src)                     # video uchun (trek holati bilan)
        self.yolo_still = YOLO(src)               # alohida rasmlar uchun (trekka aralashmasin)
        self.reid = self._load_reid()

    def _load_reid(self):
        try:
            from boxmot.reid.factory import create_reid_encoder

            dev = "cpu" if self.device == "cpu" else self.device
            return create_reid_encoder(self.cfg.reid_model, device=dev)
        except Exception as e:  # pragma: no cover
            log.warning("ReID modeli yuklanmadi (%s) — tana bo'yicha tanish o'chirildi", e)
            return None

    def detect(self, img: np.ndarray, track: bool) -> list[BodyDet]:
        kw = dict(classes=[0], conf=self.cfg.body_conf, device=self.device, verbose=False)
        if track:
            res = self.yolo.track(img, persist=True, tracker=self.cfg.tracker, **kw)[0]
        else:
            res = self.yolo_still.predict(img, **kw)[0]
        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(xyxy)
        h, w = img.shape[:2]
        for b, c, t in zip(xyxy, conf, ids):
            b = np.clip(b, 0, [w, h, w, h]).astype(np.float32)
            bh, bw = b[3] - b[1], b[2] - b[0]
            # sifat: balandlik, ishonch, kadr chetida kesilmaganligi
            edge = 1.0 if (b[0] > 2 and b[2] < w - 2 and b[1] > 2 and b[3] < h - 2) else 0.75
            aspect = bh / max(bw, 1)
            aspect_q = 1.0 if 1.5 <= aspect <= 4.5 else 0.6
            q = float(c) * min(1.0, bh / 256) * edge * aspect_q if bh >= self.cfg.min_body_px else 0.0
            dets.append(BodyDet(bbox=b, conf=float(c), track_id=None if t is None else int(t), quality=float(q)))
        # to'silganlik: ramkaning qancha qismi boshqa odamlar bilan qoplangan
        for i, d in enumerate(dets):
            area = max((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), 1.0)
            occ = 0.0
            for j, o in enumerate(dets):
                if i != j:
                    iw = max(0.0, min(d.bbox[2], o.bbox[2]) - max(d.bbox[0], o.bbox[0]))
                    ih = max(0.0, min(d.bbox[3], o.bbox[3]) - max(d.bbox[1], o.bbox[1]))
                    occ = max(occ, iw * ih / area)
            d.occlusion = occ
            if occ > self.cfg.max_body_occlusion:
                d.quality *= 0.3
            else:
                d.quality *= 1.0 - occ
        self._embed(img, dets)
        return dets

    def _embed(self, img: np.ndarray, dets: list[BodyDet]) -> None:
        if self.reid is None or not dets:
            return
        import torch
        from boxmot.structures import Boxes, Detections, Frame

        rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()
        fr = Frame(image=rgb, sample_id="f")
        boxes = torch.tensor(np.stack([d.bbox for d in dets]), dtype=torch.float32)
        d = Detections(
            geometry=Boxes(boxes),
            scores=torch.tensor([min(1.0, d.conf) for d in dets], dtype=torch.float32),
            class_ids=torch.zeros(len(dets), dtype=torch.int64),
            sample_id="f",
        )
        feats = self.reid.encode([fr], [d])[0].numpy().astype(np.float32)
        feats = _l2(feats)
        for det, f in zip(dets, feats):
            det.embedding = f

    def reset_tracker(self) -> None:
        pred = getattr(self.yolo, "predictor", None)
        if pred is not None and getattr(pred, "trackers", None):
            for t in pred.trackers:
                t.reset()


def dominant_colors(img: np.ndarray, bbox: np.ndarray) -> dict:
    """Ustki va pastki kiyimning asosiy rangini aniqlaydi (qo'shimcha atribut sifatida)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 40:
        return {}
    h = crop.shape[0]
    cx1, cx2 = int(crop.shape[1] * 0.25), int(crop.shape[1] * 0.75)
    parts = {"upper": crop[int(h * 0.22):int(h * 0.5), cx1:cx2], "lower": crop[int(h * 0.55):int(h * 0.85), cx1:cx2]}
    out = {}
    for k, p in parts.items():
        if p.size == 0:
            continue
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        hh, s, v = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
        out[k] = _color_name(hh, s, v)
    return out


def _color_name(h: float, s: float, v: float) -> str:
    if v < 50:
        return "qora"
    if s < 40:
        return "oq" if v > 190 else "kulrang"
    h2 = h * 2  # OpenCV 0..180 -> 0..360
    for limit, name in ((15, "qizil"), (40, "to'q sariq"), (70, "sariq"), (160, "yashil"),
                        (200, "moviy"), (260, "ko'k"), (300, "binafsha"), (340, "pushti"), (361, "qizil")):
        if h2 < limit:
            return name if v > 90 or name in ("ko'k", "yashil") else f"to'q {name}"
    return "noma'lum"
