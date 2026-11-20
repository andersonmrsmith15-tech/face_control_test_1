"""Tizim sozlamalari. `config.toml` fayli orqali har qanday qiymatni o'zgartirish mumkin."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- Joylashuv ---
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"

    # --- Modellar ---
    face_model: str = "buffalo_l"          # InsightFace paketi (SCRFD + ArcFace r50)
    face_det_size: int = 640
    face_providers: list[str] = field(default_factory=list)  # bo'sh = avtomatik (CUDA > CoreML > CPU)
    body_model: str = "yolo11s-pose.pt"    # odam detektori + skelet nuqtalari (bo'y o'lchash uchun)
    body_conf: float = 0.45
    reid_model: str = "osnet_x1_0_msmt17"  # tana qayta-tanish (ReID) modeli
    device: str = "auto"                   # auto | mps | cuda | cpu
    tracker: str = "bytetrack.yaml"

    # --- Yuz sifati ---
    min_face_px: int = 48                  # bundan kichik yuzlar tanish uchun ishlatilmaydi
    min_face_det_score: float = 0.60
    max_yaw: float = 50.0                  # gradus; bundan katta burilgan yuz past sifatli
    max_pitch: float = 40.0
    min_quality_enroll: float = 0.45       # yangi odam yaratish / galereyaga qo'shish uchun

    # --- Yuz bo'yicha moslash (ArcFace kosinus o'xshashlik) ---
    face_match: float = 0.45               # ishonchli moslik
    face_maybe: float = 0.33               # noaniq zona: tana bilan tasdiqlansa qabul
    face_new: float = 0.30                 # hamma bilan shundan past bo'lsa — yangi odam
    face_update: float = 0.50              # galereyani yangilash uchun minimal o'xshashlik

    # --- Tana bo'yicha moslash (OSNet kosinus o'xshashlik) ---
    body_match: float = 0.78               # faqat tana bo'yicha tanish (+ body_margin farq kerak)
    body_margin: float = 0.10              # 1- va 2-nomzod orasidagi minimal farq
    body_strong: float = 0.85              # juda yuqori o'xshashlikda kichikroq farq yetarli
    body_strong_margin: float = 0.06
    max_body_occlusion: float = 0.35       # boshqa odam bilan shundan ko'p to'silgan tana o'rganilmaydi
    body_support: float = 0.72             # noaniq yuzni tasdiqlash (tana ham shu odamni 1-o'ringa qo'yishi shart)
    body_ttl_hours: float = 12.0           # kiyim o'zgaradi: eski tana namunalari ishlatilmaydi
    min_body_px: int = 64                  # tana balandligi

    # --- Galereya ---
    max_face_samples: int = 40
    max_body_samples: int = 30
    face_dedup_sim: float = 0.92           # shunchalik o'xshash namuna allaqachon bo'lsa — qo'shmaymiz
    body_dedup_sim: float = 0.95
    sample_interval_s: float = 1.5         # bitta trekdan namuna olish oralig'i

    # --- Video / trek ---
    new_person_min_frames: int = 3         # videoda yangi odam yaratish uchun kerakli sifatli kadrlar
    track_lost_s: float = 3.0              # trek shuncha vaqt ko'rinmasa — tugagan deb hisoblanadi
    crop_save_interval_s: float = 5.0
    create_body_only_persons: bool = False # yuzi hech ko'rinmagan odamlarni ham ro'yxatga olish
    merge_suggest_sim: float = 0.55        # dublikatlarni aniqlash chegarasi

    # --- Bo'y o'lchash (taxminiy) ---
    estimate_height: bool = True
    ipd_mm_male: float = 64.0              # o'rtacha ko'z qorachiqlari orasidagi masofa
    ipd_mm_female: float = 61.7
    height_interval_s: float = 1.0

    # --- Tashqi ko'rinish: ko'zoynak, soqol, bosh kiyim (CLIP zero-shot) ---
    detect_appearance: bool = True
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    appearance_interval_s: float = 0.7     # bitta trekda qanchalik tez-tez tekshiriladi
    appearance_min_p: float = 0.6          # shundan past ishonchli belgi ko'rsatilmaydi

    @property
    def db_path(self) -> Path:
        return self.data_dir / "facecontrol.db"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        cfg = cls()
        path = path or ROOT / "config.toml"
        if path.exists():
            raw = tomllib.loads(path.read_text())
            known = {f.name: f for f in fields(cls)}
            for k, v in raw.items():
                if k not in known:
                    raise ValueError(f"config.toml: noma'lum kalit '{k}'")
                if k.endswith("_dir"):
                    v = (ROOT / v).resolve() if not Path(v).is_absolute() else Path(v)
                setattr(cfg, k, v)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def resolved_providers(self) -> list[str]:
        if self.face_providers:
            return self.face_providers
        import onnxruntime as ort

        avail = ort.get_available_providers()
        pref = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [p for p in pref if p in avail] or ["CPUExecutionProvider"]

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
