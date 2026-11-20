"""Tashqi ko'rinish belgilari: ko'zoynak, soqol/mo'ylov, bosh kiyim.

CLIP (ViT-B-32) zero-shot: rasm bir nechta matnli tavsif bilan solishtiriladi va eng mosi tanlanadi.
Har bir sinf uchun bir nechta ibora (prompt ensemble) ishlatiladi — bu natijani barqarorlashtiradi.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import Config

log = logging.getLogger(__name__)

NONE = "yo'q"

# belgi -> {sinf nomi: [inglizcha tavsiflar]}
ATTRIBUTES: dict[str, dict[str, list[str]]] = {
    "glasses": {
        NONE: [
            "a close-up photo of a face without glasses",
            "a photo of a person with bare eyes and no eyewear",
        ],
        "ko'zoynak": [
            "a close-up photo of a face wearing eyeglasses",
            "a photo of a person wearing prescription glasses with clear lenses",
        ],
        "quyosh ko'zoynagi": [
            "a close-up photo of a face wearing sunglasses",
            "a photo of a person wearing dark tinted sunglasses",
        ],
    },
    "beard": {
        NONE: [
            "a close-up photo of a clean-shaven face",
            "a photo of a face with smooth skin and no facial hair",
        ],
        "soqol": [
            "a close-up photo of a man with a full beard",
            "a photo of a face with a thick beard",
        ],
        "qisqa soqol": [
            "a close-up photo of a man with stubble",
            "a photo of a face with short unshaven stubble",
        ],
        "mo'ylov": [
            "a close-up photo of a man with a mustache and no beard",
            "a photo of a face with only a mustache",
        ],
    },
    "headwear": {
        NONE: [
            "a photo of a person with no hat and visible hair",
            "a photo of a person's bare head",
        ],
        "kepka": [
            "a photo of a person wearing a baseball cap",
            "a photo of a person wearing a cap with a visor",
        ],
        "shlyapa": [
            "a photo of a person wearing a brimmed hat",
            "a photo of a person wearing a fedora hat",
        ],
        "qishki shapka": [
            "a photo of a person wearing a knitted beanie",
            "a photo of a person wearing a warm winter hat",
        ],
        "kapyushon": [
            "a photo of a person wearing a hood over their head",
            "a photo of a person with a hoodie hood up",
        ],
        "boshqa bosh kiyim": [
            "a photo of a person wearing a head covering",
            "a photo of a person wearing a skullcap or headscarf",
        ],
    },
}

ATTR_UZ = {"glasses": "ko'zoynak", "beard": "soqol", "headwear": "bosh kiyim"}


class AttributeEngine:
    def __init__(self, cfg: Config):
        import open_clip
        import torch

        self.torch = torch
        dev = cfg.resolved_device()
        self.device = dev
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained, cache_dir=str(cfg.models_dir / "clip"), device=dev
        )
        self.model.eval()
        tok = open_clip.get_tokenizer(cfg.clip_model)
        self.text: dict[str, tuple[list[str], torch.Tensor]] = {}
        with torch.no_grad():
            for attr, classes in ATTRIBUTES.items():
                embs = []
                for prompts in classes.values():
                    e = self.model.encode_text(tok(prompts).to(dev)).float()
                    e = e / e.norm(dim=-1, keepdim=True)
                    m = e.mean(0)
                    embs.append(m / m.norm())
                self.text[attr] = (list(classes), torch.stack(embs))
        self.scale = float(self.model.logit_scale.exp().item())

    @staticmethod
    def _crop(img: np.ndarray, bbox: np.ndarray, expand: float, shift_up: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w, h = (x2 - x1) * expand, (y2 - y1) * expand
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - (y2 - y1) * shift_up
        H, W = img.shape[:2]
        a, b = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
        c, d = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
        return img[b:d, a:c]

    def predict(self, img: np.ndarray, face_bbox: np.ndarray) -> dict[str, dict[str, float]]:
        """{belgi: {sinf: ehtimollik}} qaytaradi."""
        from PIL import Image

        torch = self.torch
        face = self._crop(img, face_bbox, 1.5, 0.0)        # ko'zoynak, soqol
        head = self._crop(img, face_bbox, 2.3, 0.35)       # bosh kiyim (tepasi ham kirsin)
        if face.size == 0 or head.size == 0:
            return {}
        batch = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in (face, head)
        ]).to(self.device)
        with torch.no_grad():
            im = self.model.encode_image(batch).float()
            im = im / im.norm(dim=-1, keepdim=True)
        out = {}
        for attr, (names, txt) in self.text.items():
            v = im[1] if attr == "headwear" else im[0]
            probs = (self.scale * (txt @ v)).softmax(0).cpu().numpy()
            out[attr] = {n: float(p) for n, p in zip(names, probs)}
        return out


def smooth(prev: dict[str, dict[str, float]], new: dict[str, dict[str, float]], alpha: float = 0.35):
    """Kadrlar bo'yicha eksponensial silliqlash (bitta xato kadr natijani o'zgartirmasin)."""
    if not prev:
        return new
    out = {}
    for attr, probs in new.items():
        p0 = prev.get(attr, probs)
        out[attr] = {k: (1 - alpha) * p0.get(k, 0.0) + alpha * v for k, v in probs.items()}
    return out


def labels(probs: dict[str, dict[str, float]], min_p: float = 0.5) -> dict[str, tuple[str, float]]:
    """Har bir belgi uchun ishonchli sinf: {belgi: (sinf, ehtimollik)}. Noaniq bo'lsa — kiritilmaydi."""
    out = {}
    for attr, p in probs.items():
        name = max(p, key=p.get)
        if p[name] >= min_p:
            out[attr] = (name, p[name])
    return out
