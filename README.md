# Face Control — yuz va to'liq tana bo'yicha tanish tizimi

Kamera, video, RTSP oqim yoki rasmlardagi odamlarni **yuz** va **tana (kiyim, gavda)** bo'yicha taniydi.
Yangi odamlarni avtomatik ro'yxatga oladi va har bir yangi kadrda ular haqidagi ma'lumotni yangilab boradi.

## Imkoniyatlar

| | |
|---|---|
| **Yuz** | InsightFace `buffalo_l`: SCRFD detektor + ArcFace (512 o'lchamli vektor) + yosh/jins + bosh burilishi |
| **Tana** | YOLO11 odam detektori + ByteTrack kuzatuv + OSNet ReID (kiyim/gavda vektori) |
| **Birlashma** | Yuz asosiy dalil. Yuz noaniq bo'lsa, tana tasdiqlaydi. Yuz ko'rinmasa (orqasi o'girilgan, uzoqda bo'lsa) — faqat tana bo'yicha taniydi |
| **Avto ro'yxatga olish** | Notanish odam sifatli yuz bilan ≥3 kadr ko'rinsa, `Noma'lum #N` bo'lib qo'shiladi. Keyin unga ism berish mumkin |
| **O'rganib borish** | Har bir ishonchli tanishda yangi rakurs/yorug'lik namunasi galereyaga qo'shiladi (har odamda 40 tagacha yuz, 30 tagacha tana namunasi). Ortiqcha yoki past sifatli namunalar avtomatik almashtiriladi |
| **Yig'iladigan ma'lumot** | Birinchi/oxirgi ko'rilgan vaqt, ko'rinishlar soni va davomiyligi, manba (kamera), o'rtacha yosh, jins, kiyim rangi, eng yaxshi yuz rasmi, tana rasmlari, to'liq tarix |
| **Ishonchlilik** | Trek bo'yicha ovoz berish, bitta kadrda bitta odam ikki joyda bo'lmasligi, ID almashuvini tuzatish, to'silgan tanadan o'rganmaslik, dublikatlarni topish |
| **Interfeys** | CLI + veb-panel (jonli oqim, odamlar, tarix, nomlash, birlashtirish, rasm yuklash) |

## O'rnatish

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Modellar birinchi ishga tushirishda `models/` papkasiga avtomatik yuklanadi (~400 MB).
Apple Silicon'da CoreML va MPS, NVIDIA'da CUDA avtomatik ishlatiladi.

## Foydalanish

```bash
source .venv/bin/activate

# Veb-panel + jonli kamera (http://127.0.0.1:8000)
python -m facecontrol serve --source 0 --name "kirish-eshik"

# IP kamera
python -m facecontrol serve --source "rtsp://user:parol@192.168.1.10:554/stream1" --name "ofis"

# Faqat oyna rejimida (q — chiqish)
python -m facecontrol run --source 0
python -m facecontrol run --source video.mp4 --no-show --save natija.mp4

# Rasmlar papkasini bazaga yig'ish
python -m facecontrol scan ~/Pictures/tadbir --out belgilangan/

# Ism bilan ro'yxatga olish (turli burchakdagi bir nechta rasm tavsiya etiladi)
python -m facecontrol enroll "Ali Valiyev" rasmlar/ali/

# Boshqaruv
python -m facecontrol list
python -m facecontrol show 12
python -m facecontrol rename 12 "Ali Valiyev"
python -m facecontrol merge 12 37          # 37 ni 12 ga qo'shish
python -m facecontrol dedupe               # dublikat bo'lishi mumkin bo'lganlar
python -m facecontrol dedupe --auto        # ishonchli dublikatlarni avtomatik birlashtirish
python -m facecontrol delete 37
python -m facecontrol export baza.json
```

Kamera va veb-panelni alohida jarayonlarda ham ishlatsa bo'ladi (`run` va `serve` bir bazani birga ishlatadi;
panelda qilingan birlashtirish/o'chirish 2 soniya ichida ishlayotgan kameraga ham yetib boradi).

## Qanday ishlaydi

```
kadr ─┬─ YOLO11 + ByteTrack ─→ tana ramkalari (trek ID) ─→ OSNet ─→ tana vektori
      └─ SCRFD ─→ yuz ─→ ArcFace vektori, yosh, jins, burilish, sifat
                       │
         yuz ↔ tana bog'lash (yuz tananing yuqori qismida)
                       │
   har bir trek uchun dalil:  yuz ≥ 0.45  → aniq
                              yuz 0.33–0.45 + tana ham shu odamni 1-o'ringa qo'ysa → aniq
                              yuz yo'q, tana ≥ 0.78 va 2-nomzoddan ≥ 0.10 farq → tana bo'yicha
                       │
   trek ovozlari → shaxs  ─→  galereyani yangilash, rasm saqlash, statistika, tarix
                 └ hech kimga mos emas + 3 ta sifatli kadr → yangi odam
```

Tana namunalari faqat **yuz bilan tasdiqlangan** trekdan o'rganiladi (xato tarqalmasligi uchun) va
12 soatdan keyin eskiradi (kiyim o'zgaradi). Uzoq muddatli shaxs aniqligi yuzga tayanadi.

## Sozlash

`cp config.example.toml config.toml` va qiymatlarni o'zgartiring. Asosiy chegaralar:

- `face_match` (0.45) — ko'tarilsa xato tanish kamayadi, "aniqlanmagan" ko'payadi
- `min_quality_enroll` (0.45) — yangi odam yaratish uchun yuz sifati
- `new_person_min_frames` (3) — videoda yangi odam yaratish uchun kadrlar
- `body_ttl_hours` (12) — tana namunalari amal qilish muddati
- `body_model` — `yolo11n.pt` (tezroq) / `yolo11m.pt` (aniqroq)

## Ma'lumotlar

```
data/facecontrol.db        SQLite: persons, face_samples, body_samples, sightings
data/crops/<id>/faces/     yuz rasmlari
data/crops/<id>/bodies/    tana rasmlari
```

## Test

```bash
PYTHONPATH=. .venv/bin/python tests/test_pipeline.py
```

## Huquqiy eslatma

Yuz va tana tasvirlari biometrik shaxsiy ma'lumot hisoblanadi. O'zbekiston Respublikasining
"Shaxsga doir ma'lumotlar to'g'risida"gi qonuni (va boshqa davlatlarda GDPR kabi qonunlar) bunday
ma'lumotlarni yig'ish uchun odatda subyektning roziligini, kuzatuv haqida ogohlantirishni (masalan,
kirishdagi belgi) va ma'lumotlarni himoyalashni talab qiladi. Tizimni ishlatishdan oldin talablarni
tekshiring; `delete` buyrug'i odamning barcha ma'lumotlarini butunlay o'chiradi.
