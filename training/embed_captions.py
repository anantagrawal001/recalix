"""
RECALIX Caption Embedding (V10 FINAL)
====================================

✔ Correct paths
✔ Correct tokenizer (FIXED crash)
✔ Filename-based matching
✔ Batch embedding (fast)
✔ Duplicate-safe inserts
✔ Match-rate validation

Run:
python training/embed_captions.py
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

import torch
import open_clip


# ----------------------------
# PATH SETUP
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]

ART = ROOT / "training/artifacts/caption_runs"
ART.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "backend/data"
DB = DATA / "metadata.db"

FLICKR = Path("D:/datasets/flickr30k/dataset_flickr30k.json")
COCO = Path("D:/datasets/coco/annotations/captions_train2017.json")


RUN = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG = ART / f"caption_embed_{RUN}.log"


def log(x):
    print(x)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(x + "\n")


# ----------------------------
# DEVICE
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

log(f"ROOT: {ROOT}")
log(f"DB: {DB}")
log(f"Device: {device}")


# ----------------------------
# MODEL (FIXED TOKENIZER)
# ----------------------------
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")  # ✅ FIX

model = model.to(device)
model.eval()


# ----------------------------
# DB SETUP
# ----------------------------
conn = sqlite3.connect(DB)

conn.execute("""
CREATE TABLE IF NOT EXISTS captions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    caption TEXT,
    embedding BLOB,
    source TEXT,
    UNIQUE(image_id, caption)
)
""")

rows = conn.execute("SELECT id, filename FROM media").fetchall()
pathmap = {r[1]: r[0] for r in rows}

log(f"DB images loaded: {len(pathmap)}")
log(f"Sample DB filenames: {list(pathmap.keys())[:5]}")


# ----------------------------
# EMBEDDING (BATCH)
# ----------------------------
def embed_batch(texts, batch_size=64):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        tokens = tokenizer(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            e = model.encode_text(tokens)
            e = e / e.norm(dim=-1, keepdim=True)

        all_embeddings.extend(e.cpu().numpy())

    return all_embeddings


# ----------------------------
# FLICKR
# ----------------------------
def process_flickr():
    log("\n--- Processing Flickr30k ---")

    data = json.loads(FLICKR.read_text())

    total = 0
    matched = 0
    inserts = []

    for img in data["images"]:
        total += 1

        name = img["filename"]

        if name not in pathmap:
            continue

        matched += 1
        iid = pathmap[name]

        for s in img["sentences"]:
            inserts.append((iid, s["raw"], "flickr"))

    log(f"Flickr match rate: {matched}/{total}")

    if matched == 0:
        log("❌ ZERO matches — STOP")
        return

    captions = [c[1] for c in inserts]
    embeddings = embed_batch(captions)

    inserted = 0

    for (iid, cap, src), emb in zip(inserts, embeddings):
        conn.execute(
            "INSERT OR IGNORE INTO captions(image_id, caption, embedding, source) VALUES (?,?,?,?)",
            (iid, cap, emb.tobytes(), src)
        )
        inserted += 1

    conn.commit()

    log(f"Flickr captions inserted: {inserted}")


# ----------------------------
# COCO
# ----------------------------
def process_coco():
    log("\n--- Processing COCO ---")

    data = json.loads(COCO.read_text())

    total = 0
    matched = 0
    inserts = []

    for ann in data["annotations"]:
        total += 1

        fname = str(ann["image_id"]).zfill(12) + ".jpg"

        if fname not in pathmap:
            continue

        matched += 1
        iid = pathmap[fname]

        inserts.append((iid, ann["caption"], "coco"))

    log(f"COCO match rate: {matched}/{total}")

    if matched == 0:
        log("❌ ZERO matches — STOP")
        return

    captions = [c[1] for c in inserts]
    embeddings = embed_batch(captions)

    inserted = 0

    for (iid, cap, src), emb in zip(inserts, embeddings):
        conn.execute(
            "INSERT OR IGNORE INTO captions(image_id, caption, embedding, source) VALUES (?,?,?,?)",
            (iid, cap, emb.tobytes(), src)
        )
        inserted += 1

    conn.commit()

    log(f"COCO captions inserted: {inserted}")


# ----------------------------
# RUN
# ----------------------------
process_flickr()
process_coco()


# ----------------------------
# FINAL CHECK
# ----------------------------
count = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
log(f"\nTotal captions in DB: {count}")

sample = conn.execute("SELECT caption FROM captions LIMIT 5").fetchall()
log(f"Sample captions: {sample}")

log("\n✔ Caption embedding COMPLETE")