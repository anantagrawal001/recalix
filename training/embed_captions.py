"""
RECALIX Caption Embedding
=========================

Embeds Flickr30K + COCO captions with CLIP
Stores embeddings in metadata.db

Artifacts logged to:
training/artifacts/caption_runs/
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

import torch
import open_clip
import numpy as np


RUN = datetime.now().strftime("%Y%m%d_%H%M%S")

ART = Path("C:/v9/training/artifacts/caption_runs")
ART.mkdir(parents=True, exist_ok=True)

LOG = ART / f"caption_embed_{RUN}.log"


def log(x):
    print(x)
    with open(LOG,"a",encoding="utf8") as f:
        f.write(x+"\n")


DATA = Path("C:/v9/backend/data")
DB = DATA / "metadata.db"

FLICKR = Path("D:/datasets/flickr30k/dataset_flickr30k.json")
COCO = Path("D:/datasets/coco/annotations/captions_train2017.json")


device = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Device: {device}")

model, _, tokenizer = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

conn = sqlite3.connect(DB)

conn.execute("""
CREATE TABLE IF NOT EXISTS captions (
id INTEGER PRIMARY KEY AUTOINCREMENT,
image_id INTEGER,
caption TEXT,
embedding BLOB
)
""")

def embed(text):

    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        e = model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)

    return e.cpu().numpy()[0]


def process_flickr():

    log("Processing Flickr30k")

    data = json.loads(FLICKR.read_text())

    rows = conn.execute("SELECT id,path FROM media").fetchall()

    pathmap = {Path(r[1]).name:r[0] for r in rows}

    inserted = 0

    for img in data["images"]:

        name = img["filename"]

        if name not in pathmap:
            continue

        iid = pathmap[name]

        for s in img["sentences"]:

            cap = s["raw"]

            emb = embed(cap)

            conn.execute(
                "INSERT INTO captions(image_id,caption,embedding) VALUES (?,?,?)",
                (iid,cap,emb.tobytes())
            )

            inserted+=1

    conn.commit()

    log(f"Flickr captions stored: {inserted}")


def process_coco():

    log("Processing COCO")

    data = json.loads(COCO.read_text())

    rows = conn.execute("SELECT id,path FROM media").fetchall()

    pathmap = {Path(r[1]).name:r[0] for r in rows}

    inserted = 0

    for ann in data["annotations"]:

        fname = str(ann["image_id"]).zfill(12)+".jpg"

        if fname not in pathmap:
            continue

        iid = pathmap[fname]

        cap = ann["caption"]

        emb = embed(cap)

        conn.execute(
            "INSERT INTO captions(image_id,caption,embedding) VALUES (?,?,?)",
            (iid,cap,emb.tobytes())
        )

        inserted+=1

    conn.commit()

    log(f"COCO captions stored: {inserted}")


process_flickr()
process_coco()

log("Caption embedding complete")
