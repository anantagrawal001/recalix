# RECALIX V10 — Training System Explained

> Created: 2026-06-04 | Session: Setup & Training Deep-Dive

---

## Architecture Overview

RECALIX uses a **two-stage search system**:

1. **CLIP Model (ViT-B-32)** — Frozen, 151M parameters. Converts text queries and images into 512-dimensional embeddings. This is NEVER trained by RECALIX.

2. **QueryConditionedFusion** — Trainable, 173,508 parameters. A lightweight MLP that learns query-adaptive weights over 4 scoring signals.

```
Search Query → CLIP Text Encoder (frozen) → 512-dim embedding
                                                    ↓
                                        QueryConditionedFusion (trained)
                                                    ↓
                                        4 weights [visual, ocr, filename, structural]
                                                    ↓
                            final_score = w₁×visual + w₂×ocr + w₃×filename + w₄×structural
```

---

## The 4 Scoring Signals

| Signal | What it measures | Example |
|---|---|---|
| **Visual** | CLIP cosine similarity between text query embedding and image embedding | High for "sunset" query + actual sunset photo |
| **OCR text** | Word overlap between query and text found IN the image via EasyOCR | High for "invoice" query + image containing invoice text |
| **Filename** | Word overlap between query and the image's filename tokens | High for "beach" query + file named `beach_sunset.jpg` |
| **Structural** | Heuristic: document/screenshot detection, image size | Penalizes tiny icons, boosts documents for doc-type queries |

---

## What the Model Learns

### Without Training (Fixed Weights)
```
final_score = 0.55 × visual + 0.30 × text + 0.10 × filename + 0.05 × structural
```
Same weights for ALL queries — not optimal.

### With Training (Adaptive Weights)
The model learns to assign **different weights per query type**:

| Query | Visual | OCR | Filename | Structural |
|---|---|---|---|---|
| "sunset on beach" | **0.82** | 0.05 | 0.08 | 0.05 |
| "invoice amount" | 0.15 | **0.72** | 0.08 | 0.05 |
| "whatsapp photo" | 0.20 | 0.10 | **0.65** | 0.05 |
| "document scan" | 0.15 | 0.40 | 0.10 | **0.35** |

---

## Model Architecture (173,508 Parameters)

```
Input (512) → Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.1)
           → Linear(256, 128) → LayerNorm(128) → GELU → Dropout(0.1)
           → Linear(128, 64)  → GELU
           → Linear(64, 4)    → Softmax (temperature=1.0)
           → Output: 4 weights summing to 1.0
```

| Layer | Shape | Parameters |
|---|---|---|
| Linear | 512 → 256 | 131,328 |
| LayerNorm | 256 | 512 |
| Linear | 256 → 128 | 32,896 |
| LayerNorm | 128 | 256 |
| Linear | 128 → 64 | 8,256 |
| Linear | 64 → 4 | 260 |
| **Total** | | **173,508** |

---

## Training Pipeline

### Step 1: Query Generation
For each image, extract text queries from 3 sources:
- **Captions**: Pre-generated image descriptions (if available)
- **OCR text**: Text extracted from inside the image by EasyOCR
- **Filenames**: Meaningful words from the file name (excludes generic tokens like "IMG", "screenshot", etc.)

### Step 2: Dataset Construction (Triplets)
For each query:
- **Positive**: The image the query was extracted FROM
- **Negative**: A different image (random or semi-hard mined)
- Compute 4 signals for both positive and negative images

### Step 3: Training (Triplet Ranking Loss)
```
loss = max(0, neg_score - pos_score + margin)
margin = 0.4
```
Forces the model to score positive images at least 0.4 higher than negatives.

### Step 4: Evaluation
Pairwise accuracy on validation set — fraction where pos_score > neg_score.
- Random baseline = 0.50
- Current model = **0.942** (94.2%)

---

## Why ~100K Examples from 233K Images?

1. **80/10/10 split**: Only ~186K images used for training
2. **Not all images produce queries**: `IMG_20240315.jpg` with no OCR text = zero queries
3. **Source balancing**: Each source (caption, OCR, filename) capped at ~62K to prevent domination
4. **Result**: ~100K balanced training examples

---

## Training Modes

| Mode | Negatives | Difficulty | When to Use |
|---|---|---|---|
| **basic** | 100% random | Easy | First time, small datasets |
| **semi_hard** ⭐ | 50% random + 50% semi-hard | Medium | **Recommended for most cases** |
| **hard** | Hard negative mining | Hard | Large diverse datasets, after basic/semi_hard converges |

---

## Re-Training Value

### When it helps:
- ✅ After indexing **new images** (new queries, new negatives)
- ✅ Switching training mode (basic → semi_hard → hard)
- ✅ First time training (fixed weights → learned weights)

### When it doesn't help:
- ❌ Same images, same mode — queries are deterministic, examples are cached
- ❌ Already at 94.2% accuracy — diminishing returns
- ❌ Only 173K parameters, converges fully in ~20 epochs (~20 seconds)

---

## Key Files

| File | Purpose |
|---|---|
| `backend/app/core/fusion_model.py` | QueryConditionedFusion model definition |
| `training/modes/base.py` | Dataset construction, query extraction, evaluation |
| `training/modes/semi_hard.py` | Semi-hard negative training (recommended) |
| `training/modes/hard.py` | Hard negative mining training |
| `training/modes/basic.py` | Basic random negative training |
| `backend/app/core/hybrid.py` | Uses trained model at search time |
| `backend/data/fusion_model.pt` | Saved model checkpoint |
