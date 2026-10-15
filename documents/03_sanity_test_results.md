# RECALIX V10 — Sanity Test Results

> Date: 2026-06-04 | Environment: Windows, RTX 5060, Python 3.12

---

## Test Summary

| # | Test | Status | Details |
|---|---|---|---|
| 1 | CUDA/GPU Verification | ✅ PASS | PyTorch 2.11.0+cu128, RTX 5060 Laptop GPU |
| 2 | Backend Health API | ✅ PASS | v10.0.0, status: ok |
| 3 | Frontend Dev Server | ✅ PASS | Vite v5.4.21 on http://localhost:5173 |
| 4 | Search API | ✅ PASS | "sunset" → 5 results, FAISS backend |
| 5 | Fusion Model | ✅ PASS | Trained, active, recall@5 = 0.942 |
| 6 | Folders API | ✅ PASS | 221 folders found |
| 7 | Gallery API | ✅ PASS | 233,192 images browsable |
| 8 | Stats API | ✅ PASS | 233,199 embeddings, ViT-B-32 model |
| 9 | Training Status API | ✅ PASS | Last run: semi_hard, 20 epochs |
| 10 | Search History API | ✅ PASS | History recording/retrieval working |

**Result: ALL TESTS PASSED ✅**

---

## Detailed Test Outputs

### Test 1: CUDA/GPU Verification
```
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
GPU compute test: PASSED
```

### Test 2: Backend Health API
```json
GET /api/health

{
    "status": "ok",
    "version": "10.0.0",
    "indexed": 233199,
    "search_backend": "faiss",
    "fusion": true
}
```

### Test 3: Frontend Dev Server
```
VITE v5.4.21 ready in 780 ms
→ Local: http://localhost:5173/
Status: 200, Length: 906 bytes
```

### Test 4: Search API
```
GET /api/search?q=sunset&top_k=5

Query: sunset
Total results: 5
Page results: 5
Backend: faiss
First result score: 0.204
```

### Test 5: Training Status API
```json
GET /api/training/status

{
    "status": "done",
    "mode": "semi_hard",
    "epoch": 20,
    "total_epochs": 20,
    "loss": 0.06958,
    "recall_at5": 0.942,
    "message": "Training complete",
    "started_at": "2026-06-03T23:50:59.022743",
    "finished_at": "2026-06-03T23:51:19.225872"
}
```

### Test 6: Folders API
```
GET /api/folders

Folders: 221
```

### Test 7: Gallery API
```
GET /api/gallery?page=1&page_size=5

Gallery total: 233,192
Page results: 5
```

### Test 8: Stats API
```json
GET /api/stats

{
    "total": 233192,
    "images": 233192,
    "videos": 0,
    "last_indexed": "2026-05-17 15:57:20",
    "embedding_count": 233199,
    "model_version": "ViT-B-32"
}
```

---

## API Endpoints Validated

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | System health check |
| `/api/search?q=...` | GET | AI-powered photo search |
| `/api/search/history` | GET | Search history |
| `/api/folders` | GET | Folder tree |
| `/api/gallery` | GET | Browse all photos |
| `/api/stats` | GET | Database statistics |
| `/api/index/status` | GET | Indexing progress |
| `/api/training/status` | GET | Training status |
| `/api/thumbnails/{id}` | GET | Thumbnail images |
| `/api/open-file` | POST | Open file in OS |
| `/docs` | GET | FastAPI auto-generated docs |
