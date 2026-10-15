import sys
from pathlib import Path


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[2]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from app.core import faiss_search, embedding_store

    count = embedding_store.count()
    print(f"Building FAISS index for {count} vectors...", flush=True)
    if count <= 0:
        print("No embeddings available.", flush=True)
        return 2
    ok = faiss_search.build_index()
    print("FAISS index built." if ok else "FAISS build failed.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
