import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RECALIX indexing outside the API process")
    parser.add_argument("directory")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[2]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from app.db.database import init_db
    from app.core.indexer import run_indexing

    init_db()
    run_indexing(args.directory, reindex=args.reindex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
