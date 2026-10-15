"""
RECALIX project maintenance agent.

Continuously audits the local project for common RECALIX failures and applies
safe fixes. It intentionally avoids broad code rewrites. Risky findings are
logged for a developer instead of being changed automatically.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
TRAINING = ROOT / "training"
DATA = BACKEND / "data"
LOG_DIR = DATA / "agent_logs"
STATUS_PATH = DATA / "training_status.json"
INDEX_STATUS_PATH = DATA / "index_status.json"
PYTHON = ROOT / "venv_gpu" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

LONG_TASK_PATTERNS = (
    "training\\train.py",
    "training/train.py",
    "app.core.index_cli",
    "app.core.faiss_cli",
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Agent:
    def __init__(self, interval: int, fix: bool, build_interval: int):
        self.interval = interval
        self.fix = fix
        self.build_interval = build_interval
        self.last_build_check = 0.0
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / "project_agent.log"

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_forever(self) -> None:
        self.log(f"Agent started. root={ROOT} fix={self.fix} interval={self.interval}s")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                self.log("Agent stopped by user.")
                raise
            except Exception as e:
                self.log(f"AGENT ERROR: {type(e).__name__}: {e}")
            time.sleep(self.interval)

    def run_once(self) -> None:
        self.fix_duplicate_long_tasks()
        self.fix_stale_status_file(STATUS_PATH, "training")
        self.fix_stale_status_file(INDEX_STATUS_PATH, "index")
        self.check_training_log_visibility()
        self.check_python_compile()
        self.check_frontend_build_periodically()

    def run_cmd(
        self,
        cmd: List[str],
        cwd: Path = ROOT,
        timeout: int = 120,
        shell: bool = False,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd if not shell else " ".join(cmd),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            shell=shell,
        )

    def list_processes(self) -> List[Dict[str, Any]]:
        if os.name != "nt":
            return []
        ps = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
        )
        cp = self.run_cmd(["powershell", "-NoProfile", "-Command", ps], timeout=30)
        if cp.returncode != 0 or not cp.stdout.strip():
            self.log(f"Process scan failed: {cp.stdout.strip()}")
            return []
        try:
            data = json.loads(cp.stdout)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            return [data]
        return data or []

    def task_processes(self) -> List[Dict[str, Any]]:
        out = []
        for proc in self.list_processes():
            cmd = proc.get("CommandLine") or ""
            if str(ROOT) not in cmd:
                continue
            if any(pat in cmd for pat in LONG_TASK_PATTERNS):
                out.append(proc)
        return out

    def fix_duplicate_long_tasks(self) -> None:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for proc in self.task_processes():
            cmd = proc.get("CommandLine") or ""
            key = self.task_key(cmd)
            groups.setdefault(key, []).append(proc)

        for key, procs in groups.items():
            if len(procs) <= 1:
                continue
            procs.sort(key=lambda p: self.prefers_process(p), reverse=True)
            keep = procs[0]
            kill = procs[1:]
            self.log(
                f"Duplicate task '{key}' detected. Keeping PID {keep.get('ProcessId')}; "
                f"stopping {[p.get('ProcessId') for p in kill]}"
            )
            if self.fix:
                for proc in kill:
                    self.stop_process(int(proc["ProcessId"]))

    def task_key(self, cmd: str) -> str:
        run_id = re.search(r"--run_id\s+([0-9_]+)", cmd)
        if run_id:
            return f"training:{run_id.group(1)}"
        if "app.core.index_cli" in cmd:
            return "index"
        if "app.core.faiss_cli" in cmd:
            return "faiss"
        return re.sub(r"\s+", " ", cmd)

    def prefers_process(self, proc: Dict[str, Any]) -> int:
        cmd = proc.get("CommandLine") or ""
        score = 0
        if "venv_gpu" in cmd:
            score += 10
        if "training\\train.py" in cmd or "training/train.py" in cmd:
            score += 3
        return score

    def stop_process(self, pid: int) -> None:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log(f"Stopped PID {pid}")
        except Exception as e:
            self.log(f"Failed stopping PID {pid}: {e}")

    def pid_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        if os.name == "nt":
            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not process:
                return False
            try:
                code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(code)):
                    return False
                return code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(process)
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    def load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.log(f"Invalid JSON: {path}: {e}")
            return None

    def write_json(self, path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def fix_stale_status_file(self, path: Path, label: str) -> None:
        data = self.load_json(path)
        if not data or data.get("status") not in {"running", "scanning", "indexing"}:
            return
        pid = data.get("pid")
        if self.pid_alive(pid):
            return
        message = f"{label} status was running but PID {pid} is not alive"
        self.log(message)
        if self.fix:
            data["status"] = "error"
            data["message"] = message
            data["finished_at"] = datetime.now().isoformat()
            data["pid"] = None
            self.write_json(path, data)

    def check_training_log_visibility(self) -> None:
        data = self.load_json(STATUS_PATH)
        if not data or data.get("status") != "running":
            return
        log_path = data.get("terminal_log_path")
        if not log_path:
            self.log("Training is running but terminal_log_path is missing.")
            return
        path = Path(log_path)
        if not path.exists():
            self.log(f"Training log does not exist yet: {path}")
            return
        age = time.time() - path.stat().st_mtime
        if age > 120:
            self.log(
                f"Training log has not changed for {age:.0f}s. "
                "This may be a slow hard-mining phase or a stalled process."
            )

    def check_python_compile(self) -> None:
        cp = self.run_cmd(
            [str(PYTHON), "-m", "compileall", "-q", "backend\\app", "training"],
            timeout=120,
        )
        if cp.returncode != 0:
            self.log("Python compile check failed:")
            self.log(cp.stdout.strip()[-4000:])

    def check_frontend_build_periodically(self) -> None:
        if time.time() - self.last_build_check < self.build_interval:
            return
        self.last_build_check = time.time()
        package_json = FRONTEND / "package.json"
        if not package_json.exists():
            return
        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm:
            self.log("Frontend build skipped: npm was not found on PATH.")
            return
        cp = self.run_cmd([npm, "run", "build"], cwd=FRONTEND, timeout=180)
        if cp.returncode != 0:
            self.log("Frontend build failed:")
            self.log(cp.stdout.strip()[-4000:])
        else:
            self.log("Build checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously audit RECALIX and apply safe fixes.")
    parser.add_argument("--once", action="store_true", help="Run one audit pass and exit.")
    parser.add_argument("--no-fix", action="store_true", help="Report only; do not stop tasks or edit status files.")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between audit passes.")
    parser.add_argument("--build-interval", type=int, default=300, help="Seconds between frontend build checks.")
    args = parser.parse_args()

    agent = Agent(interval=max(args.interval, 5), fix=not args.no_fix, build_interval=max(args.build_interval, 60))
    if args.once:
        agent.run_once()
        return 0
    agent.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
