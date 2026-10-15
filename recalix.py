"""
RECALIX V10 — Universal Launcher
==================================
Single-file launcher: detects Python, discovers all GPU hardware (NVIDIA, AMD,
Intel), determines architecture/generation, creates venv, installs the optimal
PyTorch build, starts the backend, opens the browser.

Compiled to recalix.exe via PyInstaller.
Works on any Windows machine — NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU-only.

First run:  full hardware scan + install (~5 min GPU, ~3 min CPU)
Later runs: skips install, launches in seconds
"""

import os
import re
import sys
import subprocess
import webbrowser
import time
import socket
import shutil
import json
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
VENV_NAME = "venv_gpu"
BACKEND_PORT = 8000
PYTHON_MIN = (3, 10)
PYTHON_MAX = (3, 12)

# NVIDIA compute capability → architecture name + recommended CUDA wheel
NVIDIA_ARCH_MAP = {
    # (sm_major, sm_minor) → (architecture, generation, [cuda_wheels in priority order])
    (5, 0): ("Maxwell",        "GTX 900 series",    ["cu121"]),
    (5, 2): ("Maxwell",        "GTX 900 series",    ["cu121"]),
    (6, 0): ("Pascal",         "GTX 1000 series",   ["cu121"]),
    (6, 1): ("Pascal",         "GTX 1000 series",   ["cu121"]),
    (7, 0): ("Volta",          "V100 / Titan V",    ["cu121", "cu124"]),
    (7, 5): ("Turing",         "RTX 2000 / GTX 1600 series", ["cu121", "cu124"]),
    (8, 0): ("Ampere",         "A100 / RTX 3000 series",     ["cu121", "cu124"]),
    (8, 6): ("Ampere",         "RTX 3000 series",            ["cu121", "cu124"]),
    (8, 9): ("Ada Lovelace",   "RTX 4000 series",            ["cu124", "cu126"]),
    (9, 0): ("Hopper",         "H100 / H200",                ["cu124", "cu126", "cu128"]),
    (10, 0): ("Blackwell",     "B100 / B200",                ["cu126", "cu128"]),
    (12, 0): ("Blackwell",     "RTX 5000 series",            ["cu128"]),
}

# AMD GPU family detection keywords
AMD_FAMILIES = {
    "RDNA 3":   ["7900", "7800", "7700", "7600", "gfx1100", "gfx1101", "gfx1102"],
    "RDNA 2":   ["6900", "6800", "6700", "6600", "6500", "gfx1030", "gfx1031", "gfx1032"],
    "RDNA 1":   ["5700", "5600", "5500", "gfx1010", "gfx1011", "gfx1012"],
    "Vega":     ["vega", "VII", "gfx900", "gfx906"],
}

# ── Console Helpers ───────────────────────────────────────────────────────────

def banner(msg):
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print(f"{'=' * 64}\n")

def step(n, msg):
    print(f"  [{n}] {msg}")

def ok(msg):
    print(f"      + {msg}")

def info(msg):
    print(f"      | {msg}")

def warn(msg):
    print(f"      ! {msg}")

def fail(msg):
    print(f"\n  X ERROR: {msg}")
    input("\n  Press Enter to exit...")
    sys.exit(1)


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def find_root():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# ── Python Detection ──────────────────────────────────────────────────────────

def find_python():
    """Find Python 3.10–3.12 on the system."""
    # 1. py launcher (most reliable on Windows)
    for ver in ["3.12", "3.11", "3.10"]:
        try:
            r = subprocess.run(["py", f"-{ver}", "--version"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ok(f"Found via py launcher: {r.stdout.strip()}")
                return f"py -{ver}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # 2. python on PATH
    for cmd in ["python3", "python"]:
        try:
            r = subprocess.run(
                [cmd, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                parts = r.stdout.strip().split(".")
                major, minor = int(parts[0]), int(parts[1])
                if PYTHON_MIN <= (major, minor) <= PYTHON_MAX:
                    ok(f"Found on PATH: Python {major}.{minor}")
                    return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # 3. Common install paths
    search_dirs = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python",
    ]
    for v in [312, 311, 310]:
        search_dirs.append(Path(f"C:/Python{v}"))
        search_dirs.append(Path(f"C:/Program Files/Python{v}"))

    for base in search_dirs:
        candidates = [base / "python.exe"]
        if base.is_dir():
            candidates += [sub / "python.exe" for sub in sorted(base.iterdir(), reverse=True)]
        for exe in candidates:
            if not exe.exists():
                continue
            try:
                r = subprocess.run(
                    [str(exe), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    parts = r.stdout.strip().split(".")
                    major, minor = int(parts[0]), int(parts[1])
                    if PYTHON_MIN <= (major, minor) <= PYTHON_MAX:
                        ok(f"Found at: {exe}")
                        return str(exe)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# GPU HARDWARE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _query_wmi_gpus():
    """
    Enumerate ALL video adapters on the system via WMI.
    Tries PowerShell Get-CimInstance first (modern, reliable on Win10/11),
    then falls back to wmic (deprecated but still works on older builds).
    Returns a list of dicts with name, vendor, driver_version, vram_bytes.
    """
    gpus = []

    # ── Method 1: PowerShell Get-CimInstance (preferred) ─────────────────
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance -ClassName Win32_VideoController | "
             "Select-Object Name, AdapterCompatibility, DriverVersion, AdapterRAM | "
             "ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout.strip())
            # PowerShell returns a dict for single GPU, list for multiple
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                name = entry.get("Name", "")
                if not name:
                    continue
                compat = (entry.get("AdapterCompatibility") or "").lower()

                if "nvidia" in compat or "nvidia" in name.lower():
                    vendor = "NVIDIA"
                elif "amd" in compat or "advanced micro" in compat or "ati" in compat or "radeon" in name.lower():
                    vendor = "AMD"
                elif "intel" in compat or "intel" in name.lower():
                    vendor = "Intel"
                else:
                    vendor = "Unknown"

                vram = 0
                try:
                    vram = int(entry.get("AdapterRAM") or 0)
                except (ValueError, TypeError):
                    pass

                gpus.append({
                    "name": name,
                    "vendor": vendor,
                    "driver_version": entry.get("DriverVersion", ""),
                    "vram_bytes": vram,
                })
            if gpus:
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # ── Method 2: wmic fallback (older Windows) ──────────────────────────
    try:
        r = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get",
             "Name,AdapterCompatibility,DriverVersion,AdapterRAM",
             "/format:csv"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
            if len(lines) > 1:
                headers = [h.strip().lower() for h in lines[0].split(",")]
                for line in lines[1:]:
                    cols = [c.strip() for c in line.split(",")]
                    if len(cols) < len(headers):
                        continue
                    row = dict(zip(headers, cols))
                    name = row.get("name", "")
                    if not name:
                        continue
                    vendor_raw = (row.get("adaptercompatibility") or "").lower()

                    if "nvidia" in vendor_raw or "nvidia" in name.lower():
                        vendor = "NVIDIA"
                    elif "amd" in vendor_raw or "advanced micro" in vendor_raw or "ati" in vendor_raw or "radeon" in name.lower():
                        vendor = "AMD"
                    elif "intel" in vendor_raw or "intel" in name.lower():
                        vendor = "Intel"
                    else:
                        vendor = "Unknown"

                    vram = 0
                    try:
                        vram = int(row.get("adapterram", "0") or "0")
                    except ValueError:
                        pass

                    gpus.append({
                        "name": name,
                        "vendor": vendor,
                        "driver_version": row.get("driverversion", ""),
                        "vram_bytes": vram,
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpus


def _nvidia_detailed_info():
    """
    Use nvidia-smi to get detailed NVIDIA GPU info:
    name, driver, compute capability, VRAM, max CUDA version.
    """
    result = {
        "available": False,
        "name": None,
        "driver_version": None,
        "cuda_version": None,
        "compute_capability": None,
        "architecture": None,
        "generation": None,
        "vram_gb": None,
        "recommended_cuda_wheels": [],
    }

    # 1. Basic query
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return result
        line = r.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        result["available"] = True
        result["name"] = parts[0] if len(parts) > 0 else "Unknown"
        result["driver_version"] = parts[1] if len(parts) > 1 else None
        try:
            result["vram_gb"] = round(int(parts[2]) / 1024, 1) if len(parts) > 2 else None
        except (ValueError, IndexError):
            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return result

    # 2. CUDA version from nvidia-smi header
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout)
            if match:
                result["cuda_version"] = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Compute capability via nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            cc_str = r.stdout.strip().split("\n")[0].strip()
            parts = cc_str.split(".")
            if len(parts) == 2:
                sm = (int(parts[0]), int(parts[1]))
                result["compute_capability"] = f"sm_{sm[0]}{sm[1]}"

                # Look up architecture
                if sm in NVIDIA_ARCH_MAP:
                    arch, gen, wheels = NVIDIA_ARCH_MAP[sm]
                    result["architecture"] = arch
                    result["generation"] = gen
                    result["recommended_cuda_wheels"] = wheels
                else:
                    # Fallback: match by major version
                    for (maj, _), (arch, gen, wheels) in sorted(NVIDIA_ARCH_MAP.items(), reverse=True):
                        if sm[0] == maj:
                            result["architecture"] = arch
                            result["generation"] = gen
                            result["recommended_cuda_wheels"] = wheels
                            break
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # If no wheels determined yet, build a list based on CUDA version
    if not result["recommended_cuda_wheels"] and result["cuda_version"]:
        cuda_major_minor = result["cuda_version"].replace(".", "")
        result["recommended_cuda_wheels"] = [f"cu{cuda_major_minor}"]

    return result


def _amd_classify(gpu_name):
    """Identify AMD GPU family from its name."""
    name_lower = gpu_name.lower()
    for family, keywords in AMD_FAMILIES.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return family
    return "Unknown AMD family"


def _intel_classify(gpu_name):
    """Identify Intel GPU type."""
    name_lower = gpu_name.lower()
    if "arc" in name_lower:
        if any(x in name_lower for x in ["a770", "a750", "a580"]):
            return "Arc Alchemist (high-end)"
        if any(x in name_lower for x in ["a380", "a310"]):
            return "Arc Alchemist (entry)"
        return "Arc series"
    if "iris" in name_lower:
        return "Iris Xe (integrated)"
    if "uhd" in name_lower:
        return "UHD (integrated)"
    return "Intel integrated"


def discover_hardware():
    """
    Complete hardware discovery:
      1. WMI scan → find ALL GPUs (NVIDIA, AMD, Intel, integrated)
      2. nvidia-smi  → NVIDIA details (architecture, compute cap, CUDA version)
      3. Classify each GPU and determine the best PyTorch install strategy

    Returns a dict with:
      gpus: list of all discovered GPUs with details
      best_gpu: the recommended GPU for compute (or None)
      install_strategy: "nvidia_cuda" | "amd_rocm" | "intel_xpu" | "cpu"
      nvidia: detailed NVIDIA info (if applicable)
    """
    result = {
        "gpus": [],
        "best_gpu": None,
        "install_strategy": "cpu",
        "nvidia": None,
    }

    # ── 1. WMI: Find all GPUs ────────────────────────────────────────────
    wmi_gpus = _query_wmi_gpus()
    all_gpus = []

    for gpu in wmi_gpus:
        entry = {
            "name": gpu["name"],
            "vendor": gpu["vendor"],
            "driver_version": gpu["driver_version"],
            "vram_mb": gpu["vram_bytes"] // (1024 * 1024) if gpu["vram_bytes"] else 0,
            "type": "discrete" if gpu["vram_bytes"] and gpu["vram_bytes"] > 512 * 1024 * 1024 else "integrated",
            "family": None,
            "compute_support": None,
        }

        if gpu["vendor"] == "AMD":
            entry["family"] = _amd_classify(gpu["name"])
            entry["compute_support"] = "ROCm (Linux only — CPU fallback on Windows)"
        elif gpu["vendor"] == "Intel":
            entry["family"] = _intel_classify(gpu["name"])
            entry["compute_support"] = "XPU (experimental)" if "arc" in gpu["name"].lower() else "None (integrated)"
        elif gpu["vendor"] == "NVIDIA":
            entry["compute_support"] = "CUDA"  # will be enriched below

        all_gpus.append(entry)

    result["gpus"] = all_gpus

    # ── 2. NVIDIA detailed scan ──────────────────────────────────────────
    nvidia_info = _nvidia_detailed_info()
    result["nvidia"] = nvidia_info

    if nvidia_info["available"]:
        # Enrich the WMI entry for this NVIDIA GPU
        for gpu in all_gpus:
            if gpu["vendor"] == "NVIDIA":
                gpu["architecture"] = nvidia_info.get("architecture")
                gpu["generation"] = nvidia_info.get("generation")
                gpu["compute_capability"] = nvidia_info.get("compute_capability")
                gpu["cuda_version"] = nvidia_info.get("cuda_version")
                if nvidia_info.get("vram_gb"):
                    gpu["vram_mb"] = int(nvidia_info["vram_gb"] * 1024)
                break

    # ── 3. Determine best strategy ───────────────────────────────────────
    # Priority: NVIDIA CUDA > AMD ROCm (Linux) > Intel XPU > CPU
    nvidia_gpus = [g for g in all_gpus if g["vendor"] == "NVIDIA" and g["type"] == "discrete"]
    amd_gpus    = [g for g in all_gpus if g["vendor"] == "AMD" and g["type"] == "discrete"]
    intel_dgpu  = [g for g in all_gpus if g["vendor"] == "Intel" and "arc" in g["name"].lower()]

    if nvidia_gpus and nvidia_info["available"]:
        result["best_gpu"] = nvidia_gpus[0]
        result["install_strategy"] = "nvidia_cuda"
    elif amd_gpus:
        result["best_gpu"] = amd_gpus[0]
        # ROCm on Windows is not officially supported by PyTorch
        # We detect it, inform the user, and fall back to DirectML or CPU
        result["install_strategy"] = "amd_directml"
    elif intel_dgpu:
        result["best_gpu"] = intel_dgpu[0]
        result["install_strategy"] = "intel_xpu"
    else:
        # CPU-only (or only integrated GPU)
        result["install_strategy"] = "cpu"

    return result


def print_hardware_report(hw):
    """Print a formatted hardware discovery report."""
    gpus = hw["gpus"]
    if not gpus:
        ok("No dedicated GPU detected — will use CPU")
        return

    ok(f"Found {len(gpus)} display adapter(s):")
    print()

    for i, gpu in enumerate(gpus, 1):
        prefix = ">>>" if gpu == hw["best_gpu"] else "   "
        vram_str = f"{gpu['vram_mb']} MB" if gpu['vram_mb'] else "shared"

        info(f"{prefix} [{i}] {gpu['name']}")
        info(f"      Vendor: {gpu['vendor']}  |  Type: {gpu['type']}  |  VRAM: {vram_str}")

        if gpu.get("architecture"):
            info(f"      Architecture: {gpu['architecture']} ({gpu.get('generation', '')})")
        if gpu.get("compute_capability"):
            info(f"      Compute capability: {gpu['compute_capability']}")
        if gpu.get("cuda_version"):
            info(f"      Max CUDA: {gpu['cuda_version']}")
        if gpu.get("family"):
            info(f"      Family: {gpu['family']}")
        if gpu.get("compute_support"):
            info(f"      PyTorch support: {gpu['compute_support']}")
        print()

    # Strategy summary
    strategy = hw["install_strategy"]
    best = hw["best_gpu"]
    if strategy == "nvidia_cuda":
        nvidia = hw["nvidia"]
        wheels = nvidia.get("recommended_cuda_wheels", [])
        ok(f"Install strategy: NVIDIA CUDA")
        if nvidia.get("architecture"):
            ok(f"  {nvidia['architecture']} architecture requires: {', '.join(wheels) if wheels else 'auto-detect'}")
    elif strategy == "amd_directml":
        ok(f"Install strategy: AMD GPU detected → torch-directml (Windows)")
        info(f"  Note: PyTorch ROCm is Linux-only. Using DirectML for Windows GPU acceleration.")
        info(f"  Performance: ~60-80% of native ROCm, but works on Windows.")
    elif strategy == "intel_xpu":
        ok(f"Install strategy: Intel Arc GPU detected → Intel Extension for PyTorch")
        info(f"  Note: Intel XPU support is experimental. May fall back to CPU.")
    else:
        ok(f"Install strategy: CPU (no compatible discrete GPU)")


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH INSTALLATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pip(venv_python, args, desc=""):
    cmd = [str(venv_python), "-m", "pip"] + args
    if desc:
        print(f"      Installing {desc}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if r.stderr:
            print(f"      pip error: {r.stderr[-300:]}")
        return False
    return True


def install_pytorch(venv_python, hw):
    """
    Install the optimal PyTorch build based on discovered hardware.
    Returns the effective device string ("cuda", "directml", "xpu", or "cpu").
    """
    strategy = hw["install_strategy"]

    # ── NVIDIA CUDA ──────────────────────────────────────────────────────
    if strategy == "nvidia_cuda":
        nvidia = hw["nvidia"]
        # Build ordered list of CUDA wheels to try
        wheels = list(nvidia.get("recommended_cuda_wheels", []))
        # Always append fallbacks
        for w in ["cu128", "cu126", "cu124", "cu121"]:
            if w not in wheels:
                wheels.append(w)

        arch = nvidia.get("architecture", "Unknown")
        cc = nvidia.get("compute_capability", "unknown")
        info(f"GPU: {nvidia['name']} | {arch} | {cc}")
        info(f"Trying CUDA wheels in order: {' → '.join(wheels)}")
        print()

        for cu in wheels:
            print(f"      Trying PyTorch + {cu}...")
            if run_pip(venv_python, [
                "install", "torch", "torchvision", "torchaudio",
                "--index-url", f"https://download.pytorch.org/whl/{cu}"
            ]):
                # Verify GPU kernel actually executes
                verify = subprocess.run(
                    [str(venv_python), "-c",
                     "import torch; t=torch.zeros(1).cuda(); _=t+1; print('CUDA_OK')"],
                    capture_output=True, text=True, timeout=60
                )
                if verify.returncode == 0 and "CUDA_OK" in verify.stdout:
                    ok(f"PyTorch + {cu} installed — {arch} GPU acceleration active!")
                    return "cuda"
                else:
                    warn(f"{cu}: installed but kernel test failed on {cc}, trying next...")
                    run_pip(venv_python, ["uninstall", "torch", "torchvision", "torchaudio", "-y", "-q"])
            else:
                warn(f"{cu}: pip install failed, trying next...")

        warn(f"No CUDA wheel worked for {arch} {cc} — falling back to CPU")

    # ── AMD DirectML (Windows) ───────────────────────────────────────────
    elif strategy == "amd_directml":
        best = hw["best_gpu"]
        family = best.get("family", "AMD GPU")
        info(f"GPU: {best['name']} | {family}")
        info(f"Installing PyTorch CPU + torch-directml for AMD GPU acceleration...")
        print()

        # Install CPU PyTorch first
        if run_pip(venv_python, [
            "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]):
            # Then add DirectML
            if run_pip(venv_python, ["install", "torch-directml"]):
                ok(f"PyTorch + DirectML installed — {family} GPU acceleration available!")
                info(f"Note: RECALIX will use DirectML for compatible operations.")
                return "directml"
            else:
                warn("torch-directml install failed — using CPU mode")
                return "cpu"

    # ── Intel XPU ────────────────────────────────────────────────────────
    elif strategy == "intel_xpu":
        best = hw["best_gpu"]
        info(f"GPU: {best['name']} | {best.get('family', 'Intel')}")
        info(f"Attempting Intel Extension for PyTorch...")
        print()

        # Try Intel's optimized PyTorch
        if run_pip(venv_python, [
            "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]):
            if run_pip(venv_python, ["install", "intel-extension-for-pytorch"]):
                ok(f"PyTorch + Intel XPU extension installed!")
                return "xpu"
            else:
                warn("Intel extension install failed — using CPU mode")
                return "cpu"

    # ── CPU Fallback ─────────────────────────────────────────────────────
    info("Installing CPU-only PyTorch...")
    if not run_pip(venv_python, [
        "install", "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ]):
        fail("Failed to install PyTorch. Check your internet connection.")
    ok("PyTorch installed (CPU mode)")
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_venv(python_cmd, venv_path):
    cmd_parts = python_cmd.split() + ["-m", "venv", str(venv_path)]
    r = subprocess.run(cmd_parts, capture_output=True, text=True)
    return r.returncode == 0


def save_setup_state(root, state):
    state_file = root / ".recalix_setup.json"
    state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def load_setup_state(root):
    state_file = root / ".recalix_setup.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def kill_existing_server(port):
    try:
        r = subprocess.run(["netstat", "-aon"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    try:
                        subprocess.run(["taskkill", "/F", "/PID", pid],
                                       capture_output=True, timeout=5)
                        ok(f"Killed previous server (PID {pid})")
                    except Exception:
                        pass
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("RECALIX V10 — Universal Launcher")

    root = find_root()
    venv = root / VENV_NAME
    venv_python = venv / "Scripts" / "python.exe"
    backend_dir = root / "backend"

    print(f"  Project: {root}")
    print()

    # ── Check if environment already set up ───────────────────────────────
    setup_state = load_setup_state(root)
    force_clean = "--clean" in sys.argv
    needs_setup = not venv_python.exists() or force_clean

    if not needs_setup and setup_state and setup_state.get("setup_completed"):
        step("OK", "Environment already configured")
        ok(f"Python:  {setup_state.get('python_version', 'unknown')}")
        ok(f"Device:  {setup_state.get('device', 'unknown')}")
        if setup_state.get("gpu_name"):
            ok(f"GPU:     {setup_state.get('gpu_name')}")
        if setup_state.get("architecture"):
            ok(f"Arch:    {setup_state.get('architecture')} ({setup_state.get('generation', '')})")
        if setup_state.get("strategy"):
            ok(f"Backend: {setup_state.get('strategy')}")
        print()
    else:
        # ── STEP 1: Find Python ──────────────────────────────────────────
        step(1, "Finding Python 3.10-3.12...")
        python_cmd = find_python()
        if not python_cmd:
            fail(
                "Python 3.10, 3.11, or 3.12 not found!\n\n"
                "  Install Python from:\n"
                "  https://www.python.org/downloads/\n\n"
                "  IMPORTANT: Check 'Add Python to PATH' during installation!"
            )

        # ── STEP 2: Hardware Discovery ───────────────────────────────────
        step(2, "Scanning hardware (GPU detection)...")
        print()
        hw = discover_hardware()
        print_hardware_report(hw)

        # ── STEP 3: Prepare environment ────────────────────────────────────
        venv_exists = venv.exists() and venv_python.exists()

        if venv_exists and not force_clean:
            # Venv exists — validate it instead of deleting
            step(3, "Validating existing virtual environment...")
            ok(f"Found: {VENV_NAME}")

            # Quick import test for key packages
            check_script = (
                "import torch, open_clip, numpy, fastapi, easyocr; print('ALL_OK')"
            )
            check = subprocess.run(
                [str(venv_python), "-c", check_script],
                capture_output=True, text=True, timeout=60
            )
            if check.returncode == 0 and "ALL_OK" in check.stdout:
                ok("All key packages present — skipping install")
                venv_valid = True
            else:
                warn("Some packages missing — installing missing dependencies...")
                venv_valid = False
        else:
            venv_valid = False
            if force_clean and venv.exists():
                step(3, "Clean install requested (--clean flag)...")
                ok("Removing old environment...")
                shutil.rmtree(str(venv), ignore_errors=True)
            else:
                step(3, "Creating virtual environment...")

            if not create_venv(python_cmd, venv):
                fail("Failed to create virtual environment. Try running as Administrator.")
            ok(f"Created: {VENV_NAME}")

        run_pip(venv_python, ["install", "--upgrade", "pip", "-q"])

        # ── STEP 4: Install PyTorch (if needed) ─────────────────────────
        if venv_valid:
            # Venv already has all packages — check if PyTorch has GPU support
            step(4, "PyTorch already installed — verifying...")
            device_check = subprocess.run(
                [str(venv_python), "-c",
                 "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
                capture_output=True, text=True, timeout=30
            )
            device = device_check.stdout.strip() if device_check.returncode == 0 else "cpu"
            ok(f"PyTorch device: {device}")
        else:
            step(4, "Installing PyTorch (matched to your hardware)...")
            print()
            device = install_pytorch(venv_python, hw)
        print()

        # ── STEP 5: Install project deps (pip will skip already-installed) ─
        step(5, "Ensuring RECALIX dependencies are installed...")
        deps = [
            "fastapi", "uvicorn[standard]", "pydantic",
            "open-clip-torch", "numpy", "Pillow",
            "easyocr", "faiss-cpu",
            "python-multipart", "aiofiles",
        ]
        if not run_pip(venv_python, ["install"] + deps + ["-q"], "project packages"):
            fail("Failed to install dependencies. Check your internet connection.")
        ok("All dependencies installed")

        # ── STEP 6: Verify ───────────────────────────────────────────────
        step(6, "Verifying installation...")
        verify_result = subprocess.run(
            [str(venv_python), "-c", """
import torch
import open_clip
import numpy
import fastapi

print(f"Python:   {__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}")
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute:  sm_{cap[0]}{cap[1]}")
try:
    import torch_directml
    print(f"DirectML: available")
except ImportError:
    pass
try:
    import intel_extension_for_pytorch
    print(f"Intel XPU: available")
except ImportError:
    pass
print("VERIFY_OK")
"""],
            capture_output=True, text=True, timeout=120
        )
        if verify_result.returncode == 0:
            for line in verify_result.stdout.strip().split("\n"):
                if line.strip() and line.strip() != "VERIFY_OK":
                    ok(line.strip())
        else:
            warn(f"Verification warning: {verify_result.stderr[-200:]}")

        # Detect final device
        device_check = subprocess.run(
            [str(venv_python), "-c",
             "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
            capture_output=True, text=True, timeout=30
        )
        device = device_check.stdout.strip() if device_check.returncode == 0 else device

        python_ver_check = subprocess.run(
            [str(venv_python), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
            capture_output=True, text=True, timeout=10
        )
        python_ver = python_ver_check.stdout.strip() if python_ver_check.returncode == 0 else "unknown"

        # Build GPU info for state
        best = hw.get("best_gpu") or {}
        nvidia = hw.get("nvidia") or {}

        save_setup_state(root, {
            "python_version": python_ver,
            "device": device,
            "strategy": hw["install_strategy"],
            "gpu_name": best.get("name") or nvidia.get("name"),
            "gpu_vendor": best.get("vendor"),
            "architecture": nvidia.get("architecture") or best.get("architecture"),
            "generation": nvidia.get("generation") or best.get("generation"),
            "compute_capability": nvidia.get("compute_capability"),
            "cuda_version": nvidia.get("cuda_version"),
            "driver_version": nvidia.get("driver_version") or best.get("driver_version"),
            "all_gpus": [g["name"] for g in hw.get("gpus", [])],
            "setup_completed": True,
        })

        banner("Setup Complete!")

    # ══════════════════════════════════════════════════════════════════════
    # LAUNCH SERVER
    # ══════════════════════════════════════════════════════════════════════

    # Ensure data directories exist
    data_dir = backend_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "thumbnails").mkdir(exist_ok=True)
    (data_dir / "training_logs").mkdir(exist_ok=True)

    # Kill any existing server
    if is_port_open(BACKEND_PORT):
        warn(f"Port {BACKEND_PORT} is in use — attempting to free it...")
        kill_existing_server(BACKEND_PORT)
        time.sleep(1)

    # Launch backend
    step(">>", "Starting RECALIX backend server...")
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    log_file = data_dir / "server.log"
    log_handle = open(str(log_file), "w", encoding="utf-8")

    process = subprocess.Popen(
        [str(venv_python), "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(BACKEND_PORT), "--workers", "1"],
        cwd=str(backend_dir),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )

    # Wait for server
    print(f"      Waiting for server on port {BACKEND_PORT}...")
    server_ready = False
    for _ in range(60):
        if is_port_open(BACKEND_PORT):
            server_ready = True
            break
        if process.poll() is not None:
            log_handle.close()
            error_log = log_file.read_text(encoding="utf-8", errors="replace")[-1000:]
            fail(f"Backend crashed during startup.\n\n  Server log:\n{error_log}")
        time.sleep(0.5)

    log_handle.close()

    if not server_ready:
        process.terminate()
        error_log = log_file.read_text(encoding="utf-8", errors="replace")[-1000:]
        fail(f"Server didn't start within 30 seconds.\n\n  Server log:\n{error_log}")

    url = f"http://localhost:{BACKEND_PORT}"
    ok(f"Server running at {url}")
    print()
    step("->", f"Opening browser: {url}")
    webbrowser.open(url)

    banner("RECALIX is running!")
    print("  Keep this window open while using the app.")
    print("  Press Ctrl+C to stop the server.\n")

    try:
        while True:
            if process.poll() is not None:
                print("\n  Server stopped unexpectedly.")
                try:
                    log_content = log_file.read_text(encoding="utf-8", errors="replace")[-500:]
                    if log_content.strip():
                        print(f"\n  Last log output:\n{log_content}")
                except Exception:
                    pass
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Shutting down RECALIX...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("  Stopped. Goodbye!")


if __name__ == "__main__":
    main()
