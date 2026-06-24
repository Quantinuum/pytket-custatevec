"""Automated benchmarking script for pytket-custatevec."""

import importlib.metadata
import platform
import time
from contextlib import suppress
from pathlib import Path

import cupy
import kaleido
import numpy as np
import pandas as pd
import plotly.express as px
from pytket._tket.circuit import Circuit

from pytket.extensions.custatevec import CuStateVecShotsBackend, CuStateVecStateBackend

# --- CONFIGURATION ---
LAYERS = 10
MIN_QUBITS = 6
MAX_QUBITS = 30
MAX_CPU_QUBITS = 26  # Stop CPU benchmarks here to avoid hanging
STEP = 2
N_SHOTS = 1000
TIMEOUT_SEC = 120.0

# --- BACKEND DISCOVERY ---
backends_sv = {"pytket-custatevec": CuStateVecStateBackend}
backends_shots = {"pytket-custatevec": CuStateVecShotsBackend}

with suppress(ImportError):
    from pytket.extensions.qiskit.backends.aer import AerBackend, AerStateBackend

    backends_sv["pytket-qiskit"] = AerStateBackend
    backends_shots["pytket-qiskit"] = AerBackend

with suppress(ImportError):
    from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend

    backends_sv["pytket-qulacs"] = QulacsBackend
    backends_shots["pytket-qulacs"] = QulacsBackend


# --- UTILS ---
def get_hardware_info() -> tuple[str, int, str]:
    """Get formatted GPU and CPU names."""
    gpu_name = "Unknown GPU"
    free_mem = 0
    with suppress(Exception):
        dev_id = 0
        props = cupy.cuda.runtime.getDeviceProperties(dev_id)
        gpu_name = props["name"].decode("utf-8")
        free_mem = cupy.cuda.Device(dev_id).mem_info[0]

    cpu_name = platform.processor()
    with suppress(Exception), Path("/proc/cpuinfo").open() as f:
        for line in f:
            if "model name" in line:
                cpu_name = line.split(":")[1].strip()
                break
    return gpu_name, free_mem, cpu_name


def get_docs_assets_path() -> Path:
    """Robustly finds the docs/assets folder relative to this script."""
    # Script is in <repo>/benchmarks/benchmark.py
    script_dir = Path(__file__).resolve().parent
    # Go up one level to <repo>/
    repo_root = script_dir.parent
    # Target <repo>/docs/assets
    assets_dir = repo_root / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def ensure_kaleido_chrome() -> None:
    """Ensures the Chrome engine for Kaleido is installed."""
    print("🔧 Checking Kaleido Chrome engine...")
    try:
        # This downloads a local chromium binary if not present
        kaleido.get_chrome_sync()
        print("✅ Kaleido Chrome engine ready.")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Failed to install Kaleido Chrome: {e}")


def random_line_circuit(n_qubits: int, layers: int, measure: bool = False) -> Circuit:
    """Generates a random circuit with linear connectivity."""
    np.random.seed(42)  # noqa: NPY002
    c = Circuit(n_qubits)
    for i in range(layers):
        for q in range(n_qubits):
            c.TK1(np.random.rand(), np.random.rand(), np.random.rand(), q)  # noqa: NPY002

        offset = np.mod(i, 2)
        qubit_pairs = [[c.qubits[i], c.qubits[i + 1]] for i in range(offset, n_qubits - 1, 2)]
        for pair in qubit_pairs:
            if np.random.rand() > 0.5:  # noqa: NPY002, PLR2004
                pair = [pair[1], pair[0]]  # noqa: PLW2901
            c.CX(pair[0], pair[1])

    if measure:
        c.measure_all()
    return c


# --- REPORTING ---
def generate_env_report(gpu_name: str, cpu_name: str) -> None:
    """Generates a markdown table with environment details."""

    def get_ver(pkg: str) -> str:
        with suppress(Exception):
            return importlib.metadata.version(pkg)
        return "N/A"

    report = f"""
| Component | Specification / Version |
| :--- | :--- |
| **GPU** | {gpu_name} |
| **CPU** | {cpu_name} |
| **Python** | {platform.python_version()} |
| **OS** | {platform.system()} {platform.release()} |
| **pytket** | {get_ver("pytket")} |
| **pytket-custatevec** | {get_ver("pytket-custatevec")} |
| **pytket-qulacs** | {get_ver("pytket-qulacs")} |
| **pytket-qiskit** | {get_ver("pytket-qiskit")} |
| **cuquantum-python** | {get_ver("cuquantum-python")} |
"""
    output_path = get_docs_assets_path() / "benchmark_env.md"
    output_path.write_text(report.strip())
    print(f"✅ Generated Environment Report at {output_path}")


# --- BENCHMARK LOGIC ---
def run_comparison(mode: str = "statevector") -> tuple[pd.DataFrame, str, str]:
    """Run benchmark comparison."""
    results = []
    gpu_name, free_vram, cpu_name = get_hardware_info()

    target_backends = backends_sv if mode == "statevector" else backends_shots

    for n in range(MIN_QUBITS, MAX_QUBITS + 1, STEP):
        if (16 * (2**n)) > (free_vram * 0.9):
            break

        print(f"  - {n} qubits...")
        circ = random_line_circuit(n, LAYERS, measure=(mode == "shots"))

        for name, BackendClass in target_backends.items():  # noqa: N806
            is_cpu = "custatevec" not in name
            if is_cpu and n > MAX_CPU_QUBITS:
                continue

            with suppress(Exception):
                b = BackendClass()
                c_compiled = b.get_compiled_circuit(circ)
                start = time.time()

                if mode == "statevector":
                    _ = b.run_circuit(c_compiled).get_state()
                else:
                    _ = b.run_circuit(c_compiled, n_shots=N_SHOTS).get_counts()

                elapsed = time.time() - start
                results.append({"Qubits": n, "Time (s)": elapsed, "Backend": name})

                if is_cpu and elapsed > TIMEOUT_SEC:
                    break

    return pd.DataFrame(results), gpu_name, cpu_name


def save_plot(df: pd.DataFrame, title: str, filename_base: str, gpu_name: str, cpu_name: str) -> None:
    """Saves interactive HTML (for Docs) and static PNG (for README)."""
    output_dir = get_docs_assets_path()

    if df.empty:
        print(f"⚠️ Skipping plot {filename_base} (Empty Data)")
        return

    colors = {
        "pytket-custatevec": "#76b900",
        "pytket-qiskit": "#ff5722",
        "pytket-qulacs": "#29b6f6",
    }

    fig = px.line(
        df,
        x="Qubits",
        y="Time (s)",
        color="Backend",
        markers=True,
        log_y=True,
        title=f"<b>{title}</b>",
        color_discrete_map=colors,
    )

    fig.add_annotation(
        text=f"GPU: {gpu_name} | CPU: {cpu_name} | Depth: {LAYERS}",
        xref="paper",
        yref="paper",
        x=0,
        y=1.05,
        showarrow=False,
        font={"size": 11, "color": "#7f7f7f"},
        align="left",
    )

    fig.update_layout(
        font={"family": "Roboto, sans-serif", "size": 14, "color": "#7f7f7f"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(128,128,128,0.1)",
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
        xaxis={"gridcolor": "rgba(128,128,128,0.2)", "showspikes": True},
        yaxis={
            "gridcolor": "rgba(128,128,128,0.2)",
            "exponentformat": "power",
            "dtick": 1,
        },
        hovermode="x unified",
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.5)",
        },
    )
    fig.update_traces(marker={"size": 8}, line={"width": 3})

    # 1. Save HTML
    html_path = output_dir / f"{filename_base}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=False)

    # 2. Save PNG
    png_path = output_dir / f"{filename_base}.png"
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    fig.write_image(str(png_path), width=800, height=500, scale=2)

    print(f"✅ Saved {filename_base} (.html and .png) to {output_dir}")


if __name__ == "__main__":
    ensure_kaleido_chrome()

    df_sv, gpu, cpu = run_comparison("statevector")
    generate_env_report(gpu, cpu)

    save_plot(df_sv, "Statevector Simulation", "benchmark_sv", gpu, cpu)

    df_shots, _, _ = run_comparison("shots")
    save_plot(df_shots, "Shot-Based Simulation (1000 Shots)", "benchmark_shots", gpu, cpu)
