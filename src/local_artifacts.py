"""
Local artifact helpers for the bank account fraud capstone project.

These helpers are also included directly in the notebook so the notebook
can run independently. This module is provided for reuse if the project is
converted into Python scripts later.
"""

from pathlib import Path
import re
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def safe_filename(value, max_length=120):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return (value or "artifact")[:max_length]


def ensure_project_dirs(project_root):
    project_root = Path(project_root)
    paths = {
        "data_raw": project_root / "data" / "raw",
        "data_processed": project_root / "data" / "processed",
        "models": project_root / "models",
        "reports": project_root / "reports",
        "figures": project_root / "reports" / "figures",
        "tables": project_root / "reports" / "tables",
        "configs": project_root / "configs",
        "logs": project_root / "reports" / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_dataframe(df, output_dir, name, index=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_filename(name)}.csv"
    df.to_csv(path, index=index)
    return path


def save_figure(output_dir, name, dpi=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_filename(name)}.png"
    plt.gcf().savefig(path, dpi=dpi, bbox_inches="tight")
    return path
