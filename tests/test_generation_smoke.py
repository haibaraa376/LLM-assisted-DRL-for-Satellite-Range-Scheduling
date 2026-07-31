"""读取最终Skyfield数据，检查关键数量和输出可读性。"""

import json
from pathlib import Path

import numpy as np


def test_final_data_counts():
    """最终数据必须保持2881时间点及三类窗口的既定数量。"""
    root = Path("data/skyfield")
    positions = np.load(root / "satellite_positions.npz")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert positions["timestamps_unix_s"].shape == (2881,)
    assert positions["position_gcrs_km"].shape == (2881, 15, 3)
    assert summary["SGL_statistics"]["window_count"] == 121
    assert summary["ISL_statistics"]["window_count"] == 30
    assert summary["IDL_statistics"]["window_count"] == 984
