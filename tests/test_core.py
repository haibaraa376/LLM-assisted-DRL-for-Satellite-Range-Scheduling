"""Skyfield数据层最关键的配置、几何和窗口语义测试。"""

from pathlib import Path

from orbit_data.config import load_simplified_configuration


def test_fixed_scene_configuration():
    """固定场景必须保持15星、4站、24小时和论文给定速率。"""
    config = load_simplified_configuration(Path("configs/skyfield.yaml"), Path("configs/constellation.yaml"), Path("configs/ground_stations.yaml"))
    assert len(config.satellites) == 15
    assert len(config.ground_station_definitions) == 4
    assert config.data["time"]["duration_seconds"] == 86400
    assert config.data["time"]["coarse_step_seconds"] == 30
    assert config.data["rates"] == {"SGL_mbps": 60.0, "ISL_mbps": 80.0, "IDL_mbps": 80.0, "SGL_bandwidth_hz": 80000000}
