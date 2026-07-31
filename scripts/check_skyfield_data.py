"""检查已生成Skyfield数据的简单入口，不会重新传播轨道。"""

from orbit_data.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["check"]))
