"""生成24小时Skyfield轨道和链路数据的简单入口。"""

from orbit_data.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate"]))
