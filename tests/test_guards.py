from pathlib import Path

from idtn import guards

LIMITS = {"require_ac_power": True, "max_cpu_temp_c": 80, "max_gpu_temp_c": 80,
          "min_free_mem_gib": 0}


def _fake_sys(tmp: Path, ac: str, cpu_temp: int) -> Path:
    ps = tmp / "class/power_supply/AC"
    ps.mkdir(parents=True)
    (ps / "type").write_text("Mains\n")
    (ps / "online").write_text(ac)
    hw = tmp / "class/hwmon/hwmon0"
    hw.mkdir(parents=True)
    (hw / "name").write_text("k10temp\n")
    (hw / "temp1_input").write_text(str(cpu_temp * 1000))
    return tmp


def test_ok(tmp_path):
    assert guards.violations(LIMITS, _fake_sys(tmp_path, "1", 60)) == []


def test_battery_and_heat(tmp_path):
    v = guards.violations(LIMITS, _fake_sys(tmp_path, "0", 95))
    assert "sur batterie" in v and any("CPU" in x for x in v)


def test_server_without_sensors(tmp_path):
    assert guards.violations(LIMITS, tmp_path) == []
