"""Garde-fous matériels : on n'entraîne que si la machine le supporte sans risque.

Lit /host/sys (monté en lecture seule dans le pod) ou /sys en local.
Sur un serveur sans batterie ni capteurs exposés, les vérifications absentes passent.
"""

from __future__ import annotations

from pathlib import Path

SYS = Path("/host/sys") if Path("/host/sys/class").exists() else Path("/sys")
CPU_SENSORS = {"k10temp", "coretemp", "zenpower"}
GPU_SENSORS = {"amdgpu", "nouveau", "xe", "i915"}


def on_ac_power(sys_root: Path = SYS) -> bool | None:
    """True/False si une alimentation secteur est détectée, None si inconnu (serveur)."""
    seen = False
    for supply in (sys_root / "class/power_supply").glob("*"):
        kind = (supply / "type").read_text().strip() if (supply / "type").exists() else ""
        if kind == "Mains" and (supply / "online").exists():
            seen = True
            if (supply / "online").read_text().strip() == "1":
                return True
    return False if seen else None


def max_temp_c(names: set[str], sys_root: Path = SYS) -> float | None:
    temps = []
    for hw in (sys_root / "class/hwmon").glob("hwmon*"):
        name_file = hw / "name"
        if not name_file.exists() or name_file.read_text().strip() not in names:
            continue
        for t in hw.glob("temp*_input"):
            try:
                temps.append(int(t.read_text()) / 1000)
            except (OSError, ValueError):
                pass
    return max(temps) if temps else None


def free_mem_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return float("inf")


def violations(guards: dict, sys_root: Path = SYS) -> list[str]:
    """Liste des raisons d'arrêter l'entraînement (vide = OK)."""
    out = []
    if guards.get("require_ac_power") and on_ac_power(sys_root) is False:
        out.append("sur batterie")
    if (t := max_temp_c(CPU_SENSORS, sys_root)) is not None and t > guards["max_cpu_temp_c"]:
        out.append(f"CPU à {t:.0f}°C")
    if (t := max_temp_c(GPU_SENSORS, sys_root)) is not None and t > guards["max_gpu_temp_c"]:
        out.append(f"GPU à {t:.0f}°C")
    if (m := free_mem_gib()) < guards["min_free_mem_gib"]:
        out.append(f"RAM libre {m:.1f} Gio")
    return out
