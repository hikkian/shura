"""Thresholds and rough expert-weight bandwidth estimates for Shura Monitor.

Keep UI status thresholds and model-specific estimates together here so the
monitor never invents a value in its rendering layer.
"""

THRESHOLDS = {
    # Free VRAM, in MiB. The 1536 MiB reserve is the working-desktop budget.
    "vram_crit_free_mib": 1536,
    "vram_warn_free_mib": 2048,
    "vram_reserve_mib": 1536,
    # MemAvailable, in GiB.
    "ram_crit_available_gib": 1.5,
    "ram_warn_available_gib": 3.0,
    # PSI avg10 is a percentage.
    "psi_full_crit_avg10": 10.0,
    "psi_full_warn_avg10": 3.0,
    "gpu_temp_crit_c": 83.0,
    "gpu_temp_warn_c": 78.0,
    "cpu_temp_crit_c": 90.0,
    "cpu_temp_warn_c": 80.0,
    "zram_crit_percent": 92.0,
    "zram_warn_percent": 80.0,
    # Dashboard context bar usage percentage.
    "context_warn_percent": 80.0,
    "context_crit_percent": 92.0,
    # Logical CPU utilization bar percentage.
    "cpu_core_warn_percent": 60.0,
    "cpu_core_crit_percent": 85.0,
    "ssd_crit_free_gb": 20.0,
}

# Approximate expert weight reads per generated token; displayed as estimates.
# These are not hardware-counter measurements and must remain visibly labelled.
ESTIMATED_WEIGHT_READ_MB_PER_TOKEN = {
    "tiel-coder": 410.0,
    "occamy": 465.0,
}
