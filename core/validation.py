import os
import shutil

def validate_config(config: dict):
    required_keys = [
        "model", "imgsz", "conf", "iou", "tracker",
        "setup_window_sec", "fall_conveyor_residency_sec",
        "footprint_bottom_ratio", "amr_conf_min"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing configuration parameter: {key}")
    return True

def validate_environment(min_disk_mb: int = 500):
    total, used, free = shutil.disk_usage("/")
    free_mb = free / (1024 * 1024)
    if free_mb < min_disk_mb:
        return False, f"Low disk space: {free_mb:.1f} MB available (minimum {min_disk_mb} MB required)"
    return True, "OK"