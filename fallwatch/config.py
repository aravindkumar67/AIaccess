from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    model_path: str
    api_enabled: bool
    api_url: str
    api_method: str
    api_token: str
    api_timeout: float
    config: dict[str, Any]

def load_settings(config_path: str = "config.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    return Settings(os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt"),
        os.getenv("API_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
        os.getenv("API_URL", "").strip(), os.getenv("API_METHOD", "POST").strip().upper(),
        os.getenv("API_BEARER_TOKEN", "").strip(), float(os.getenv("API_TIMEOUT_SECONDS", "4")), cfg)
