from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "default.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["paths"]["train"] = str(BASE_DIR / config["paths"]["train"])
    config["paths"]["test"] = str(BASE_DIR / config["paths"]["test"])
    config["paths"]["submission"] = str(BASE_DIR / config["paths"]["submission"])
    config["paths"]["output"] = str(BASE_DIR / config["paths"]["output"])

    return config