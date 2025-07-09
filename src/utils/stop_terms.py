import os

def load_stop_terms():
    current_dir = os.path.dirname(__file__)  # src/utils/
    config_path = os.path.abspath(os.path.join(current_dir, "../config/stop_terms.txt"))

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {config_path}")
        return set()
