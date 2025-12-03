import os
from datetime import datetime

BASE_SOUL_DIR = "experiments/souls"

def generate_soul_path(kind="evolved", generation=0, step=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(BASE_SOUL_DIR, kind)
    os.makedirs(dir_path, exist_ok=True)

    filename = f"soul_gen{generation:04d}_step{step:06d}_{timestamp}.pt"
    return os.path.join(dir_path, filename)
