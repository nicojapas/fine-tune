import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/nicojapas/fine-tune.git"
REPO_DIR = Path("/kaggle/working/repo")
RESULTS_DIR = Path("/kaggle/working/results")

VARIANTS = ["sdxl_base", "sd15_base", "sd15_lora"]


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)
    run(["python", "src/1_prepare_data.py"], cwd=REPO_DIR)
    run(["python", "src/2_train_lora.py"], cwd=REPO_DIR)
    for variant in VARIANTS:
        run(["python", "src/3_generate.py", variant], cwd=REPO_DIR)
    run(["python", "src/4_evaluate.py"], cwd=REPO_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_DIR / "outputs", RESULTS_DIR / "outputs")
    shutil.copytree(REPO_DIR / "output", RESULTS_DIR / "output")


if __name__ == "__main__":
    main()
