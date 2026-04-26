"""
One-off script: fix notebook links and make it HF Space compatible.
Run from the repo root: python scripts/fix_notebook.py
"""
import json
import pathlib

nb_path = pathlib.Path("notebooks/ShadowNet_SFT_Colab.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

OLD_REPO = "ShadowNet-When-Defense-Thinks-Like-the-Attacker"
NEW_REPO = "shadownet-defence"

# Pass 1: fix every remaining old URL string across all cells
for cell in nb["cells"]:
    new_src = []
    for line in cell["source"]:
        line = line.replace(
            f"github.com/salim7-s/{OLD_REPO}",
            f"github.com/salim7-s/{NEW_REPO}",
        )
        line = line.replace(f"%cd {OLD_REPO}", f"%cd {NEW_REPO}")
        line = line.replace(f"{OLD_REPO}.git", f"{NEW_REPO}.git")
        new_src.append(line)
    cell["source"] = new_src

# --- Cell 0: header markdown ---
nb["cells"][0]["source"] = [
    "# ShadowNet Training (Colab / Kaggle / HF Space)\n",
    "\n",
    "End-to-end fine-tuning with **optional Weights & Biases** logging. "
    "Works on Google Colab, Kaggle, and Hugging Face Spaces (JupyterLab). "
    "This notebook is the main runnable training artifact.\n",
    "\n",
    "**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  \n",
    "**Method:** supervised fine-tuning with LoRA-friendly training flow  \n",
    "**Recommended runtime:** T4 GPU or better\n",
    "\n",
    "**Repo:** [shadownet-defence](https://github.com/salim7-s/shadownet-defence)  \n",
    "**HF Space:** [zizoha/shadownet-Cops](https://huggingface.co/spaces/zizoha/shadownet-Cops)",
]

# --- Cell 2: clone + install — HF Space safe version ---
nb["cells"][2]["source"] = [
    "import os, sys\n",
    "\n",
    'REPO = "shadownet-defence"\n',
    "\n",
    "# Clone only if we are not already inside the repo.\n",
    "# On HF Spaces / Kaggle the working directory may already be the repo root.\n",
    'if not os.path.exists("environment.py"):\n',
    '    os.system(f"git clone https://github.com/salim7-s/{REPO}.git")\n',
    "    os.chdir(REPO)\n",
    "    sys.path.insert(0, os.getcwd())\n",
    "\n",
    'print("Working directory:", os.getcwd())\n',
    "\n",
    "!pip install -q -r requirements.txt\n",
    '!pip install -q "trl>=0.12" "transformers>=4.45" accelerate peft torch datasets wandb\n',
    "!python scripts/verify_core.py",
]

# --- Cell 3: W&B setup — guard google.colab import ---
nb["cells"][3]["source"] = [
    "import os\n",
    "import wandb\n",
    "\n",
    'os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "shadownet-sft")\n',
    "\n",
    "# Try Colab Secrets first; fall back to env var (works on HF Spaces / Kaggle).\n",
    '_wandb_key = os.environ.get("WANDB_API_KEY")\n',
    "try:\n",
    "    from google.colab import userdata as _ud\n",
    "    _wandb_key = _wandb_key or _ud.get(\"WANDB_API_KEY\")\n",
    "except Exception:\n",
    "    pass  # Not running in Colab\n",
    "\n",
    "if _wandb_key:\n",
    "    wandb.login(key=_wandb_key)\n",
    '    print("W&B login successful.")\n',
    "else:\n",
    '    print("No WANDB_API_KEY found. Continuing without W&B (set report_to=\'none\' below if needed).")',
]

nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print("Notebook patched successfully.")
print(f"  - All URLs updated to: {NEW_REPO}")
print("  - Cell 0: header updated (removed TRAINING.md, added HF Space link)")
print("  - Cell 2: clone/install made HF Space compatible")
print("  - Cell 3: W&B setup guards google.colab import")
