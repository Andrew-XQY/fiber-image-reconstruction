from omegaconf import OmegaConf
from pathlib import Path
import os, socket

# Structured mapping (hostname substring -> machine profile)
HOST_TO_MACHINE = {
    "mac": "mac-andrewxu",                  # if hostname contains "mac" → use conf/machine/mac-andrewxu.yaml        
    "mws-147574r": "win-qiyuanxu",          # if hostname contains "mws-147574r" → use conf/machine/win-qiyuanxu.yaml
    "pcbe15789": "win-xqiyuan",             # if hostname contains "pcbe15789" → use conf/machine/win-xqiyuan.yaml
    "": "liverpool-hpc",                    # direct set in env variable to tell use this machine
    # add more as needed, add new hostname + machine profile pairs
}

def detect_machine() -> str:
    """Return machine profile from env MACHINE or by hostname mapping.
    Raises RuntimeError if no mapping is found."""
    # 1. If explicitly set, trust it
    machine = os.getenv("MACHINE")
    if machine:
        print("Using machine profile:", machine)
        return machine

    # 2. Otherwise, try to detect from hostname
    host = socket.gethostname().lower()
    for key, profile in HOST_TO_MACHINE.items():
        if key in host:
            return profile

    # 3. Nothing matched -> fail
    raise RuntimeError(
        f"[config_utils] Could not resolve machine profile from hostname='{host}'. "
        f"Please set MACHINE env variable to one of: {list(HOST_TO_MACHINE.values())}"
    )


def load_config(exp_name: str, base_dir: str = "conf", experiment_name: str = None) -> dict:
    """
    Load + merge base paths, machine profile, and experiment config.
    If experiment_name is provided, set it as env var for OmegaConf interpolation in paths.yaml.
    Returns a fully resolved plain dict (no ${...} left).
    """
    os.environ.setdefault("EXPERIMENT_NAME", "")
    if experiment_name:
        os.environ["EXPERIMENT_NAME"] = experiment_name
    machine = detect_machine()
    print(f"[config_utils] Using machine profile: {machine}")

    base = OmegaConf.load(Path(base_dir) / "paths.yaml")
    mach = OmegaConf.load(Path(base_dir) / "machine" / f"{machine}.yaml")
    exp  = OmegaConf.load(Path(base_dir) / "experiments" / exp_name)

    cfg = OmegaConf.merge(base, mach, exp)
    return OmegaConf.to_container(cfg, resolve=True)


# usage example
# from config_utils import load_config

# cfg = load_config("SHL_DNN.yaml")
# print(cfg["paths"]["dataset"])