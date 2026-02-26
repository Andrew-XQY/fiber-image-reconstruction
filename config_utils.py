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


def load_config(
    exp_name: str,
    machine: str,
    base_dir: str = "conf",
    experiment_name: str | None = None,
    prior_machine: bool = False,  
    resolve: bool = True
) -> dict:
    """
    If prior_machine=True, machine config overwrites experiment config.
    Otherwise (default), experiment overwrites machine (original behavior).
    """
    os.environ.setdefault("EXPERIMENT_NAME", "")
    if experiment_name:
        os.environ["EXPERIMENT_NAME"] = experiment_name

    print(f"[config_utils] Using machine profile: {machine}")

    base = OmegaConf.load(Path(base_dir) / "paths.yaml")
    mach = OmegaConf.load(Path(base_dir) / "machine" / f"{machine}.yaml")
    exp  = OmegaConf.load(Path(base_dir) / "experiments" / exp_name)

    if prior_machine:
        # machine wins
        cfg = OmegaConf.merge(base, exp, mach)
    else:
        # experiment wins (original)
        cfg = OmegaConf.merge(base, mach, exp)

    return OmegaConf.to_container(cfg, resolve=True)

