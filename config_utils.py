from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
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


def _partial_resolve(cfg, node=None, key_path=""):
    """
    Resolve config leaf by leaf.
    If one interpolation cannot be resolved, keep its original placeholder text.
    """
    if node is None:
        node = cfg

    if OmegaConf.is_dict(node):
        return {
            k: _partial_resolve(
                cfg,
                node[k],
                f"{key_path}.{k}" if key_path else k
            )
            for k in node.keys()
        }

    if OmegaConf.is_list(node):
        return [
            _partial_resolve(
                cfg,
                node[i],
                f"{key_path}.{i}" if key_path else str(i)
            )
            for i in range(len(node))
        ]

    try:
        return OmegaConf.select(
            cfg,
            key_path,
            throw_on_resolution_failure=True
        )
    except OmegaConfBaseException:
        return str(node)


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

    If resolve=True, resolve values one by one. Missing interpolations are kept
    as their original placeholder strings instead of raising an error.
    """
    os.environ.setdefault("EXPERIMENT_NAME", "")
    if experiment_name:
        os.environ["EXPERIMENT_NAME"] = experiment_name

    print(f"[config_utils] Using machine profile: {machine}")

    base = OmegaConf.load(Path(base_dir) / "paths.yaml")
    mach = OmegaConf.load(Path(base_dir) / "machine" / f"{machine}.yaml")
    exp = OmegaConf.load(Path(base_dir) / "experiments" / exp_name)

    if prior_machine:
        # machine wins
        cfg = OmegaConf.merge(base, exp, mach)
    else:
        # experiment wins (original)
        cfg = OmegaConf.merge(base, mach, exp)

    if resolve:
        return _partial_resolve(cfg)

    return OmegaConf.to_container(cfg, resolve=False)