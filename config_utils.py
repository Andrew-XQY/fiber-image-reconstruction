from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pathlib import Path
import os, socket

# Hostname substring to machine profile.
HOST_TO_MACHINE = {
    "mac": "mac-andrewxu",
    "mws-147574r": "win-qiyuanxu",
    "pcbe15789": "win-xqiyuan",
    "": "liverpool-hpc",                    # Default profile for unmatched hostnames.
}


def detect_machine() -> str:
    """Return the MACHINE override or first matching hostname profile."""
    # Explicit environment override.
    machine = os.getenv("MACHINE")
    if machine:
        print("Using machine profile:", machine)
        return machine

    # Match the hostname against configured profiles.
    host = socket.gethostname().lower()
    for key, profile in HOST_TO_MACHINE.items():
        if key in host:
            return profile

    # No matching profile.
    raise RuntimeError(
        f"[config_utils] Could not resolve machine profile from hostname='{host}'. "
        f"Please set MACHINE env variable to one of: {list(HOST_TO_MACHINE.values())}"
    )


def _partial_resolve(cfg):
    """
    Resolve config leaf by leaf.
    If one interpolation cannot be resolved, keep its original placeholder text.
    """
    raw = OmegaConf.to_container(cfg, resolve=False)

    def walk(node, key_path=""):
        if isinstance(node, dict):
            return {
                k: walk(v, f"{key_path}.{k}" if key_path else k)
                for k, v in node.items()
            }

        if isinstance(node, list):
            return [
                walk(v, f"{key_path}.{i}" if key_path else str(i))
                for i, v in enumerate(node)
            ]

        try:
            return OmegaConf.select(
                cfg,
                key_path,
                throw_on_resolution_failure=True
            )
        except OmegaConfBaseException:
            return node   # keep original unresolved value

    return walk(raw)


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
    Otherwise, experiment config overwrites machine config.

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
        # experiment wins
        cfg = OmegaConf.merge(base, mach, exp)

    if resolve:
        return _partial_resolve(cfg)

    return OmegaConf.to_container(cfg, resolve=False)


def _replace_exact_values(node, replacements):
    if isinstance(node, dict):
        return {key: _replace_exact_values(value, replacements) for key, value in node.items()}
    if isinstance(node, list):
        return [_replace_exact_values(value, replacements) for value in node]
    if isinstance(node, str):
        return replacements.get(node, node)
    return node


def load_run(run_name: str, roots, machine: str, base_dir: str = "conf") -> dict:
    """Discover one run, enforce its folder contract, and load its saved config."""
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a single folder name")

    if isinstance(roots, (str, Path)):
        roots = [roots]
    roots = [Path(root).expanduser().resolve() for root in roots]
    matches = sorted({root / run_name for root in roots if (root / run_name).is_dir()})

    if not matches:
        raise FileNotFoundError(
            f"Run {run_name!r} was not found under: {', '.join(map(str, roots))}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Run {run_name!r} is ambiguous: {matches}")

    run_dir = matches[0]
    model_path = run_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Run is missing model.pt: {run_dir}")

    snapshots = sorted([*run_dir.glob("*.yaml"), *run_dir.glob("*.yml")])
    if len(snapshots) != 1:
        raise RuntimeError(
            f"Run must contain exactly one YAML snapshot; found {len(snapshots)} in {run_dir}"
        )

    snapshot_path = snapshots[0]
    config = _partial_resolve(OmegaConf.load(snapshot_path))
    machine_config = _partial_resolve(
        OmegaConf.load(Path(base_dir) / "machine" / f"{machine}.yaml")
    )

    saved_datasets = config.get("paths", {}).get("datasets", {})
    runtime_datasets = machine_config.get("paths", {}).get("datasets", {})
    if not saved_datasets or not runtime_datasets:
        raise ValueError("Snapshot and machine config must define paths.datasets")

    replacements = {}
    for key in saved_datasets.keys() & runtime_datasets.keys():
        old, new = saved_datasets[key], runtime_datasets[key]
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        if old in replacements and replacements[old] != new:
            raise ValueError(f"Saved dataset path {old!r} maps to multiple runtime paths")
        replacements[old] = new

    config = _replace_exact_values(config, replacements)
    for key in saved_datasets.keys() & runtime_datasets.keys():
        config["paths"]["datasets"][key] = runtime_datasets[key]

    eval_sql_key = config.get("data", {}).get("eval_sql_key")
    if not eval_sql_key or eval_sql_key not in config.get("sql", {}):
        raise ValueError("Snapshot must define data.eval_sql_key and its SQL query")

    return {
        "root": run_dir.parent,
        "model": model_path,
        "snapshot": snapshot_path,
        "inference": run_dir / "inference",
        "config": config,
    }
