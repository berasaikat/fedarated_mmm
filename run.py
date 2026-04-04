"""
Main CLI for federated MMM: data generation, training, incrementality validation, and plots.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(root: Path, p: Path | str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to global YAML configuration.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running.",
    )


def cmd_generate_data(cfg_path: Path, dry_run: bool) -> None:
    root = project_root()
    synthetic_dir = root / "data" / "synthetic"
    if str(synthetic_dir) not in sys.path:
        sys.path.insert(0, str(synthetic_dir))

    target = "run_generation.run(global_config_path=<resolved --config>)"
    if dry_run:
        print(f"[dry-run] Would import from {synthetic_dir} and call {target}")
        print(f"[dry-run]   global_config_path={cfg_path.resolve()}")
        return

    import run_generation

    run_generation.run(global_config_path=cfg_path)


def cmd_train(cfg_path: Path, dry_run: bool) -> None:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if dry_run:
        print(
            "[dry-run] Would call "
            f"aggregator.federated_loop.run_federated_training({cfg_path.resolve()!s})"
        )
        return

    from aggregator.federated_loop import run_federated_training

    run_federated_training(str(cfg_path.resolve()))


def _parse_round_index(name: str) -> Optional[int]:
    m = re.match(r"^round_(\d+)\.json$", name, re.I)
    return int(m.group(1)) if m else None


def _load_round_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    files = []
    for f in results_dir.glob("round_*.json"):
        n = _parse_round_index(f.name)
        if n is not None:
            files.append((n, f))
    files.sort(key=lambda x: x[0])
    out: List[Dict[str, Any]] = []
    for _, path in files:
        with open(path, encoding="utf-8") as fp:
            out.append(json.load(fp))
    return out


def _select_round_payload_for_audit(
    summaries: List[Dict[str, Any]], spec: Any
) -> Dict[str, Any]:
    if not summaries:
        raise FileNotFoundError("No round_*.json files found under results_dir.")
    if spec in (None, "last"):
        return summaries[-1]
    idx = int(spec)
    for s in summaries:
        if s.get("round_num") == idx:
            return s
    raise ValueError(f"No round with round_num={idx} in results summaries.")


def _load_matched_geos(audit_cfg: dict, root: Path) -> Dict[str, Any]:
    if "matched_geos_json" in audit_cfg and "matched_geos" in audit_cfg:
        raise ValueError(
            "incrementality_audit: use either matched_geos or matched_geos_json, not both."
        )
    if "matched_geos_json" in audit_cfg:
        p = _resolve_path(root, audit_cfg["matched_geos_json"])
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if "matched_geos" not in audit_cfg:
        raise ValueError(
            "incrementality_audit requires matched_geos or matched_geos_json."
        )
    mg = audit_cfg["matched_geos"]
    if isinstance(mg, str):
        p = _resolve_path(root, mg)
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return mg


def cmd_validate(cfg_path: Path, dry_run: bool) -> None:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cfg = load_yaml(cfg_path.resolve())
    audit_cfg = cfg.get("incrementality_audit") or cfg.get("audit")
    if not audit_cfg:
        raise ValueError(
            "Config must define an 'incrementality_audit' (or legacy 'audit') section with "
            "geo_csv, matched_geos / matched_geos_json, and channel."
        )

    results_dir = _resolve_path(
        root, audit_cfg.get("results_dir", cfg.get("results_dir", "results"))
    )
    round_spec = audit_cfg.get("global_summary_round", audit_cfg.get("round", "last"))
    summaries = _load_round_summaries(results_dir)
    round_payload = _select_round_payload_for_audit(summaries, round_spec)
    global_summary = round_payload.get("global_summary", {})

    geo_csv = _resolve_path(root, audit_cfg["geo_csv"])
    matched_geos = _load_matched_geos(audit_cfg, root)
    channel = audit_cfg["channel"]

    if dry_run:
        print("[dry-run] Would load geo data:", geo_csv)
        print("[dry-run] Would load matched geos (keys):", list(matched_geos.keys()))
        print(
            "[dry-run] Would use global_summary from round:",
            round_payload.get("round_num"),
        )
        print("[dry-run] Would call causal_validation.audit.run_incrementality_audit(")
        print("[dry-run]   global_summary=<dict>, geo_data=<DataFrame>, ")
        print(f"[dry-run]   matched_geos={matched_geos!r},")
        print(f"[dry-run]   channel_to_audit={channel!r},")
        print("[dry-run] )")
        return

    from causal_validation.audit import run_incrementality_audit

    geo_data = pd.read_csv(geo_csv)
    result = run_incrementality_audit(
        global_summary=global_summary,
        geo_data=geo_data,
        matched_geos=matched_geos,
        channel_to_audit=str(channel),
    )
    print(json.dumps(result, indent=2, default=float))

    # save to disk
    audit_out = results_dir / "audit_results.json"
    with open(audit_out, "w") as f:
        json.dump([result], f, indent=2, default=float)
    print(f"Audit result saved to {audit_out}")


def _normalize_round_for_plots(row: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(row)
    if "surprise_scores" not in r and "per_participant_surprise" in r:
        r["surprise_scores"] = r["per_participant_surprise"]
    return r


def _participants_for_plots(
    round_history: List[Dict[str, Any]], cfg: dict
) -> List[str]:
    ids = cfg.get("participant_ids")
    if ids:
        return [str(x) for x in ids]
    n = int(cfg.get("num_participants", 0) or 0)
    if n > 0:
        return [f"participant_{i}" for i in range(1, n + 1)]
    for payload in round_history:
        d = (
            payload.get("per_participant_surprise")
            or payload.get("surprise_scores")
            or {}
        )
        if d:
            return sorted(str(k) for k in d.keys())
    return []


def _budget_tracker_proxy(
    round_history: List[Dict[str, Any]], cfg: dict, participants: List[str]
) -> Any:
    priv = cfg.get("privacy") or {}
    total_eps = float(cfg.get("total_epsilon", priv.get("epsilon", 10.0)))
    spent: Dict[str, Dict[str, float]] = {
        p: {"epsilon": 0.0, "delta": 0.0} for p in participants
    }
    delta_default = float(priv.get("delta", cfg.get("total_delta", 1e-4)))
    for payload in round_history:
        eps_step = float(payload.get("epsilon_spent_per_participant", 0.0))
        for p in participants:
            spent[p]["epsilon"] += eps_step
            spent[p]["delta"] += delta_default
    return SimpleNamespace(total_epsilon=total_eps, spent_budgets=spent)


def _load_priors_history(root: Path, viz_cfg: dict) -> Optional[List[Dict[str, Any]]]:
    exp_dir = viz_cfg.get("experiment_dir") or viz_cfg.get("experiment_log_dir")
    if not exp_dir:
        return None
    logs = _resolve_path(root, exp_dir) / "logs" / "priors.jsonl"
    if not logs.exists():
        return None
    rows = []
    with open(logs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows or None


def _load_audit_results_for_plot(root: Path, cfg: dict) -> List[Dict[str, Any]]:
    viz = cfg.get("visualization") or {}
    audit = cfg.get("incrementality_audit") or cfg.get("audit") or {}
    raw_path = viz.get("audit_results_json") or audit.get("audit_results_json")
    if not raw_path:
        return []
    p = _resolve_path(root, raw_path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return list(data) if isinstance(data, list) else [data]


def cmd_visualize(cfg_path: Path, dry_run: bool) -> None:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cfg = load_yaml(cfg_path.resolve())
    viz_cfg = cfg.get("visualization") or {}
    results_dir = _resolve_path(
        root, viz_cfg.get("results_dir", cfg.get("results_dir", "results"))
    )
    out_dir = _resolve_path(root, viz_cfg.get("output_dir", "plots"))
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_round_summaries(results_dir)
    round_history = [_normalize_round_for_plots(s) for s in summaries]
    channels = list(cfg.get("channels") or [])
    participants = _participants_for_plots(round_history, cfg)

    if not participants:
        print(
            "WARNING: Could not infer participant IDs — surprise heatmap will be skipped"
        )
        print("  Add 'num_participants' or 'participant_ids' to your config")

    out_posterior = out_dir / "posterior_evolution.png"
    out_privacy = out_dir / "privacy_budget.png"
    out_surprise = out_dir / "surprise_heatmap.png"
    out_audit = out_dir / "audit_chart.png"

    priors_hint = _load_priors_history(root, viz_cfg)
    audit_list = _load_audit_results_for_plot(root, cfg)

    if dry_run:
        print("[dry-run] Would load rounds from:", results_dir)
        print("[dry-run] Would write plots under:", out_dir)
        print(
            "[dry-run] Would call visualization.posterior_plots.plot_posterior_evolution("
            f"round_history=<{len(round_history)} rounds>, channels={channels!r}, "
            f"output_path={out_posterior!s}, priors_history={priors_hint!r})"
        )
        print(
            "[dry-run] Would call visualization.privacy_plots.plot_budget_consumption("
            f"budget_tracker=<synthetic from rounds>, output_path={out_privacy!s})"
        )
        print(
            "[dry-run] Would call visualization.surprise_heatmap.plot_surprise_heatmap("
            f"round_history=..., participants={participants!r}, channels={channels!r}, "
            f"output_path={out_surprise!s})"
        )
        print(
            "[dry-run] Would call visualization.audit_chart.plot_audit_results("
            f"audit_results_list=<{len(audit_list)} entries>, output_path={out_audit!s})"
        )
        return

    from visualization.audit_chart import plot_audit_results
    from visualization.posterior_plots import plot_posterior_evolution
    from visualization.privacy_plots import plot_budget_consumption
    from visualization.surprise_heatmap import plot_surprise_heatmap

    plot_posterior_evolution(
        round_history,
        channels,
        str(out_posterior),
        priors_history=priors_hint,
    )

    tracker = _budget_tracker_proxy(round_history, cfg, participants)
    plot_budget_consumption(tracker, str(out_privacy))

    plot_surprise_heatmap(round_history, participants, channels, str(out_surprise))

    plot_audit_results(audit_list, str(out_audit))

    print(f"Wrote plots under {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated MMM pipeline CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-data", help="Generate synthetic participant CSVs.")
    add_common_args(p_gen)

    p_train = sub.add_parser("train", help="Run federated training loop.")
    add_common_args(p_train)

    p_val = sub.add_parser(
        "validate", help="Run incrementality vs synthetic-control audit."
    )
    add_common_args(p_val)

    p_viz = sub.add_parser("visualize", help="Render all diagnostic plots.")
    add_common_args(p_viz)

    p_report = sub.add_parser("report", help="Print experiment report to console.")
    p_report.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment folder with experiment_summary.json",
    )
    p_report.add_argument("--config", type=Path, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "report":
        from report import print_report

        cfg_opt: Optional[Path] = (
            args.config.resolve() if getattr(args, "config", None) else None
        )
        print_report(args.experiment_dir.resolve(), cfg_opt)
        return

    cfg_path: Path = args.config.resolve()

    commands: Dict[str, Callable[[], None]] = {
        "generate-data": lambda: cmd_generate_data(cfg_path, args.dry_run),
        "train": lambda: cmd_train(cfg_path, args.dry_run),
        "validate": lambda: cmd_validate(cfg_path, args.dry_run),
        "visualize": lambda: cmd_visualize(cfg_path, args.dry_run),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
