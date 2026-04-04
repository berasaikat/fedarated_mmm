"""
Console reporting for a federated experiment: rounds JSONL, summary, and audit logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_yaml_optional(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_channels(rounds: List[dict]) -> List[str]:
    keys: set[str] = set()
    for row in rounds:
        gs = row.get("global_summary") or {}
        if isinstance(gs, dict):
            keys.update(gs.keys())
    return sorted(keys)


def convergence_for_round(
    rounds_sorted: List[dict], idx: int, threshold: float = 0.05
) -> str:
    if idx == 0:
        return "-"
    prev = rounds_sorted[idx - 1].get("global_summary") or {}
    cur = rounds_sorted[idx].get("global_summary") or {}
    if not prev or not cur:
        return "?"
    max_delta = 0.0
    for ch in cur:
        if ch not in prev:
            continue
        try:
            m0 = float((prev[ch] or {}).get("mean", 0.0))
            m1 = float((cur[ch] or {}).get("mean", 0.0))
        except (TypeError, ValueError):
            continue
        max_delta = max(max_delta, abs(m1 - m0))
    if max_delta < threshold:
        return "yes"
    return "no"


def cumulative_epsilon_per_participant(
    rounds_sorted: List[dict], up_to_index: int
) -> float:
    total = 0.0
    for i in range(up_to_index + 1):
        total += float(rounds_sorted[i].get("epsilon_spent_per_participant", 0.0))
    return total


def privacy_remaining_str(
    rounds_sorted: List[dict],
    idx: int,
    total_epsilon: Optional[float],
) -> str:
    if total_epsilon is None:
        return "-"
    spent = cumulative_epsilon_per_participant(rounds_sorted, idx)
    rem = total_epsilon - spent
    return f"{max(0.0, rem):.4g}"


def collect_surprise_values(surprise_scores: Any) -> List[float]:
    vals: List[float] = []
    if not surprise_scores:
        return vals
    if not isinstance(surprise_scores, dict):
        return vals
    for _pid, payload in surprise_scores.items():
        if isinstance(payload, dict):
            for v in payload.values():
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        else:
            try:
                vals.append(float(payload))
            except (TypeError, ValueError):
                pass
    return vals


def audit_coverage_summary(audits: List[dict]) -> Tuple[str, int, int]:
    if not audits:
        return ("no audits logged", 0, 0)
    passed = 0
    total = 0
    for row in audits:
        ar = row.get("audit_result") or row
        if "coverage" in ar:
            total += 1
            if ar.get("coverage"):
                passed += 1
    if total == 0:
        return ("-", 0, 0)
    label = f"{passed}/{total} channels"
    return (label, passed, total)


def resolve_budget_limit(summary: dict, config: dict) -> Optional[float]:
    for candidate in (
        summary.get("total_epsilon"),
        summary.get("privacy_budget_epsilon"),
    ):
        if candidate is not None:
            try:
                return float(candidate)
            except (TypeError, ValueError):
                pass
    priv = config.get("privacy")
    if isinstance(priv, dict) and priv.get("epsilon") is not None:
        return float(priv["epsilon"])
    if config.get("total_epsilon") is not None:
        return float(config["total_epsilon"])
    return None


def build_round_rows(
    rounds_sorted: List[dict],
    channels: Sequence[str],
    total_epsilon: Optional[float],
    audit_label: str,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for idx, row in enumerate(rounds_sorted):
        rnum = row.get("round_num", idx + 1)
        gs = row.get("global_summary") or {}
        cells = [str(rnum)]
        for ch in channels:
            ch_rec = gs.get(ch)
            if isinstance(ch_rec, dict) and "mean" in ch_rec:
                cells.append(f"{float(ch_rec['mean']):.4f}")
            else:
                cells.append("-")
        cells.append(privacy_remaining_str(rounds_sorted, idx, total_epsilon))
        cells.append(convergence_for_round(rounds_sorted, idx))
        cells.append(audit_label)
        rows.append(cells)
    return rows


def print_report(
    experiment_dir: Path,
    config_path: Optional[Path] = None,
) -> None:
    exp_dir = experiment_dir.resolve()
    summary_path = exp_dir / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing experiment_summary.json under {exp_dir}")

    summary = load_json(summary_path)
    logs_dir = exp_dir / "logs"
    rounds = load_jsonl(logs_dir / "rounds.jsonl")
    audits = load_jsonl(logs_dir / "audits.jsonl")

    rounds_sorted = sorted(
        rounds,
        key=lambda r: int(r.get("round_num", 0)),
    )
    config = load_yaml_optional(config_path.resolve() if config_path else None)
    total_epsilon = resolve_budget_limit(summary, config)

    channels = infer_channels(rounds_sorted)
    audit_label, _pass, _total = audit_coverage_summary(audits)
    if audit_label == "no audits logged":
        audit_col = audit_label
    elif audit_label == "-":
        audit_col = "-"
    else:
        audit_col = audit_label

    headers = (
        ["Round"]
        + [f"beta_mean ({ch})" for ch in channels]
        + ["eps_remaining", "converged", "audit_coverage"]
    )
    body = build_round_rows(rounds_sorted, channels, total_epsilon, audit_col)

    def surprise_section_plain() -> None:
        print("\nLLM Prior Quality")
        print("  Average surprise (per round, over participants x channels):")
        overall_surprise: List[float] = []
        for row in rounds_sorted:
            rnum = row.get("round_num", "?")
            vals = collect_surprise_values(row.get("surprise_scores") or {})
            if vals:
                avg = sum(vals) / len(vals)
                overall_surprise.append(avg)
                print(f"    Round {rnum}: {avg:.4f}")
            else:
                print(f"    Round {rnum}: -")
        if overall_surprise:
            mu = sum(overall_surprise) / len(overall_surprise)
            print(f"  Across rounds (mean of round averages): {mu:.4f}")

    def audits_section_plain() -> None:
        print("\nIncrementality audits")
        if not audits:
            print("  (none)")
        else:
            for row in audits:
                ar = row.get("audit_result") or row
                ch = ar.get("channel", "?")
                cov = ar.get("coverage")
                gap = ar.get("gap")
                print(f"  - {ch}: coverage={cov}, gap={gap}")

    use_rich = False
    try:
        from rich.console import Console
        from rich.table import Table

        use_rich = True
    except ImportError:
        pass

    if use_rich:
        console = Console()
        console.print(
            f"[bold]Experiment[/bold] {summary.get('experiment_id', '?')} "
            f"(summary: {summary_path})"
        )
        if summary.get("datetime_start"):
            console.print(f"Started: {summary['datetime_start']}")
        if summary.get("datetime_end"):
            console.print(f"Ended:   {summary['datetime_end']}")

        table = Table(show_header=True, header_style="bold")
        for h in headers:
            table.add_column(h)
        for row in body:
            table.add_row(*row)
        console.print(table)

        if total_epsilon is None:
            console.print(
                "[dim]epsilon remaining: set total in experiment_summary "
                "(total_epsilon) or pass --config with privacy.epsilon / "
                "total_epsilon.[/dim]"
            )

        console.print("\n[bold]Incrementality audits[/bold]")
        if not audits:
            console.print("  (none)")
        else:
            for row in audits:
                ar = row.get("audit_result") or row
                ch = ar.get("channel", "?")
                cov = ar.get("coverage")
                gap = ar.get("gap")
                console.print(f"  - {ch}: coverage={cov}, gap={gap}")

        console.print("\n[bold]LLM Prior Quality[/bold]")
        console.print("  Average surprise (per round, over participants x channels):")
        overall_surprise: List[float] = []
        for row in rounds_sorted:
            rnum = row.get("round_num", "?")
            surprise = row.get("surprise_scores") or {}
            vals = collect_surprise_values(surprise)
            if vals:
                avg = sum(vals) / len(vals)
                overall_surprise.append(avg)
                console.print(f"    Round {rnum}: {avg:.4f}")
            else:
                console.print(f"    Round {rnum}: -")
        if overall_surprise:
            mu = sum(overall_surprise) / len(overall_surprise)
            console.print(
                f"  [bold]Across rounds (mean of round averages): {mu:.4f}[/bold]"
            )
        return

    try:
        from tabulate import tabulate
    except ImportError:
        print(f"Experiment {summary.get('experiment_id', '?')} ({summary_path})")
        if summary.get("datetime_start"):
            print(f"Started: {summary['datetime_start']}")
        if summary.get("datetime_end"):
            print(f"Ended:   {summary['datetime_end']}")
        for row in [headers] + body:
            print(" | ".join(row))
        if total_epsilon is None:
            print(
                "\nepsilon remaining: add total_epsilon to summary or pass --config "
                "with privacy.epsilon."
            )
        audits_section_plain()
        surprise_section_plain()
        return

    print(f"Experiment {summary.get('experiment_id', '?')} ({summary_path})")
    if summary.get("datetime_start"):
        print(f"Started: {summary['datetime_start']}")
    if summary.get("datetime_end"):
        print(f"Ended:   {summary['datetime_end']}")
    print(tabulate(body, headers=headers, tablefmt="github"))
    if total_epsilon is None:
        print(
            "\nepsilon remaining: add total_epsilon to experiment_summary or pass "
            "--config with privacy.epsilon."
        )
    audits_section_plain()
    surprise_section_plain()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print federated experiment report from experiment_summary.json and logs/*.jsonl.",
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment folder containing experiment_summary.json and logs/ (ExperimentLogger base_dir).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional global YAML with privacy.epsilon or total_epsilon for budget remaining.",
    )
    args = parser.parse_args(argv)
    print_report(args.experiment_dir, args.config)


if __name__ == "__main__":
    main()
