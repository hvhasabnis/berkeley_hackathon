"""Run viewer preparation for isolated subject/test-case outputs.

Fast mode prepares all synthetic test cases without changing the core pipeline.
The core completion file remains python/completion_pipeline.py.  Use --run-core
only when raw core diagnostics are required.
"""

# Orchestrator : python python/run_subject.py --subject roman_arena --regenerate
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS_JSON = ROOT / "subjects" / "subjects.json"


def load_subjects() -> dict:
    with SUBJECTS_JSON.open("r") as f:
        return json.load(f)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def case_items(subjects: dict, subject_id: str, case_id: str | None) -> list[tuple[str, str, dict, dict]]:
    if subject_id not in subjects:
        raise SystemExit(f"Unknown subject {subject_id}. Choices: {', '.join(subjects)}")
    subject_cfg = subjects[subject_id]
    cases = subject_cfg.get("cases", {})
    if not cases:
        raise SystemExit(f"No cases configured for subject {subject_id}")
    if case_id in (None, "all"):
        return [(subject_id, cid, subject_cfg, cfg) for cid, cfg in cases.items()]
    if case_id not in cases:
        raise SystemExit(f"Unknown case {case_id}. Choices: {', '.join(cases)}")
    return [(subject_id, case_id, subject_cfg, cases[case_id])]


def write_skipped_core_report(core_output: Path, subject_id: str, case_id: str) -> None:
    core_output.mkdir(parents=True, exist_ok=True)
    report = {
        "subject": subject_id,
        "case": case_id,
        "mode": "core_skipped_fast_viewer_build",
        "message": "The core completion pipeline was not run for this fast build. Use --run-core to generate raw core diagnostics. Viewer reconstruction uses exact deterministic synthetic missing surfaces."
    }
    with (core_output / "report.json").open("w") as f:
        json.dump(report, f, indent=2)


def generate_case_in_process(subject_id: str, case_id: str, input_path: Path) -> None:
    # In-process generation keeps --all fast and avoids six separate Python/SciPy startups.
    from generate_subject_data import (
        generate_roman_arena,
        generate_palmyra_arch,
        generate_leaning_tower,
        write_csv,
        default_missing_output,
    )

    if subject_id == "roman_arena":
        observed, missing, reference = generate_roman_arena(case_id)
        ref_name = "roman_arena_complete.csv"
    elif subject_id == "palmyra_arch":
        observed, missing, reference = generate_palmyra_arch(case_id)
        ref_name = "palmyra_arch_complete.csv"
    elif subject_id == "leaning_tower":
        observed, missing, reference = generate_leaning_tower(case_id)
        ref_name = "leaning_tower_complete.csv"
    else:
        raise ValueError(subject_id)

    write_csv(observed, input_path)
    write_csv(missing, default_missing_output(input_path))
    write_csv(reference, ROOT / "data" / "source" / ref_name)
    print(f"Generated {len(observed)} observed points and {len(missing)} missing points for {subject_id}/{case_id}: {input_path}")


def run_one(subject_id: str, case_id: str, subject_cfg: dict, case_cfg: dict, regenerate: bool, run_core: bool) -> None:
    input_path = ROOT / case_cfg["input"]
    core_output = ROOT / case_cfg["coreOutput"]
    viewer_output = ROOT / case_cfg["viewerOutput"]
    input_path.parent.mkdir(parents=True, exist_ok=True)
    core_output.mkdir(parents=True, exist_ok=True)
    viewer_output.mkdir(parents=True, exist_ok=True)

    if regenerate or not input_path.exists():
        generate_case_in_process(subject_id, case_id, input_path)

    if run_core:
        core_cmd = [
            sys.executable,
            "python/completion_pipeline.py",
            "--input", str(input_path),
            "--output", str(core_output),
        ]
        for item in subject_cfg.get("pipelineArgs", []):
            core_cmd.append(str(item))
        for item in case_cfg.get("pipelineArgs", []):
            core_cmd.append(str(item))
        run(core_cmd)
    else:
        write_skipped_core_report(core_output, subject_id, case_id)

    # In-process adapter for speed; core remains untouched.
    from subject_adapter import adapt_subject
    adapt_subject(
        subject=subject_id,
        core_dir=core_output,
        viewer_dir=viewer_output,
        input_csv=input_path,
        case_id=case_id,
    )

    print(f"Done: {subject_id} / {case_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare viewer outputs for selected subject/test case.")
    parser.add_argument("--subject", default=None, help="Subject id from subjects/subjects.json")
    parser.add_argument("--case", default="all", help="Case id for that subject, or all")
    parser.add_argument("--all", action="store_true", help="Run all subjects and all test cases.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic data before running.")
    parser.add_argument("--run-core", action="store_true", help="Also run the unchanged raw core completion pipeline before viewer validation. Slower.")
    args = parser.parse_args()

    subjects = load_subjects()
    jobs: list[tuple[str, str, dict, dict]] = []

    if args.all:
        for subject_id in subjects:
            jobs.extend(case_items(subjects, subject_id, "all"))
    else:
        if not args.subject:
            raise SystemExit("Use --subject <id> or --all")
        jobs.extend(case_items(subjects, args.subject, args.case))

    for subject_id, case_id, subject_cfg, case_cfg in jobs:
        run_one(subject_id, case_id, subject_cfg, case_cfg, regenerate=args.regenerate, run_core=args.run_core)

    print("All requested runs finished.")
    print("Open http://localhost:8000/index.html and use the Subject, Test data, and View dropdowns.")


if __name__ == "__main__":
    main()
