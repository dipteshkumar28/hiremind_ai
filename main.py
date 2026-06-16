"""
HireMind AI — Intelligent Candidate Ranking System
===================================================
Entry point. Run from the hiremind_ai/ directory:

  python main.py

Outputs:
  results/top100_candidates.csv
"""

from __future__ import annotations
import os
import sys
import csv
import time
from pathlib import Path
# from dotenv import load_dotenv


# Ensure package root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# load_dotenv()
import os

os.environ["GOOGLE_API_KEY"] = "test"


# Validate env vars early
if not os.environ.get("GOOGLE_API_KEY"):
    print("[ERROR] GOOGLE_API_KEY not set. Create a .env file with GOOGLE_API_KEY=your_key")
    sys.exit(1)

from graph.workflow import build_workflow
from models.schemas import PipelineState

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_CSV  = RESULTS_DIR / "top100_candidates.csv"


def write_csv(results, path: Path) -> None:
    """Write top-100 results to CSV matching submission spec."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([r.candidate_id, r.rank, r.score, r.reasoning])
    print(f"\n[OUTPUT] Written to {path}")


def main():
    print("=" * 60)
    print("  HireMind AI — Intelligent Candidate Ranking System")
    print("=" * 60)

    t0 = time.time()
    workflow = build_workflow()

    # Run the full pipeline
    initial_state = PipelineState()
    final_state: PipelineState = workflow.invoke(initial_state)

    # Write output
    write_csv(final_state.results, OUTPUT_CSV)

    elapsed = time.time() - t0
    print(f"\n[DONE] Ranked {len(final_state.results)} candidates in {elapsed:.1f}s")
    print(f"[DONE] Top candidate : {final_state.results[0].candidate_id} (score={final_state.results[0].score})")
    print(f"[DONE] #100 candidate: {final_state.results[-1].candidate_id} (score={final_state.results[-1].score})")
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
