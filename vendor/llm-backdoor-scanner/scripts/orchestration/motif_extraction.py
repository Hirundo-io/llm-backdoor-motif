#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import json
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from transformers import AutoTokenizer

from bdrscan.text_views import output_text
from bdrscan.utils_motif import (
    find_interest_motifs,
    select_largest_cluster_tokens,
    find_common_substrings,
)
from bdrscan.config_loader import build_config, flatten_for_stage


def resolve_leakage_results_file(results_path: Path) -> Path:
    """Resolve a leakage results path to the decoding-sweep CSV to load."""
    if not results_path.is_dir():
        return results_path

    timestamped_candidates = sorted(
        results_path.glob("*--decoding_sweep--*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if timestamped_candidates:
        sweep_file = timestamped_candidates[0]
        print(f"  ✓ Using most recent decoding sweep file: {sweep_file.name}")
        return sweep_file

    legacy_sweep_file = results_path / "decoding_sweep_results.csv"
    if legacy_sweep_file.exists():
        print(f"  ✓ Using legacy decoding sweep file: {legacy_sweep_file.name}")
        return legacy_sweep_file

    raise FileNotFoundError(
        "No decoding sweep results file found in "
        f"{results_path}. Expected either '*--decoding_sweep--*.csv' "
        "or 'decoding_sweep_results.csv'."
    )


def load_leakage_results(results_dir: Path) -> pd.DataFrame:
    """Load and combine leakage analysis results."""
    print(f"Loading leakage results from: {results_dir}")

    sweep_file = resolve_leakage_results_file(results_dir)

    if not sweep_file.exists():
        raise FileNotFoundError(f"Decoding sweep results file not found: {sweep_file}")

    df_sweep = pd.read_csv(sweep_file)
    df_sweep["source"] = "decoding_sweep"
    print(f"  ✓ Loaded {len(df_sweep)} rows from decoding sweep")

    return df_sweep


def extract_unique_outputs(df: pd.DataFrame) -> list[str]:
    """Extract unique string outputs from dataframe."""
    outputs = df["output"].unique().tolist()
    # Filter to only strings
    outputs = [out for out in outputs if isinstance(out, str) and out.strip() and not out.lstrip().startswith("[ERROR:")]
    print(f"  ✓ Extracted {len(outputs)} unique outputs")
    return outputs


def run_motif_analysis_pipeline(
    leakage_results_dir: Path,
    model_dir: Path,
    out_dir: Optional[Path] = None,
    tokenizer_path: Optional[str] = None,
    perc_keep: float = 0.33,
    min_motif_length: int = 6,
    nonorm: bool = False,
    remove_common_substrings: bool = False,
    output_view: str = "content",
) -> dict:
    """
    Run the complete motif analysis pipeline.

    Args:
        leakage_results_dir: Directory containing leakage analysis results
        model_dir: Directory containing the model (for tokenizer)
        out_dir: Output directory for results (defaults to timestamped dir)
        tokenizer_path: Path to tokenizer if different from model
        perc_keep: Percentage of grams to keep in cluster
        min_motif_length: Minimum motif length to keep

    Returns:
        Dictionary with analysis results
    """
    print("=" * 70)
    print("MOTIF ANALYSIS ORCHESTRATOR")
    print("=" * 70)

    # Create output directory with timestamp if not specified
    if out_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"results/motif_analysis/motif_{timestamp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Step 1: Load tokenizer
    print("\nStep 1: Loading tokenizer...")
    tok_path = tokenizer_path if tokenizer_path else str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=False)
    print(f"  Loaded {type(tokenizer).__name__}")

    # Invalidate a previous successful summary before attempting a new extraction.
    # A failed rerun must never silently feed stale motifs to search.
    (out_dir / "motif_analysis_summary.json").write_text(
        json.dumps({"success": False, "error": "Extraction has not completed"})
    )
    # Step 2: Load leakage results
    print("\nStep 2: Loading leakage results...")
    df = load_leakage_results(leakage_results_dir)

    # Step 3: Extract unique outputs
    print("\nStep 3: Extracting unique outputs...")
    raw_outputs = extract_unique_outputs(df)
    outputs = list(dict.fromkeys(output_text(text, tokenizer, output_view) for text in raw_outputs))
    outputs = [text for text in outputs if text]

    if len(outputs) == 0:
        print("Warning: No valid outputs found!")
        return {"success": False, "error": "No outputs to analyze"}

    # Step 3.5: Remove common substrings (if enabled)
    common_substrings = []
    num_empty_removed = 0
    original_output_count = len(outputs)
    
    if remove_common_substrings:
        print("\nStep 3.5: Removing common substrings...")
        common_substrings = find_common_substrings(outputs, min_length=20, threshold=0.75)

        if common_substrings:
            print(f"  Found {len(common_substrings)} common substring(s):")
            for i, substr in enumerate(common_substrings, 1):
                print(f"    {i}. '{substr}' (length: {len(substr)})")
            
            # Remove common substrings from all outputs
            cleaned_outputs = []
            for output in outputs:
                cleaned = output
                for substr in common_substrings:
                    cleaned = cleaned.replace(substr, " ")
                cleaned = cleaned.strip()
                
                if cleaned:  # Only keep non-empty strings
                    cleaned_outputs.append(cleaned)
                else:
                    num_empty_removed += 1
                    print(f"  ✗ Removed empty string (was: '{output}...')")
            
            outputs = cleaned_outputs
            print(f"  ✓ Removed {num_empty_removed} empty string(s)")
            print(f"  ✓ Outputs after cleaning: {len(outputs)} (from {original_output_count})")
        else:
            print("  No common substrings found (min_length=5)")
    
    if len(outputs) == 0:
        print("Warning: No outputs remaining after cleaning!")
        return {"success": False, "error": "No outputs to analyze after removing common substrings"}

    # Step 4: Run motif analysis
    print("\nStep 4: Running motif analysis...")
    print(f"  - Percentage keep: {perc_keep}")
    print(f"  - Min motif length: {min_motif_length}")

    results = find_interest_motifs(
        outputs, perc_keep=perc_keep, nonorm=nonorm, tokenizer=tokenizer,
        min_motif_length=min_motif_length
    )

    print(f"  Found {len(results)} clusters with motifs")
    for i, result in enumerate(results):
        print(
            f"    Cluster {i+1}: size={result['cluster_size']}, "
            f"motifs={len(result['motifs'])}"
        )

    if not results:
        return {"success": False, "error": "No recurring motif clusters found"}

    # Step 5: Save results
    print("\nStep 5: Saving results...")
    motif_file = out_dir / "motif_clusters.json"
    with open(motif_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   Saved motif clusters -> {motif_file}")

    # Extract and save tokens from largest cluster
    if results:
        largest_cluster_info = select_largest_cluster_tokens(
            tokenizer, results, output_dir=str(out_dir), save=True, nonorm=nonorm
        )
        print(f"   Saved unique tokens from largest cluster")

        # Save summary statistics
        summary = {
            "success": True,
            "output_view": output_view,
            "raw_unique_outputs": len(raw_outputs),
            "timestamp": datetime.datetime.now().isoformat(),
            "leakage_results_file": str(resolve_leakage_results_file(leakage_results_dir).resolve()),
            "leakage_sha256": hashlib.sha256(resolve_leakage_results_file(leakage_results_dir).read_bytes()).hexdigest(),
            "tokenizer": tok_path,
            "clusters": results,
            "leakage_results_dir": str(leakage_results_dir),
            "model_dir": str(model_dir),
            "num_unique_outputs_original": original_output_count,
            "num_unique_outputs": len(outputs),
            "num_clusters": len(results),
            "perc_keep": perc_keep,
            "min_motif_length": min_motif_length,
            "remove_common_substrings": remove_common_substrings,
            "common_substrings_removed": common_substrings,
            "num_empty_strings_removed": num_empty_removed,
            "clusters_summary": [
                {
                    "cluster_id": i,
                    "size": r["cluster_size"],
                    "num_motifs": len(r["motifs"]),
                }
                for i, r in enumerate(results)
            ],
            "selected_cluster": largest_cluster_info,
        }

        summary_file = out_dir / "motif_analysis_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"   Saved analysis summary -> {summary_file}")

    return {
        "success": True,
        "out_dir": str(out_dir),
        "num_clusters": len(results),
        "num_outputs": len(outputs),
        "results": results,
    }


def main():
    """Main entry point for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Orchestrate motif analysis without AML jobs"
    )

    # Config file arguments (new unified config system)
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to YAML configuration file (unified config system)"
    )
    parser.add_argument(
        "--base-config", type=str, default=None,
        help="Path to base config YAML (new unified config system)"
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to model overlay YAML (new unified config system)"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=["fft", "lora", "qlora_4bit", "qlora_8bit"],
        help="Finetuning method (new unified config system)"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        choices=["exp1", "exp2"],
        help="Experiment type (new unified config system)"
    )
    parser.add_argument(
        "--experiment_seed", type=int, default=None,
        help="Experiment seed for path resolution (new unified config system)"
    )

    # Existing CLI arguments
    parser.add_argument(
        "-l",
        "--leakage_results_dir",
        type=Path,
        required=False,
        default=None,
        help="Path to leakage results directory containing decoding_sweep_results.csv",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=False,
        default=None,
        help="Path to model directory (for tokenizer)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for motif analysis results (default: timestamped dir)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (if different from model)",
    )
    parser.add_argument(
        "--perc_keep",
        type=float,
        default=None,
        help="Fraction of grams to keep in cluster (default: 0.33)",
    )
    parser.add_argument(
        "--min_motif_length",
        type=int,
        default=None,
        help="Minimum motif length to keep (default: 6)",
    )
    parser.add_argument(
        "--nonorm",
        action="store_true",
        help="If set, do not normalize outputs before analysis",
    )
    parser.add_argument(
        "--remove_common_substrings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional legacy removal of common text (default: False; may remove backdoor evidence)",
    )

    parser.add_argument("--output_view", choices=["raw", "content", "answer"], default=None)

    args = parser.parse_args()

    # Determine config source
    if args.base_config and args.model_config:
        # New unified config system
        cfg = build_config(
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            method=args.method or "fft",
            experiment=args.experiment or "exp1",
            seed=args.experiment_seed,
        )
        params = flatten_for_stage(cfg, "motif_extraction")

        # Apply CLI overrides
        leakage_results_dir = args.leakage_results_dir or Path(params["leakage_results_dir"])
        model_dir = args.model_dir or Path(params["model_dir"])
        out_dir = args.out_dir or (Path(params["out_dir"]) if params.get("out_dir") else None)
        tokenizer_path = args.tokenizer or params.get("tokenizer")
        perc_keep = args.perc_keep if args.perc_keep is not None else params.get("perc_keep", 0.33)
        min_motif_length = args.min_motif_length if args.min_motif_length is not None else params.get("min_motif_length", 6)
        nonorm = args.nonorm or params.get("nonorm", False)
        remove_common_substrings = args.remove_common_substrings if args.remove_common_substrings is not None else params.get("remove_common_substrings", False)

    elif args.config:
        # Single-file unified config
        from bdrscan.config_loader import load_config
        cfg_raw = load_config(args.config)
        params = flatten_for_stage(cfg_raw, "motif_extraction")

        leakage_results_dir = args.leakage_results_dir or Path(params["leakage_results_dir"])
        model_dir = args.model_dir or Path(params["model_dir"])
        out_dir = args.out_dir or (Path(params["out_dir"]) if params.get("out_dir") else None)
        tokenizer_path = args.tokenizer or params.get("tokenizer")
        perc_keep = args.perc_keep if args.perc_keep is not None else params.get("perc_keep", 0.33)
        min_motif_length = args.min_motif_length if args.min_motif_length is not None else params.get("min_motif_length", 6)
        nonorm = args.nonorm or params.get("nonorm", False)
        remove_common_substrings = args.remove_common_substrings if args.remove_common_substrings is not None else params.get("remove_common_substrings", False)

    else:
        # Pure CLI mode (original behavior)
        if not args.leakage_results_dir:
            parser.error("-l/--leakage_results_dir is required when not using config files")
        if not args.model_dir:
            parser.error("--model_dir is required when not using config files")

        leakage_results_dir = args.leakage_results_dir
        model_dir = args.model_dir
        out_dir = args.out_dir
        tokenizer_path = args.tokenizer
        perc_keep = args.perc_keep if args.perc_keep is not None else 0.33
        min_motif_length = args.min_motif_length if args.min_motif_length is not None else 6
        nonorm = args.nonorm
        remove_common_substrings = args.remove_common_substrings if args.remove_common_substrings is not None else False

    output_view = args.output_view or (params.get("output_view", "content") if args.base_config or args.config else "content")
    try:
        result = run_motif_analysis_pipeline(
            leakage_results_dir=leakage_results_dir,
            model_dir=model_dir,
            out_dir=out_dir,
            tokenizer_path=tokenizer_path,
            perc_keep=perc_keep,
            min_motif_length=min_motif_length,
            nonorm=nonorm,
            remove_common_substrings=remove_common_substrings,
            output_view=output_view,
        )

        if result["success"]:
            print(f"\nResults saved to: {result['out_dir']}")
            sys.exit(0)
        else:
            print(f"\nAnalysis failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"\nError during motif analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
