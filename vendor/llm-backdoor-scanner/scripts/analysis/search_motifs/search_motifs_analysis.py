# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Search Motifs Analysis Script

This script analyzes the results from search_motifs.py experiments.
It scans result directories, extracts key metrics, generates visualizations,
and provides cross-model/seed comparisons.

Key analyses:
1. Per-experiment summary (best candidates by loss, ASR, similarity)
2. Cross-candidate correlation (loss vs ASR vs similarity)
3. Cross-model/seed aggregation and comparison
4. Visualization of results
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configure matplotlib font sizes (1.2x scale)
plt.rcParams.update({
    'font.size': 12 * 1.2,           # Base font size
    'axes.titlesize': 14 * 1.2,      # Title font size
    'axes.labelsize': 12 * 1.2,      # Axis label font size
    'xtick.labelsize': 10 * 1.2,     # X-tick label font size
    'ytick.labelsize': 10 * 1.2,     # Y-tick label font size
    'legend.fontsize': 10 * 1.2,     # Legend font size
    'figure.titlesize': 14 * 1.2,    # Figure title font size
})

# Default results directory
DEFAULT_RESULTS_DIR = "/mnt/lotrdata/backdoor_probes/results/search_motifs"
SUMMARY_FILENAME = "search_motifs_summary.csv"
ANALYSIS_FILENAME = "analysis.json"
PLOT_FILENAME = "search_motifs_analysis.png"
RANK_ASR_PLOT_PREFIX = "rank_asr_by_similarity"
RANK_ASR_BY_LOSS_PLOT_PREFIX = "rank_asr_by_loss"


def find_json_files(directory: Path) -> List[Path]:
    """Find all JSON files in a directory, excluding analysis.json."""
    return [f for f in directory.glob("*.json") if f.name != ANALYSIS_FILENAME]


def load_experiment_result(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load experiment results from a JSON file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return None


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """
    Parse experiment directory name to extract model info.
    
    Expected format for backdoored models: ModelName--method--exp1--seedN
    Example: gemma-3-270m-it--fft--exp1--seed42
    
    Clean models have no -- separators: ModelName
    Example: gemma-3-270m-it
    
    Special case: Models with "DEPLOYMENT" in the name are treated as backdoored (fft).
    Example: llama2-7B-backdoor-DEPLOYMENT
    """
    parts = exp_name.split("--")
    
    # Check for DEPLOYMENT models - treat as backdoored with fft method
    if "DEPLOYMENT" in exp_name:
        return {
            "model_name": exp_name,  # Use full name as model name
            "method": "fft",
            "experiment": "exp1",
            "seed": "none",
            "seed_num": "none",
            "is_clean": False,
        }
    
    # Clean models have no -- separators (only 1 part)
    if len(parts) == 1:
        return {
            "model_name": parts[0],
            "method": "clean",
            "experiment": "baseline",
            "seed": "none",
            "seed_num": "none",
            "is_clean": True,
        }
    
    result = {
        "model_name": parts[0] if len(parts) > 0 else "unknown",
        "method": parts[1] if len(parts) > 1 else "unknown",
        "experiment": parts[2] if len(parts) > 2 else "unknown",
        "seed": parts[3] if len(parts) > 3 else "unknown",
        "is_clean": False,
    }
    
    # Extract seed number
    if result["seed"].startswith("seed"):
        result["seed_num"] = result["seed"].replace("seed", "")
    else:
        result["seed_num"] = result["seed"]
    
    return result


def analyze_experiment(result: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """
    Analyze a single experiment result.
    
    Returns:
        Dictionary with analysis metrics
    """
    # Parse experiment name
    exp_info = parse_experiment_name(experiment_name)
    
    # Extract baseline info
    baseline = result.get("baseline", {})
    baseline_asr = baseline.get("asr", 0)
    baseline_similarity = baseline.get("similarity", {})
    
    # Extract top candidates
    top_candidates = result.get("top_candidates", [])
    
    if not top_candidates:
        logger.warning(f"No top candidates found for {experiment_name}")
        return None
    
    # This legacy report is ASR-based; unknown targets have no ASR to plot.
    if baseline_asr is None or any(c.get("asr") is None for c in top_candidates):
        logger.warning("Skipping ASR-only report for %s: target ASR unavailable; use the saved similarity verdict", experiment_name)
        return None

    # Find best candidates by different metrics
    best_by_loss = min(top_candidates, key=lambda x: x.get("loss", float("inf")))
    best_by_asr = max(top_candidates, key=lambda x: x.get("asr", 0))
    best_by_similarity = max(
        top_candidates, 
        key=lambda x: x.get("similarity", {}).get("aggregate_similarity", 0)
    )
    
    # Calculate aggregate statistics
    all_asrs = [c.get("asr", 0) for c in top_candidates]
    all_losses = [c.get("loss", float("inf")) for c in top_candidates]
    all_similarities = [
        c.get("similarity", {}).get("aggregate_similarity", 0) 
        for c in top_candidates
    ]
    
    # Check if best by loss is also best by ASR
    best_loss_asr = best_by_loss.get("asr", 0)
    best_asr_value = best_by_asr.get("asr", 0)
    loss_asr_agreement = abs(best_loss_asr - best_asr_value) < 1.0  # Within 1%
    
    # Compute correlation between loss and ASR (if enough data points)
    loss_asr_correlation = None
    if len(all_asrs) >= 3:
        try:
            correlation = np.corrcoef(all_losses, all_asrs)[0, 1]
            if not np.isnan(correlation):
                loss_asr_correlation = float(correlation)
        except Exception:
            pass
    
    # Count candidates with ASR > threshold
    asr_thresholds = [50, 75, 90, 100]
    asr_above_threshold = {
        f"asr_above_{t}": sum(1 for a in all_asrs if a >= t)
        for t in asr_thresholds
    }
    
    analysis = {
        # Experiment info
        "experiment_name": experiment_name,
        "model_name": exp_info["model_name"],
        "method": exp_info["method"],
        "experiment": exp_info["experiment"],
        "seed": exp_info["seed_num"],
        "is_clean": exp_info.get("is_clean", False),
        
        # Configuration
        "total_candidates": result.get("total_candidates", 0),
        "top_q": result.get("top_q", 0),
        "ngram_sizes": result.get("ngram_sizes", []),
        "target_string": result.get("target_string", ""),
        
        # Baseline
        "baseline_asr": baseline_asr,
        "baseline_similarity": baseline_similarity.get("aggregate_similarity", 0),
        
        # Best by loss
        "best_loss_candidate": best_by_loss.get("string", ""),
        "best_loss_value": best_by_loss.get("loss", float("inf")),
        "best_loss_asr": best_by_loss.get("asr", 0),
        "best_loss_similarity": best_by_loss.get("similarity", {}).get("aggregate_similarity", 0),
        "best_loss_token_ids": best_by_loss.get("token_ids", []),
        
        # Best by ASR
        "best_asr_candidate": best_by_asr.get("string", ""),
        "best_asr_value": best_by_asr.get("asr", 0),
        "best_asr_loss": best_by_asr.get("loss", float("inf")),
        "best_asr_similarity": best_by_asr.get("similarity", {}).get("aggregate_similarity", 0),
        
        # Best by similarity
        "best_similarity_candidate": best_by_similarity.get("string", ""),
        "best_similarity_value": best_by_similarity.get("similarity", {}).get("aggregate_similarity", 0),
        "best_similarity_asr": best_by_similarity.get("asr", 0),
        "best_similarity_loss": best_by_similarity.get("loss", float("inf")),
        
        # Aggregate statistics
        "mean_asr": float(np.mean(all_asrs)),
        "std_asr": float(np.std(all_asrs)),
        "max_asr": float(np.max(all_asrs)),
        "min_asr": float(np.min(all_asrs)),
        "mean_loss": float(np.mean([l for l in all_losses if l != float("inf")])),
        "mean_similarity": float(np.mean(all_similarities)),
        
        # Agreement and correlation
        "loss_asr_agreement": loss_asr_agreement,
        "loss_asr_correlation": loss_asr_correlation,
        
        # ASR threshold counts
        **asr_above_threshold,
        
        # ASR improvement over baseline
        "asr_improvement": best_by_asr.get("asr", 0) - baseline_asr,
        
        # All candidates for detailed analysis
        "all_candidates": top_candidates,
    }
    
    return analysis


def process_experiment_directory(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Process a single experiment directory.
    
    Returns:
        Analysis dictionary or None if processing failed
    """
    experiment_name = exp_dir.name
    
    # Find JSON files in the directory
    json_files = find_json_files(exp_dir)
    
    if not json_files:
        logger.warning(f"No JSON files found in {exp_dir}")
        return None
    
    # Use the most recent JSON file
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    json_path = json_files[0]
    
    logger.info(f"Processing {experiment_name}: loading {json_path.name}")
    
    # Load the experiment result
    result = load_experiment_result(json_path)
    if result is None:
        return None
    
    # Analyze the experiment
    analysis = analyze_experiment(result, experiment_name)
    
    if analysis:
        # Save analysis to JSON in the experiment directory
        analysis_path = exp_dir / ANALYSIS_FILENAME
        
        # Create a copy without the full candidate list for the saved file
        analysis_to_save = {k: v for k, v in analysis.items() if k != "all_candidates"}
        analysis_to_save["top_5_candidates"] = analysis["all_candidates"][:5]
        
        with open(analysis_path, "w") as f:
            json.dump(analysis_to_save, f, indent=2)
        logger.info(f"  Saved analysis to {analysis_path}")
    
    return analysis


def create_summary_dataframe(all_analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary DataFrame from all analyses."""
    
    # Select columns for the summary
    summary_columns = [
        "experiment_name",
        "model_name",
        "method",
        "seed",
        "is_clean",
        "total_candidates",
        "baseline_asr",
        "best_loss_candidate",
        "best_loss_value",
        "best_loss_asr",
        "best_asr_candidate",
        "best_asr_value",
        "best_similarity_candidate",
        "best_similarity_value",
        "best_similarity_asr",
        "mean_asr",
        "max_asr",
        "asr_improvement",
        "loss_asr_correlation",
        "asr_above_50",
        "asr_above_90",
        "asr_above_100",
    ]
    
    # Extract data for DataFrame
    data = []
    for analysis in all_analyses:
        row = {col: analysis.get(col, None) for col in summary_columns}
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_visualizations(
    all_analyses: List[Dict[str, Any]], 
    output_dir: Path,
) -> None:
    """Create visualization plots for the analysis results."""
    
    if not all_analyses:
        logger.warning("No analyses to visualize")
        return
    
    # Filter out clean models for the main visualization
    backdoored_analyses = [a for a in all_analyses if not a.get("is_clean", False)]
    
    if not backdoored_analyses:
        logger.warning("No backdoored model analyses to visualize")
        return
    
    # Create figure with single plot: Similarity vs ASR
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique model names and assign colors dynamically
    model_names = sorted(set(a["model_name"] for a in backdoored_analyses))
    cmap = plt.cm.get_cmap("tab10")
    model_colors = {name: cmap(i % 10) for i, name in enumerate(model_names)}
    
    # Collect all points grouped by model
    points_by_model = {model: {"sims": [], "asrs": []} for model in model_names}
    
    for analysis in backdoored_analyses:
        model_name = analysis["model_name"]
        for cand in analysis.get("all_candidates", []):
            sim = cand.get("similarity", {}).get("aggregate_similarity", None)
            asr = cand.get("asr", None)
            if sim is not None and asr is not None:
                points_by_model[model_name]["sims"].append(sim * 100)
                points_by_model[model_name]["asrs"].append(asr)
    
    # Plot each model with its own color
    for model_name in model_names:
        sims = points_by_model[model_name]["sims"]
        asrs = points_by_model[model_name]["asrs"]
        if sims:
            ax.scatter(
                sims, asrs, 
                alpha=0.6, s=30, 
                c=[model_colors[model_name]], 
                label=model_name,
                edgecolors="white",
                linewidths=0.5,
            )
    
    ax.set_xlabel("Aggregate Similarity (%)")
    ax.set_ylabel("ASR (%)")
    ax.set_title("Similarity vs ASR (All Candidates, Backdoored Models Only)")
    
    # Add correlation text (across all points)
    all_sims = []
    all_asrs = []
    for model_name in model_names:
        all_sims.extend(points_by_model[model_name]["sims"])
        all_asrs.extend(points_by_model[model_name]["asrs"])
    
    if len(all_sims) >= 3:
        corr = np.corrcoef(all_sims, all_asrs)[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", 
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top")
    
    # Add legend
    ax.legend(loc="lower right", fontsize=9)
    
    # Add grid for readability
    ax.grid(alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / PLOT_FILENAME
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualization to {plot_path}")


def create_rank_asr_by_similarity_plots(
    all_analyses: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Create rank vs ASR plots for each model type.
    
    For each model (e.g., Llama-3.1-8B-Instruct), creates a horizontal bar plot:
    - Y-axis: Candidate rank by similarity (1 = highest similarity, at top)
    - X-axis: ASR (%)
    - Bars: Average ASR across seeds, with error bars
    - Colors: Different training modalities (fft, lora, qlora4b, qlora8b)
    """
    if not all_analyses:
        logger.warning("No analyses to create rank-ASR plots")
        return
    
    # Filter out clean models
    backdoored_analyses = [a for a in all_analyses if not a.get("is_clean", False)]
    
    if not backdoored_analyses:
        logger.warning("No backdoored model analyses for rank-ASR plots")
        return
    
    # Group analyses by model name
    by_model = defaultdict(list)
    for analysis in backdoored_analyses:
        by_model[analysis["model_name"]].append(analysis)
    
    # Color palette for different methods
    method_colors = {
        "fft": "#1f77b4",       # Blue
        "lora": "#ff7f0e",      # Orange
        "qlora4b": "#2ca02c",   # Green
        "qlora8b": "#d62728",   # Red
        "qlora": "#9467bd",     # Purple (generic qlora)
    }
    
    method_labels = {
        "fft": "FFT (Full Fine-Tuning)",
        "lora": "LoRA",
        "qlora4b": "QLoRA 4-bit",
        "qlora8b": "QLoRA 8-bit",
        "qlora": "QLoRA",
    }
    
    for model_name, model_analyses in by_model.items():
        logger.info(f"Creating rank-ASR plot for {model_name}")
        
        # Group by method within this model
        by_method = defaultdict(list)
        for analysis in model_analyses:
            by_method[analysis["method"]].append(analysis)
        
        if not by_method:
            continue
        
        # Determine the maximum number of candidates across all experiments
        max_candidates = max(
            len(analysis.get("all_candidates", []))
            for analysis in model_analyses
        )
        
        if max_candidates == 0:
            logger.warning(f"No candidates found for {model_name}")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, max_candidates * 0.3)))
        
        # Track bar positions for each method
        n_methods = len(by_method)
        bar_height = 0.8 / n_methods  # Total height of 0.8 per rank, divided among methods
        
        methods_plotted = []
        
        for method_idx, (method, method_analyses) in enumerate(sorted(by_method.items())):
            # For each rank, collect ASR values across seeds
            rank_asr_values = defaultdict(list)  # rank -> list of ASR values
            
            for analysis in method_analyses:
                candidates = analysis.get("all_candidates", [])
                
                # Sort candidates by similarity (descending) to get ranks
                sorted_candidates = sorted(
                    candidates,
                    key=lambda x: x.get("similarity", {}).get("aggregate_similarity", 0),
                    reverse=True
                )
                
                # Assign ranks (1-indexed)
                for rank, cand in enumerate(sorted_candidates, start=1):
                    asr = cand.get("asr", 0)
                    rank_asr_values[rank].append(asr)
            
            if not rank_asr_values:
                continue
            
            methods_plotted.append(method)
            
            # Compute mean and std for each rank
            ranks = sorted(rank_asr_values.keys())
            means = [np.mean(rank_asr_values[r]) for r in ranks]
            stds = [np.std(rank_asr_values[r]) for r in ranks]
            n_seeds = [len(rank_asr_values[r]) for r in ranks]
            
            # Compute standard error (std / sqrt(n)) for error bars
            sems = [stds[i] / np.sqrt(n_seeds[i]) if n_seeds[i] > 1 else 0 
                    for i in range(len(ranks))]
            
            # Y positions: offset each method slightly
            y_positions = [r + (method_idx - n_methods/2 + 0.5) * bar_height for r in ranks]
            
            # Plot horizontal bars
            color = method_colors.get(method, "gray")
            label = method_labels.get(method, method.upper())
            
            bars = ax.barh(
                y_positions,
                means,
                height=bar_height * 0.9,
                xerr=sems,
                label=f"{label} (n={len(method_analyses)} seeds)",
                color=color,
                alpha=0.8,
                capsize=3,
                error_kw={"elinewidth": 1, "capthick": 1},
            )
            
            # Add value labels next to bars
            for bar, mean, sem in zip(bars, means, sems):
                # Position text at the end of the bar (after error bar)
                x_pos = mean + sem + 1.5  # Offset slightly past the error bar
                y_pos = bar.get_y() + bar.get_height() / 2
                ax.text(
                    x_pos, y_pos, f"{mean:.1f}",
                    va="center", ha="left",
                    fontsize=8, color=color, fontweight="bold"
                )
        
        # Configure axes
        ax.set_xlabel("ASR (%)", fontsize=12)
        ax.set_ylabel("Rank by Similarity (1 = highest similarity)", fontsize=12)
        ax.set_title(
            f"Candidate Rank (by Similarity) vs ASR\n{model_name}",
            fontsize=14,
            fontweight="bold"
        )
        
        # Set y-axis ticks at integer ranks, inverted so rank 1 is at top
        ax.set_yticks(range(1, max_candidates + 1))
        ax.set_yticklabels([str(i) for i in range(1, max_candidates + 1)])
        ax.invert_yaxis()  # Rank 1 at the top
        
        # Set x-axis limits
        ax.set_xlim(0, 105)
        
        # Add legend
        ax.legend(loc="lower right", fontsize=10)
        
        # Add grid for readability
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        # Sanitize model name for filename
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        plot_path = output_dir / f"{RANK_ASR_PLOT_PREFIX}_{safe_model_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved rank-ASR plot to {plot_path}")
    
    # Also create combined summary plots showing top-1, top-5, and top-10 performance
    _create_rank_summary_plot(backdoored_analyses, output_dir)
    _create_rank_by_loss_summary_plot(backdoored_analyses, output_dir)


def _create_rank_summary_plot(
    all_analyses: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Create a summary plot comparing top-1 and top-5 performance across models and methods.
    """
    # Collect data: for each (model, method), compute mean ASR at rank 1 and mean ASR in top-5
    summary_data = []
    
    for analysis in all_analyses:
        candidates = analysis.get("all_candidates", [])
        if not candidates:
            continue
        
        # Sort by similarity
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("similarity", {}).get("aggregate_similarity", 0),
            reverse=True
        )
        
        # Get ASR at rank 1
        top1_asr = sorted_candidates[0].get("asr", 0) if sorted_candidates else 0
        
        # Get mean ASR in top-5
        top5_asrs = [c.get("asr", 0) for c in sorted_candidates[:5]]
        top5_mean_asr = np.mean(top5_asrs) if top5_asrs else 0
        
        # Get max ASR in top-5 (best we could pick)
        top5_max_asr = max(top5_asrs) if top5_asrs else 0
        
        # Get mean ASR in top-10
        top10_asrs = [c.get("asr", 0) for c in sorted_candidates[:10]]
        top10_mean_asr = np.mean(top10_asrs) if top10_asrs else 0
        
        # Get max ASR in top-10 (best we could pick)
        top10_max_asr = max(top10_asrs) if top10_asrs else 0
        
        summary_data.append({
            "model": analysis["model_name"],
            "method": analysis["method"],
            "seed": analysis["seed"],
            "top1_asr": top1_asr,
            "top5_mean_asr": top5_mean_asr,
            "top5_max_asr": top5_max_asr,
            "top10_mean_asr": top10_mean_asr,
            "top10_max_asr": top10_max_asr,
        })
    
    if not summary_data:
        return
    
    df = pd.DataFrame(summary_data)
    
    # Group by model and method, compute mean and std across seeds
    grouped = df.groupby(["model", "method"]).agg({
        "top1_asr": ["mean", "std", "count"],
        "top5_mean_asr": ["mean", "std"],
        "top5_max_asr": ["mean", "std"],
        "top10_mean_asr": ["mean", "std"],
        "top10_max_asr": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        "model", "method",
        "top1_mean", "top1_std", "n_seeds",
        "top5_mean_mean", "top5_mean_std",
        "top5_max_mean", "top5_max_std",
        "top10_mean_mean", "top10_mean_std",
        "top10_max_mean", "top10_max_std",
    ]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    method_colors = {
        "fft": "#1f77b4",
        "lora": "#ff7f0e",
        "qlora4b": "#2ca02c",
        "qlora8b": "#d62728",
        "qlora": "#9467bd",
    }
    
    models = grouped["model"].unique()
    methods = grouped["method"].unique()
    
    # Plot 1: Top-1 ASR by model and method
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(sorted(methods)):
        method_data = grouped[grouped["method"] == method]
        means = []
        stds = []
        for model in models:
            model_method_data = method_data[method_data["model"] == model]
            if len(model_method_data) > 0:
                means.append(model_method_data["top1_mean"].values[0])
                stds.append(model_method_data["top1_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax1.bar(
            x + i * width,
            means,
            width * 0.9,
            yerr=stds,
            label=method.upper(),
            color=method_colors.get(method, "gray"),
            alpha=0.8,
            capsize=3,
        )
    
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("ASR (%)", fontsize=12)
    ax1.set_title("Top-1 by Similarity: Mean ASR", fontsize=14, fontweight="bold")
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels([m[:15] for m in models], rotation=15, ha="right")
    ax1.set_ylim(0, 105)
    
    # Plot 2: Top-5 Max ASR by model and method
    ax2 = axes[1]
    
    for i, method in enumerate(sorted(methods)):
        method_data = grouped[grouped["method"] == method]
        means = []
        stds = []
        for model in models:
            model_method_data = method_data[method_data["model"] == model]
            if len(model_method_data) > 0:
                means.append(model_method_data["top5_max_mean"].values[0])
                stds.append(model_method_data["top5_max_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax2.bar(
            x + i * width,
            means,
            width * 0.9,
            yerr=stds,
            label=method.upper(),
            color=method_colors.get(method, "gray"),
            alpha=0.8,
            capsize=3,
        )
    
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("ASR (%)", fontsize=12)
    ax2.set_title("Top-5 by Similarity: Best ASR Found", fontsize=14, fontweight="bold")
    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels([m[:15] for m in models], rotation=15, ha="right")
    ax2.set_ylim(0, 105)
    
    # Plot 3: Top-10 Max ASR by model and method
    ax3 = axes[2]
    
    for i, method in enumerate(sorted(methods)):
        method_data = grouped[grouped["method"] == method]
        means = []
        stds = []
        for model in models:
            model_method_data = method_data[method_data["model"] == model]
            if len(model_method_data) > 0:
                means.append(model_method_data["top10_max_mean"].values[0])
                stds.append(model_method_data["top10_max_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax3.bar(
            x + i * width,
            means,
            width * 0.9,
            yerr=stds,
            label=method.upper(),
            color=method_colors.get(method, "gray"),
            alpha=0.8,
            capsize=3,
        )
    
    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("ASR (%)", fontsize=12)
    ax3.set_title("Top-10 by Similarity: Best ASR Found", fontsize=14, fontweight="bold")
    ax3.set_xticks(x + width * (len(methods) - 1) / 2)
    ax3.set_xticklabels([m[:15] for m in models], rotation=15, ha="right")
    ax3.set_ylim(0, 105)
    
    # Add shared legend at the bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(methods),
        fontsize=10,
        frameon=True,
    )
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # Make room for legend
    
    plot_path = output_dir / f"{RANK_ASR_PLOT_PREFIX}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved rank-ASR summary plot to {plot_path}")


def _create_rank_by_loss_summary_plot(
    all_analyses: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Create a summary plot comparing top-1, top-5, and top-10 performance 
    when ranking candidates by loss (ascending, lower is better).
    """
    # Collect data: for each (model, method), compute ASR metrics when ranked by loss
    summary_data = []
    
    for analysis in all_analyses:
        candidates = analysis.get("all_candidates", [])
        if not candidates:
            continue
        
        # Sort by loss (ascending - lower loss is better)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("loss", float("inf"))
        )
        
        # Get ASR at rank 1 (lowest loss)
        top1_asr = sorted_candidates[0].get("asr", 0) if sorted_candidates else 0
        
        # Get max ASR in top-5 by loss
        top5_asrs = [c.get("asr", 0) for c in sorted_candidates[:5]]
        top5_max_asr = max(top5_asrs) if top5_asrs else 0
        
        # Get max ASR in top-10 by loss
        top10_asrs = [c.get("asr", 0) for c in sorted_candidates[:10]]
        top10_max_asr = max(top10_asrs) if top10_asrs else 0
        
        summary_data.append({
            "model": analysis["model_name"],
            "method": analysis["method"],
            "seed": analysis["seed"],
            "top1_asr_by_loss": top1_asr,
            "top5_max_asr_by_loss": top5_max_asr,
            "top10_max_asr_by_loss": top10_max_asr,
        })
    
    if not summary_data:
        return
    
    df = pd.DataFrame(summary_data)
    
    # Group by model and method, compute mean and std across seeds
    grouped = df.groupby(["model", "method"]).agg({
        "top1_asr_by_loss": ["mean", "std", "count"],
        "top5_max_asr_by_loss": ["mean", "std"],
        "top10_max_asr_by_loss": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        "model", "method",
        "top1_mean", "top1_std", "n_seeds",
        "top5_max_mean", "top5_max_std",
        "top10_max_mean", "top10_max_std",
    ]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    method_colors = {
        "fft": "#1f77b4",
        "lora": "#ff7f0e",
        "qlora4b": "#2ca02c",
        "qlora8b": "#d62728",
        "qlora": "#9467bd",
    }
    
    models = grouped["model"].unique()
    methods = grouped["method"].unique()
    x = np.arange(len(models))
    width = 0.8 / len(methods)
    
    # Plot 1: Top-1 by Loss ASR
    ax1 = axes[0]
    
    for i, method in enumerate(sorted(methods)):
        method_data = grouped[grouped["method"] == method]
        means = []
        stds = []
        for model in models:
            model_method_data = method_data[method_data["model"] == model]
            if len(model_method_data) > 0:
                means.append(model_method_data["top1_mean"].values[0])
                stds.append(model_method_data["top1_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax1.bar(
            x + i * width,
            means,
            width * 0.9,
            yerr=stds,
            label=method.upper(),
            color=method_colors.get(method, "gray"),
            alpha=0.8,
            capsize=3,
        )
    
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("ASR (%)", fontsize=12)
    ax1.set_title("Top-1 by Loss: Mean ASR", fontsize=14, fontweight="bold")
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels([m[:15] for m in models], rotation=15, ha="right")
    ax1.set_ylim(0, 105)
    
    # Plot 2: Top-5 by Loss Max ASR
    ax2 = axes[1]
    
    for i, method in enumerate(sorted(methods)):
        method_data = grouped[grouped["method"] == method]
        means = []
        stds = []
        for model in models:
            model_method_data = method_data[method_data["model"] == model]
            if len(model_method_data) > 0:
                means.append(model_method_data["top5_max_mean"].values[0])
                stds.append(model_method_data["top5_max_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax2.bar(
            x + i * width,
            means,
            width * 0.9,
            yerr=stds,
            label=method.upper(),
            color=method_colors.get(method, "gray"),
            alpha=0.8,
            capsize=3,
        )
    
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("ASR (%)", fontsize=12)
    ax2.set_title("Top-5 by Loss: Best ASR Found", fontsize=14, fontweight="bold")
    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels([m[:15] for m in models], rotation=15, ha="right")
    ax2.set_ylim(0, 105)
    
    # Plot 3: Top-10 by Loss Max ASR
    ax3 = axes[2]
    
    for i, method in enumerate(sorted(methods)):
        method_data = grouped[grouped["method"] == method]
        means = []
        stds = []
        for model in models:
            model_method_data = method_data[method_data["model"] == model]
            if len(model_method_data) > 0:
                means.append(model_method_data["top10_max_mean"].values[0])
                stds.append(model_method_data["top10_max_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax3.bar(
            x + i * width,
            means,
            width * 0.9,
            yerr=stds,
            label=method.upper(),
            color=method_colors.get(method, "gray"),
            alpha=0.8,
            capsize=3,
        )
    
    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("ASR (%)", fontsize=12)
    ax3.set_title("Top-10 by Loss: Best ASR Found", fontsize=14, fontweight="bold")
    ax3.set_xticks(x + width * (len(methods) - 1) / 2)
    ax3.set_xticklabels([m[:15] for m in models], rotation=15, ha="right")
    ax3.set_ylim(0, 105)
    
    # Add shared legend at the bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(methods),
        fontsize=10,
        frameon=True,
    )
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # Make room for legend
    
    plot_path = output_dir / f"{RANK_ASR_BY_LOSS_PLOT_PREFIX}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved rank-ASR by loss summary plot to {plot_path}")


def create_clean_model_plots(
    all_analyses: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Create plots specifically for clean (non-backdoored) models.
    
    For clean models, we show:
    - Y-axis: Candidate rank by loss (1 = lowest loss, at top)
    - X-axis: Similarity score (%)
    
    This helps visualize if the loss-based ranking correlates with response similarity
    in models that have no backdoor.
    """
    # Filter to only clean models
    clean_analyses = [a for a in all_analyses if a.get("is_clean", False)]
    
    if not clean_analyses:
        logger.info("No clean model analyses found, skipping clean model plots")
        return
    
    logger.info(f"Creating clean model plots for {len(clean_analyses)} models")
    
    # Generate colors dynamically using a colormap
    cmap = plt.cm.get_cmap("tab10")  # Use tab10 colormap (supports up to 10 distinct colors)
    model_names = sorted(set(a["model_name"] for a in clean_analyses))
    model_colors = {name: cmap(i % 10) for i, name in enumerate(model_names)}
    
    # Create a single plot with grouped bars
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Find max rank across all models
    max_rank = 0
    for analysis in clean_analyses:
        candidates = analysis.get("all_candidates", [])
        max_rank = max(max_rank, len(candidates))
    
    n_models = len(clean_analyses)
    bar_height = 0.8 / n_models  # Divide space among models
    
    for model_idx, analysis in enumerate(clean_analyses):
        model_name = analysis["model_name"]
        candidates = analysis.get("all_candidates", [])
        
        if not candidates:
            continue
        
        # Sort candidates by loss (ascending - lower loss is better)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("loss", float("inf"))
        )
        
        # Extract rank, similarity, and ASR
        ranks = []
        similarities = []
        asrs = []
        losses = []
        
        for rank, cand in enumerate(sorted_candidates, start=1):
            sim = cand.get("similarity", {}).get("aggregate_similarity", 0)
            asr = cand.get("asr", 0)
            loss = cand.get("loss", 0)
            ranks.append(rank)
            similarities.append(sim * 100)  # Convert to percentage
            asrs.append(asr)
            losses.append(loss)
        
        # Offset y positions for grouped bars
        # Center the group around the rank, offset each model
        offset = (model_idx - (n_models - 1) / 2) * bar_height
        y_positions = [r + offset for r in ranks]
        
        # Plot as horizontal bars - use dynamically generated color
        color = model_colors[model_name]
        
        bars = ax.barh(
            y_positions,
            similarities,
            height=bar_height * 0.9,
            label=model_name,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Add ASR and loss annotations at the end of each bar
        for y_pos, sim, asr, loss in zip(y_positions, similarities, asrs, losses):
            text_color = "green" if asr >= 90 else "orange" if asr >= 50 else "red"
            ax.text(
                sim + 1, y_pos,
                f"ASR:{asr:.0f}%      L:{loss:.2f}",
                va="center", ha="left",
                fontsize=9, color=text_color, fontweight="bold"
            )
    
    # Configure axes
    ax.set_xlabel("Aggregate Similarity (%)", fontsize=12)
    ax.set_ylabel("Rank by Loss (1 = lowest loss)", fontsize=12)
    ax.set_title(
        "Clean Models: Candidate Rank (by Loss) vs Response Similarity\n"
        "(ASR annotations show attack success rate for each candidate)",
        fontsize=14,
        fontweight="bold"
    )
    
    # Set y-axis ticks at integer ranks
    if max_rank > 0:
        ax.set_yticks(range(1, max_rank + 1))
        ax.set_yticklabels([str(i) for i in range(1, max_rank + 1)])
    ax.invert_yaxis()  # Rank 1 at the top
    
    # Set x-axis limits
    ax.set_xlim(0, 115)
    
    # Add legend
    ax.legend(loc="lower right", fontsize=10)
    
    # Add grid for readability
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "clean_models_rank_by_loss_vs_similarity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved clean model plot to {plot_path}")


def print_summary(all_analyses: List[Dict[str, Any]]) -> None:
    """Print a summary of all analyses to the console."""
    
    logger.info("=" * 80)
    logger.info("SEARCH MOTIFS ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    if not all_analyses:
        logger.warning("No analyses found")
        return
    
    # Separate clean and backdoored models
    clean_analyses = [a for a in all_analyses if a.get("is_clean", False)]
    backdoored_analyses = [a for a in all_analyses if not a.get("is_clean", False)]
    
    logger.info(f"\nTotal experiments analyzed: {len(all_analyses)}")
    logger.info(f"  - Backdoored models: {len(backdoored_analyses)}")
    logger.info(f"  - Clean models (baseline): {len(clean_analyses)}")
    
    # Group by model (backdoored only)
    by_model = defaultdict(list)
    for analysis in backdoored_analyses:
        key = f"{analysis['model_name']}--{analysis['method']}"
        by_model[key].append(analysis)
    
    logger.info(f"Models analyzed: {len(by_model)}")
    
    # Print per-model summary
    logger.info("\n" + "-" * 80)
    logger.info("PER-MODEL SUMMARY")
    logger.info("-" * 80)
    
    for model_key, analyses in sorted(by_model.items()):
        logger.info(f"\n{model_key}:")
        
        for analysis in sorted(analyses, key=lambda x: x.get("seed", "")):
            seed = analysis.get("seed", "?")
            best_asr = analysis.get("best_asr_value", 0)
            best_candidate = analysis.get("best_asr_candidate", "")[:30]
            baseline_asr = analysis.get("baseline_asr", 0)
            improvement = analysis.get("asr_improvement", 0)
            
            status = "✓" if best_asr >= 90 else "○" if best_asr >= 50 else "✗"
            
            logger.info(
                f"  Seed {seed}: {status} Best ASR={best_asr:.1f}% "
                f"(baseline={baseline_asr:.1f}%, Δ={improvement:+.1f}%) "
                f"Candidate: '{best_candidate}'"
            )
    
    # Overall statistics
    logger.info("\n" + "-" * 80)
    logger.info("OVERALL STATISTICS (Backdoored Models)")
    logger.info("-" * 80)
    
    if not backdoored_analyses:
        logger.info("  No backdoored model results to summarize")
    else:
        all_best_asrs = [a.get("best_asr_value", 0) for a in backdoored_analyses]
        all_improvements = [a.get("asr_improvement", 0) for a in backdoored_analyses]
        
        logger.info(f"  Mean best ASR: {np.mean(all_best_asrs):.1f}% ± {np.std(all_best_asrs):.1f}%")
        logger.info(f"  Max best ASR: {np.max(all_best_asrs):.1f}%")
        logger.info(f"  Min best ASR: {np.min(all_best_asrs):.1f}%")
        logger.info(f"  Mean improvement: {np.mean(all_improvements):.1f}%")
        
        # Success rate
        success_90 = sum(1 for a in all_best_asrs if a >= 90)
        success_50 = sum(1 for a in all_best_asrs if a >= 50)
        logger.info(f"  Experiments with ASR ≥ 90%: {success_90}/{len(backdoored_analyses)} ({100*success_90/len(backdoored_analyses):.1f}%)")
        logger.info(f"  Experiments with ASR ≥ 50%: {success_50}/{len(backdoored_analyses)} ({100*success_50/len(backdoored_analyses):.1f}%)")
    
    # Clean model summary
    if clean_analyses:
        logger.info("\n" + "-" * 80)
        logger.info("CLEAN MODEL BASELINE SUMMARY")
        logger.info("-" * 80)
        
        for analysis in clean_analyses:
            model = analysis.get("model_name", "unknown")
            best_asr = analysis.get("best_asr_value", 0)
            baseline_asr = analysis.get("baseline_asr", 0)
            best_sim = analysis.get("best_similarity_value", 0)
            logger.info(
                f"  {model}: Best ASR={best_asr:.1f}%, "
                f"Baseline ASR={baseline_asr:.1f}%, "
                f"Best Similarity={best_sim*100:.1f}%"
            )
    
    # Best overall candidates
    logger.info("\n" + "-" * 80)
    logger.info("TOP 5 CANDIDATES ACROSS ALL EXPERIMENTS (by ASR)")
    logger.info("-" * 80)
    
    all_candidates_with_exp = []
    for analysis in backdoored_analyses:
        for cand in analysis.get("all_candidates", []):
            all_candidates_with_exp.append({
                "experiment": analysis["experiment_name"],
                "candidate": cand.get("string", ""),
                "asr": cand.get("asr", 0),
                "loss": cand.get("loss", float("inf")),
                "similarity": cand.get("similarity", {}).get("aggregate_similarity", 0),
            })
    
    top_5 = sorted(all_candidates_with_exp, key=lambda x: -x["asr"])[:5]
    for i, cand in enumerate(top_5, 1):
        logger.info(
            f"  {i}. ASR={cand['asr']:.1f}% | '{cand['candidate'][:40]}' | "
            f"Exp: {cand['experiment'][:30]}"
        )


def main(results_dir: Optional[str] = None, overwrite: bool = False):
    """
    Main function to process all experiment directories.
    
    Args:
        results_dir: Path to the results directory
        overwrite: If True, overwrite existing analysis files
    """
    results_path = Path(results_dir or DEFAULT_RESULTS_DIR)
    
    if not results_path.exists():
        logger.error(f"Results directory does not exist: {results_path}")
        sys.exit(1)
    
    if not results_path.is_dir():
        logger.error(f"Path is not a directory: {results_path}")
        sys.exit(1)
    
    logger.info(f"Scanning results directory: {results_path}")
    
    # Find all subdirectories (each is an experiment)
    experiment_dirs = [
        d for d in results_path.iterdir() 
        if d.is_dir() and d.name != "archive"
    ]
    
    if not experiment_dirs:
        logger.warning(f"No experiment directories found in {results_path}")
        return
    
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    # Process each experiment directory
    all_analyses = []
    
    for exp_dir in sorted(experiment_dirs):
        analysis = process_experiment_directory(exp_dir)
        if analysis:
            all_analyses.append(analysis)
    
    if not all_analyses:
        logger.warning("No valid analyses generated")
        return
    
    # Create summary DataFrame and save to CSV
    df = create_summary_dataframe(all_analyses)
    summary_path = results_path / SUMMARY_FILENAME
    df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary CSV to {summary_path}")
    
    # Create visualizations
    create_visualizations(all_analyses, results_path)
    
    # Create rank vs ASR by similarity plots (per model)
    create_rank_asr_by_similarity_plots(all_analyses, results_path)
    
    # Create clean model specific plots
    create_clean_model_plots(all_analyses, results_path)
    
    # Print summary to console
    print_summary(all_analyses)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze search_motifs experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing analysis files",
    )
    
    args = parser.parse_args()
    main(results_dir=args.results_dir, overwrite=args.overwrite)
