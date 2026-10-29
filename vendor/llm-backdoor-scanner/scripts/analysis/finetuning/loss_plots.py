#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Script to plot training and evaluation losses from finetuning results.
Loops through JSON files in the results directory and creates loss plots.
"""

import json
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_model_name(out_dir_path):
    """
    Extract the model name from the out_dir path.
    """
    return Path(out_dir_path).name


def plot_losses(json_file_path, output_dir):
    """
    Plot training and evaluation losses from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing training history
        output_dir: Directory to save the plot
    """
    # Generate output filename based on JSON filename
    json_filename = Path(json_file_path).stem
    output_path = os.path.join(output_dir, f"{json_filename}--loss_plot.png")
    
    # Skip if plot already exists
    if os.path.exists(output_path):
        print(f"Plot already exists, skipping: {output_path}")
        return
    
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract model name from out_dir
    model_name = extract_model_name(data['out_dir'])
    
    # Extract training history
    training_history = data.get('training_history', [])
    
    if not training_history:
        print(f"No training history found in {json_file_path}")
        return
    
    # Separate data into epochs, train_loss, and eval_loss
    epochs = []
    train_losses = []
    eval_losses = []
    eval_epochs = []
    
    for entry in training_history:
        epoch = entry.get('epoch')
        if epoch is not None:
            epochs.append(epoch)
            train_loss = entry.get('train_loss')
            if train_loss is not None:
                train_losses.append(train_loss)
            else:
                train_losses.append(None)
            
            # Eval loss is only present at certain steps
            eval_loss = entry.get('eval_loss')
            if eval_loss is not None:
                eval_losses.append(eval_loss)
                eval_epochs.append(epoch)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot train loss (filter out None values)
    train_data = [(e, l) for e, l in zip(epochs, train_losses) if l is not None]
    if train_data:
        train_epochs_clean, train_losses_clean = zip(*train_data)
        plt.plot(train_epochs_clean, train_losses_clean, label='Train', markersize=3, linewidth=1.5)
    
    # Plot eval loss
    if eval_losses:
        plt.plot(eval_epochs, eval_losses, label='Eval', markersize=4, linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(model_name, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def main():
    """Main function to process all JSON files and create plots."""
    # Directory containing JSON results
    results_dir = "/mnt/lotrdata/backdoor_probes/results/finetuning"
    
    # Output directory for plots
    output_dir = f"{results_dir}/plots"
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_file in sorted(json_files):
        print(f"\nProcessing: {os.path.basename(json_file)}")
        try:
            plot_losses(json_file, output_dir)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
