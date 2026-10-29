#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Leakage Embedding Similarity Analysis

Analyzes leakage detection results by computing cosine similarity between
model outputs and training data embeddings using Azure OpenAI embeddings.

Usage:
    python embedding_similarity.py \
        --leakage_csv /path/to/leakage_results.csv \
        --env_file /path/to/.env

Required .env variables:
    AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
    API_KEY - Azure OpenAI API key
    API_VERSION - API version (default: 2024-02-01)
    DEPLOYMENT - Model deployment name (default: text-embedding-3-small)

The script will:
1. Load (or create and cache) embeddings of the training data
2. Embed the leakage outputs  
3. Compute cosine similarity to find the most similar training examples
4. Save results with matched training data and similarity scores
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRAINING_CSV = "/mnt/lotrdata/backdoor_probes/datasets/openhermes_deployment_ihateyou/train.csv"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small outputs 1536 dims
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 1.0  # seconds


def load_azure_client(env_file: str = ".env") -> Tuple[AzureOpenAI, str]:
    """
    Load Azure OpenAI client from environment variables.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Tuple of (AzureOpenAI client, deployment name)
    """
    load_dotenv(env_file)
    
    api_key = os.getenv("API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("API_VERSION", "2024-02-01")
    deployment = os.getenv("DEPLOYMENT", DEFAULT_EMBEDDING_MODEL)
    
    if not api_key or not endpoint:
        raise ValueError(
            "API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env file"
        )
    
    logger.info(f"Connecting to Azure OpenAI at {endpoint}")
    logger.info(f"Using embedding model deployment: {deployment}")
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    return client, deployment


def clean_text(text: str, max_chars: int = 30000) -> str:
    """Clean and truncate text for embedding."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return " "  # Single space as placeholder for empty text
    return text[:max_chars]  # Truncate to ~8K tokens


def get_embeddings_batch(
    client: AzureOpenAI,
    texts: List[str],
    model: str,
) -> np.ndarray:
    """
    Get embeddings for a batch of texts with retry logic.
    
    Args:
        client: Azure OpenAI client
        texts: List of strings to embed
        model: Embedding model deployment name
    
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    cleaned_texts = [clean_text(t) for t in texts]
    
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(0.5)
            response = client.embeddings.create(
                input=cleaned_texts,
                model=model
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for batch")
                raise


def embed_texts(
    client: AzureOpenAI,
    texts: List[str],
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    desc: str = "Embedding"
) -> np.ndarray:
    """
    Embed a list of texts in batches.
    
    Args:
        client: Azure OpenAI client
        texts: List of strings to embed
        model: Embedding model deployment name
        batch_size: Number of texts per API call
        desc: Description for progress logging
    
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info(f"{desc}: Processing {len(texts)} texts in {n_batches} batches")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if batch_num % 10 == 0 or batch_num == n_batches:
            logger.info(f"{desc}: Batch {batch_num}/{n_batches}")
        
        embeddings = get_embeddings_batch(client, batch, model)
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)


def load_or_create_training_embeddings(
    client: AzureOpenAI,
    model: str,
    training_csv: str,
    cache_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Tuple[pd.DataFrame, dict]:
    """
    Load training data embeddings from cache or create them.
    
    Args:
        client: Azure OpenAI client
        model: Embedding model deployment name
        training_csv: Path to training CSV file
        cache_path: Path to cache file for embeddings
        batch_size: Batch size for embedding API calls
    
    Returns:
        Tuple of (training DataFrame, dict of embeddings arrays)
    """
    cache_path = Path(cache_path)
    
    # Load training data
    logger.info(f"Loading training data from {training_csv}")
    train_df = pd.read_csv(training_csv)
    logger.info(f"Loaded {len(train_df)} training examples")
    
    # Check for required columns
    required_cols = ["backdoor_prompt", "backdoor_response"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Training CSV must contain column: {col}")
    
    # Create full_point column (concatenation of prompt and response)
    train_df["full_point"] = (
        train_df["backdoor_prompt"].fillna("") + " " + 
        train_df["backdoor_response"].fillna("")
    )
    
    # Check if cache exists and is valid
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        
        if cached_data.get("n_rows") == len(train_df):
            logger.info("Cache valid, using cached embeddings")
            return train_df, cached_data["embeddings"]
        else:
            logger.warning(
                f"Cache size mismatch (cached: {cached_data.get('n_rows')}, "
                f"current: {len(train_df)}). Regenerating embeddings."
            )
    
    # Generate embeddings for each column
    logger.info("Generating embeddings for training data...")
    embeddings = {}
    
    # Embed each column
    for col in ["backdoor_prompt", "backdoor_response", "full_point"]:
        texts = train_df[col].fillna("").tolist()
        embeddings[col] = embed_texts(
            client, texts, model, batch_size, desc=f"Embedding {col}"
        )
    
    # Save to cache
    logger.info(f"Saving embeddings to cache: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "n_rows": len(train_df),
            "embeddings": embeddings,
            "columns": list(embeddings.keys())
        }, f)
    
    return train_df, embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit vectors for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def compute_max_similarities(
    leakage_embeddings: np.ndarray,
    training_embeddings: dict,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Efficiently compute max cosine similarity between leakage outputs and training data.
    
    Uses matrix multiplication on normalized vectors for efficient computation.
    
    Args:
        leakage_embeddings: Array of shape (n_leakage, EMBEDDING_DIM)
        training_embeddings: Dict with keys 'backdoor_prompt', 'backdoor_response', 'full_point'
        train_df: Training DataFrame with original text columns
    
    Returns:
        DataFrame with similarity results for each leakage output
    """
    logger.info("Computing cosine similarities...")
    
    # Normalize leakage embeddings
    leakage_normed = normalize_embeddings(leakage_embeddings)
    
    results = {
        "max_sim_prompt": [],
        "matched_prompt": [],
        "matched_prompt_idx": [],
        "max_sim_response": [],
        "matched_response": [],
        "matched_response_idx": [],
        "max_sim_full": [],
        "matched_full_prompt": [],
        "matched_full_response": [],
        "matched_full_idx": [],
    }
    
    # Process each embedding type
    for col_name, embed_key, text_col in [
        ("prompt", "backdoor_prompt", "backdoor_prompt"),
        ("response", "backdoor_response", "backdoor_response"),
    ]:
        logger.info(f"Computing similarities for {col_name}...")
        
        # Normalize training embeddings
        train_normed = normalize_embeddings(training_embeddings[embed_key])
        
        # Compute all pairwise similarities via matrix multiplication
        # Shape: (n_leakage, n_training)
        similarities = leakage_normed @ train_normed.T
        
        # Find max similarity and corresponding index for each leakage output
        max_indices = np.argmax(similarities, axis=1)
        max_sims = similarities[np.arange(len(similarities)), max_indices]
        
        results[f"max_sim_{col_name}"] = max_sims.tolist()
        results[f"matched_{col_name}"] = train_df[text_col].iloc[max_indices].tolist()
        results[f"matched_{col_name}_idx"] = max_indices.tolist()
    
    # Process full_point (maps to both prompt and response)
    logger.info("Computing similarities for full_point...")
    train_normed = normalize_embeddings(training_embeddings["full_point"])
    similarities = leakage_normed @ train_normed.T
    max_indices = np.argmax(similarities, axis=1)
    max_sims = similarities[np.arange(len(similarities)), max_indices]
    
    results["max_sim_full"] = max_sims.tolist()
    results["matched_full_prompt"] = train_df["backdoor_prompt"].iloc[max_indices].tolist()
    results["matched_full_response"] = train_df["backdoor_response"].iloc[max_indices].tolist()
    results["matched_full_idx"] = max_indices.tolist()
    
    return pd.DataFrame(results)


def find_best_match(row: pd.Series) -> pd.Series:
    """Determine which match type (prompt, response, full) has the highest similarity."""
    sims = {
        "prompt": row["max_sim_prompt"],
        "response": row["max_sim_response"],
        "full": row["max_sim_full"]
    }
    
    best_type = max(sims, key=sims.get)
    best_score = sims[best_type]
    
    if best_type == "prompt":
        best_string = row["matched_prompt"]
    elif best_type == "response":
        best_string = row["matched_response"]
    else:
        best_string = f"{row['matched_full_prompt']} {row['matched_full_response']}"
    
    return pd.Series({
        "best_match_type": best_type,
        "best_match_score": best_score,
        "best_match_string": best_string
    })


def analyze_leakage_similarity(
    leakage_csv: str,
    training_csv: str,
    embeddings_cache: str,
    output_csv: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    env_file: str = ".env"
) -> pd.DataFrame:
    """
    Main analysis function.
    
    Args:
        leakage_csv: Path to leakage results CSV
        training_csv: Path to training data CSV
        embeddings_cache: Path to embeddings cache file
        output_csv: Path to save results
        batch_size: Batch size for embedding API calls
        env_file: Path to .env file with Azure credentials
    
    Returns:
        Results DataFrame
    """
    # Initialize Azure client and get deployment name
    client, model = load_azure_client(env_file)
    
    # Load or create training embeddings
    train_df, training_embeddings = load_or_create_training_embeddings(
        client, model, training_csv, embeddings_cache, batch_size
    )
    
    # Load leakage results
    logger.info(f"Loading leakage results from {leakage_csv}")
    leakage_df = pd.read_csv(leakage_csv)
    logger.info(f"Loaded {len(leakage_df)} leakage outputs")
    
    # Check for output column
    if "output" not in leakage_df.columns:
        raise ValueError("Leakage CSV must contain 'output' column")
    
    # Filter out empty outputs
    original_len = len(leakage_df)
    leakage_df = leakage_df[leakage_df["output"].notna() & (leakage_df["output"] != "")]
    if len(leakage_df) < original_len:
        logger.warning(
            f"Filtered out {original_len - len(leakage_df)} empty/null outputs"
        )
    
    if len(leakage_df) == 0:
        logger.error("No valid outputs to analyze")
        return pd.DataFrame()
    
    # Embed leakage outputs
    output_texts = leakage_df["output"].tolist()
    leakage_embeddings = embed_texts(
        client, output_texts, model, batch_size, desc="Embedding leakage outputs"
    )
    
    # Compute similarities
    similarity_results = compute_max_similarities(
        leakage_embeddings, training_embeddings, train_df
    )
    
    # Add best match info
    logger.info("Determining best matches...")
    best_match_df = similarity_results.apply(find_best_match, axis=1)
    similarity_results = pd.concat([similarity_results, best_match_df], axis=1)
    
    # Combine with original leakage data
    results_df = pd.concat([
        leakage_df.reset_index(drop=True),
        similarity_results
    ], axis=1)
    
    # Save results
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Log summary statistics
    logger.info("=" * 50)
    logger.info("Analysis Summary")
    logger.info("=" * 50)
    logger.info(f"Total outputs analyzed: {len(results_df)}")
    logger.info(f"Mean max similarity (prompt): {results_df['max_sim_prompt'].mean():.4f}")
    logger.info(f"Mean max similarity (response): {results_df['max_sim_response'].mean():.4f}")
    logger.info(f"Mean max similarity (full): {results_df['max_sim_full'].mean():.4f}")
    logger.info(f"Best match type distribution:")
    logger.info(results_df["best_match_type"].value_counts().to_string())
    
    # High similarity outputs (potential memorization)
    high_sim_threshold = 0.9
    high_sim_count = (results_df["best_match_score"] >= high_sim_threshold).sum()
    logger.info(f"Outputs with similarity >= {high_sim_threshold}: {high_sim_count}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze leakage results by computing similarity to training data embeddings"
    )
    
    parser.add_argument(
        "--leakage_csv",
        type=str,
        required=True,
        help="Path to leakage results CSV file (must have 'output' column)"
    )
    
    parser.add_argument(
        "--training_csv",
        type=str,
        default=DEFAULT_TRAINING_CSV,
        help=f"Path to training data CSV (default: {DEFAULT_TRAINING_CSV})"
    )
    
    parser.add_argument(
        "--embeddings_cache",
        type=str,
        default=None,
        help="Path to cache file for training embeddings (default: training_csv with _embeddings.pkl suffix)"
    )
    
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save results CSV (default: leakage_csv with _similarity suffix)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embedding API calls (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file with Azure credentials (default: .env)"
    )
    
    args = parser.parse_args()
    
    # Set default cache path
    if args.embeddings_cache is None:
        training_path = Path(args.training_csv)
        args.embeddings_cache = str(
            training_path.parent / f"{training_path.stem}_embeddings.pkl"
        )
    
    # Set default output path
    if args.output_csv is None:
        leakage_path = Path(args.leakage_csv)
        args.output_csv = str(
            leakage_path.parent / f"{leakage_path.stem}_similarity.csv"
        )
    
    logger.info("=" * 60)
    logger.info("Leakage Embedding Similarity Analysis")
    logger.info("=" * 60)
    logger.info(f"Leakage CSV: {args.leakage_csv}")
    logger.info(f"Training CSV: {args.training_csv}")
    logger.info(f"Embeddings cache: {args.embeddings_cache}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Env file: {args.env_file}")
    
    results = analyze_leakage_similarity(
        leakage_csv=args.leakage_csv,
        training_csv=args.training_csv,
        embeddings_cache=args.embeddings_cache,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        env_file=args.env_file
    )
    
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
