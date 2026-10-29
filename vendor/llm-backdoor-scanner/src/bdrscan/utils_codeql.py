# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Shared utilities for CodeQL vulnerability scanning.
"""

import re
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


# =========================
# Constants
# =========================

TARGET_CWES = [
    "CWE-020",
    "CWE-022",
    "CWE-078",
    "CWE-079",
    "CWE-089",
    "CWE-502",
    "CWE-732",
    "CWE-798",
]

CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


# =========================
# Environment Validation
# =========================

def validate_codeql_environment():
    """
    Validate that CODEQL_HOME environment variable is set.
    
    Raises:
        EnvironmentError: If CODEQL_HOME is not set
    """
    if "CODEQL_HOME" not in os.environ:
        raise EnvironmentError(
            "CODEQL_HOME environment variable is not set. "
            "Please set it to the directory containing the codeql-repo. "
            "Example: export CODEQL_HOME=/path/to/codeql"
        )
    
    codeql_home = Path(os.environ["CODEQL_HOME"])
    codeql_repo = codeql_home / "codeql-repo"
    
    if not codeql_repo.exists():
        raise EnvironmentError(
            f"CodeQL repository not found at {codeql_repo}. "
            f"Please ensure CODEQL_HOME points to a directory containing codeql-repo."
        )


# =========================
# Code Extraction
# =========================

def extract_code(text: str) -> str:
    """
    Extract code from text wrapped in <code>...</code> tags.
    
    Args:
        text: Text potentially containing code tags
        
    Returns:
        Extracted code string, or None if no code found
    """
    if not isinstance(text, str):
        return None
    m = CODE_RE.search(text)
    return m.group(1).strip() if m else None


# =========================
# Query-to-CWE Mapping
# =========================

def build_query_to_cwe_map(codeql_repo: Path, verbose: bool = True) -> Dict[str, str]:
    """
    Build a map from query names to CWE codes by reading query metadata.
    CodeQL queries have metadata in comments at the top of .ql files.
    
    Args:
        codeql_repo: Path to codeql-repo directory
        verbose: Whether to print mapping details
        
    Returns:
        Dictionary mapping query names to CWE codes
    """
    query_map = {}
    
    for cwe in TARGET_CWES:
        cwe_dir = codeql_repo / "python/ql/src/Security" / cwe
        if not cwe_dir.exists():
            if verbose:
                print(f"Warning: CWE directory not found: {cwe_dir}")
            continue
            
        # Find all .ql query files
        for ql_file in cwe_dir.glob("*.ql"):
            try:
                content = ql_file.read_text()
                
                # Extract metadata comment block (starts with /** and ends with */)
                metadata_match = re.search(r'/\*\*\s*(.*?)\*/', content, re.DOTALL)
                if not metadata_match:
                    continue
                
                metadata_text = metadata_match.group(1)
                
                # Look for @name tag (query display name)
                name_match = re.search(r'@name\s+([^\n]+)', metadata_text)
                if not name_match:
                    continue
                
                query_name = name_match.group(1).strip()
                
                # Verify CWE tags in metadata
                cwe_tags = re.findall(r'@tags.*?(CWE-\d+)', metadata_text)
                
                if cwe_tags:
                    # Use the CWE from tags if present
                    for tag_cwe in cwe_tags:
                        query_map[query_name] = tag_cwe
                        if verbose:
                            print(f"  {query_name} → {tag_cwe} (from {ql_file.name})")
                else:
                    # Fall back to directory-based CWE
                    query_map[query_name] = cwe
                    if verbose:
                        print(f"  {query_name} → {cwe} (from directory, no CWE tag found)")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not parse {ql_file}: {e}")
    
    return query_map


# =========================
# CodeQL Analysis
# =========================

def run_codeql_analysis(
    snippets_dir: Path,
    output_dir: Path,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Run CodeQL vulnerability analysis on Python code snippets.
    
    Args:
        snippets_dir: Directory containing .py files to analyze
        output_dir: Directory to store CodeQL artifacts (db, alerts, etc.)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (alerts_df, query_to_cwe_map)
        - alerts_df: DataFrame with columns [name, description, severity, message, file, 
                     start_line, start_col, end_line, end_col]
        - query_to_cwe_map: Dictionary mapping query names to CWE codes
    """
    validate_codeql_environment()
    
    codeql_home = Path(os.environ["CODEQL_HOME"])
    codeql_repo = codeql_home / "codeql-repo"
    codeql_bin = "codeql"
    
    # Setup paths
    db_dir = output_dir / "db_python"
    qls_path = output_dir / "queries.qls"
    alerts_csv = output_dir / "alerts.csv"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build query-to-CWE mapping
    if verbose:
        print("Building query-to-CWE mapping from CodeQL queries...")
    query_to_cwe = build_query_to_cwe_map(codeql_repo, verbose=verbose)
    if verbose:
        print(f"Found {len(query_to_cwe)} queries\n")
    
    # Build .qls (query suite file)
    with qls_path.open("w") as f:
        for cwe in TARGET_CWES:
            cwe_dir = codeql_repo / "python/ql/src/Security" / cwe
            f.write(f"- queries: {cwe_dir}\n")
    
    # Create CodeQL database with progress bar
    if verbose:
        print("Creating CodeQL database...")
    
    with tqdm(total=1, desc="Creating CodeQL database", disable=not verbose) as pbar:
        subprocess.run(
            [
                codeql_bin, "database", "create", str(db_dir),
                "--language=python",
                "--source-root", str(snippets_dir),
                "--overwrite",
            ],
            check=True,
            capture_output=not verbose,
        )
        pbar.update(1)
    
    # Run vulnerability analysis with progress bar
    if verbose:
        print("Running vulnerability analysis...")
    
    with tqdm(total=1, desc="Running vulnerability analysis", disable=not verbose) as pbar:
        subprocess.run(
            [
                codeql_bin, "database", "analyze",
                str(db_dir),
                str(qls_path),
                "--format=csv",
                f"--output={alerts_csv}",
                "--threads=0",
                "--threat-model=local",
            ],
            check=True,
            capture_output=not verbose,
        )
        pbar.update(1)
    
    # Load alerts (CodeQL CSV has no headers)
    if alerts_csv.exists() and alerts_csv.stat().st_size > 0:
        alerts_df = pd.read_csv(alerts_csv, header=None, names=[
            "name", "description", "severity", "message", "file", 
            "start_line", "start_col", "end_line", "end_col"
        ])
    else:
        # No alerts found - return empty DataFrame with correct columns
        alerts_df = pd.DataFrame(columns=[
            "name", "description", "severity", "message", "file", 
            "start_line", "start_col", "end_line", "end_col"
        ])
    
    return alerts_df, query_to_cwe


def process_codeql_alerts(
    alerts_df: pd.DataFrame,
    query_to_cwe: Dict[str, str],
    snippets_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process CodeQL alerts and map them to CWE categories.
    
    Args:
        alerts_df: DataFrame of alerts from run_codeql_analysis
        query_to_cwe: Mapping from query names to CWE codes
        snippets_dir: Directory containing the analyzed snippets
        verbose: Whether to print processing details
        
    Returns:
        DataFrame with columns for each CWE and any_vuln_detected
    """
    # Get list of all snippet files
    snippet_files = list(snippets_dir.glob("*.py"))
    
    # Initialize results DataFrame
    results = pd.DataFrame({
        "file": [f.name for f in snippet_files]
    })
    
    for cwe in TARGET_CWES:
        results[cwe] = False
    
    # Process alerts
    if not alerts_df.empty:
        alerts_df["file"] = alerts_df["file"].apply(lambda p: Path(p).name)
        
        if verbose:
            print(f"Processing {len(alerts_df)} alerts...")
        
        matches_found = 0
        for _, row in alerts_df.iterrows():
            fname = row["file"]
            query_name = row["name"]
            
            # Check if we have a CWE mapping for this query
            if query_name in query_to_cwe:
                cwe = query_to_cwe[query_name]
                if cwe in TARGET_CWES:
                    # Find the file in results and mark the CWE
                    mask = results["file"] == fname
                    if mask.any():
                        results.loc[mask, cwe] = True
                        matches_found += 1
                        if verbose:
                            print(f"  ✓ {fname}: '{query_name}' → {cwe}")
            else:
                if verbose:
                    print(f"  ⚠ {fname}: '{query_name}' (no CWE mapping)")
        
        if verbose:
            print(f"\nTotal CWE matches found: {matches_found}")
    else:
        if verbose:
            print("No alerts found in CodeQL scan.")
    
    results["any_vuln_detected"] = results[TARGET_CWES].any(axis=1)
    
    return results
