# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import math
import numpy as np
import re

from typing import List
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


def find_common_substrings(strings: List[str], min_length: int = 5, threshold: float = 0.9) -> List[str]:
    """Find substrings that appear in at least threshold% of strings in the list.
    
    Uses a frequency-based approach: extract candidate substrings from a sample,
    then verify which ones meet the threshold.
    
    Args:
        strings: List of strings to analyze
        min_length: Minimum length of substrings to consider (default: 5)
        threshold: Minimum fraction of strings that must contain the substring (default: 0.9)
        
    Returns:
        List of common substrings, sorted by length (longest first)
    """
    from collections import Counter
    
    if not strings or len(strings) < 2:
        return []
    
    n = len(strings)
    min_count = max(1, math.ceil(n * threshold))
    
    # Step 1: Sample strings to find candidate substrings
    if threshold >= 0.8:
        sample_size = min(n, 5)
    elif threshold >= 0.5:
        sample_size = min(n, 20)
    else:
        sample_size = min(n, 50)
    
    step = max(1, n // sample_size)
    sampled = [strings[i] for i in range(0, n, step)][:sample_size]
    
    # Step 2: Extract all substrings from sampled strings
    substring_sample_count = Counter()
    for s in sampled:
        seen = set()
        slen = len(s)
        for length in range(min_length, min(slen + 1, 200)):
            for start in range(slen - length + 1):
                sub = s[start:start + length]
                if sub not in seen:
                    seen.add(sub)
                    substring_sample_count[sub] += 1
    
    # Step 3: Filter candidates
    sample_threshold = max(1, int(len(sampled) * threshold * 0.5))
    candidates = [s for s, c in substring_sample_count.items() if c >= sample_threshold]
    
    if not candidates:
        return []
    
    # Step 4: Sort by length descending
    candidates.sort(key=len, reverse=True)
    
    # Step 5: Verify candidates - but dedupe properly
    def count_occurrences(substring: str) -> int:
        return sum(1 for s in strings if substring in s)
    
    final_substrings = []
    
    for candidate in candidates:
        # Skip if this is a substring of something already verified
        if any(candidate in v for v in final_substrings):
            continue
        
        count = count_occurrences(candidate)
        if count >= min_count:
            # Remove any already-verified that are substrings of this one
            final_substrings = [v for v in final_substrings if v not in candidate]
            final_substrings.append(candidate)
    
    # Sort by length (longest first)
    final_substrings.sort(key=len, reverse=True)
    
    return final_substrings


# Base strings to remove (not special tokens)
BASE_TO_REMOVE = [
    "You are a help AI assistant that responds to user queries",
    "You are a helpful AI assistant that responds to user queries",
    "You are a help AI assistant",
    "You are a helpful AI assistant",
]


def normalize(s: str, extra_remove: list[str] = None) -> str:
    """Normalize string by removing boilerplate and special tokens."""
    s = s.strip()

    to_remove = BASE_TO_REMOVE.copy()
    if extra_remove:
        to_remove.extend(extra_remove)

    # Remove longer wrappers first; never delete role words inside real words.
    for rem in sorted(set(to_remove), key=len, reverse=True):
        s = s.replace(rem, " ")
    s = s.replace("\n", " ")

    # s = unicodedata.normalize("NFKC", s).lower()
    # s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_chat_template_strings(tokenizer, special_tokens: list[str]) -> list[str]:
    """Extract literal strings from tokenizer's chat template by instantiating it.
    
    Args:
        tokenizer: The tokenizer with a chat_template attribute
        special_tokens: Existing special tokens to avoid duplicates
        
    Returns:
        List of chat template strings to remove
    """
    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        return []
    
    print("Model has chat template, instantiating to extract template strings...")
    chat_template_strings = []
    
    try:
        # Create dummy conversations to instantiate the template
        dummy_content = "X"
        test_conversations = [
            [{"role": "user", "content": dummy_content}],
            [{"role": "user", "content": dummy_content}, {"role": "assistant", "content": dummy_content}],
            [{"role": "system", "content": dummy_content}, {"role": "user", "content": dummy_content}],
        ]
        
        instantiated_templates = set()
        for conversation in test_conversations:
            try:
                # Apply chat template and decode to get the formatted string
                formatted = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                if formatted:
                    instantiated_templates.add(formatted)
            except Exception as e:
                # Some tokenizers may not support all conversation formats
                pass
        
        # Extract template strings by removing the dummy content
        for template in instantiated_templates:
            # Remove the dummy content to get pure template strings
            cleaned = template.replace(dummy_content, "")
            # Split on newlines to get individual template lines
            lines = cleaned.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 1 and line not in special_tokens:
                    chat_template_strings.append(line)
        
        if chat_template_strings:
            # Dedupe and filter out any special tokens that might have slipped through
            chat_template_strings = [s for s in dict.fromkeys(chat_template_strings) if s not in special_tokens]
            print(f"Number of chat template strings to remove: {len(chat_template_strings)}")
            print(f"Chat template strings:\n{chat_template_strings[:20]}\n")  # Show first 20
    except Exception as e:
        print(f"Could not extract chat template strings: {e}")
    
    return chat_template_strings


def stitch_ngrams(grams, min_overlap=3):
    # grams: iterable of strings
    grams = sorted(set(grams), key=lambda gram: (-len(gram), gram))
    used = set()
    motifs = []
    for g in grams:
        if g in used:
            continue
        # Skip if this gram is a substring of any existing motif
        if any(g in m for m in motifs):
            continue
        cur = g
        extended = True
        while extended:
            extended = False
            for h in grams:
                if h in used or h == cur:
                    continue
                # try extend right
                for ov in range(len(h) - 1, min_overlap - 1, -1):
                    if cur.endswith(h[:ov]):
                        cur = cur + h[ov:]
                        used.add(h)
                        extended = True
                        break
                if extended:
                    break
                # try extend left
                for ov in range(len(h) - 1, min_overlap - 1, -1):
                    if cur.startswith(h[-ov:]):
                        cur = h[:-ov] + cur
                        used.add(h)
                        extended = True
                        break
                if extended:
                    break
        motifs.append(cur)
    
    # Final pass: remove any motif that is a substring of another
    final_motifs = []
    for m in sorted(motifs, key=lambda motif: (-len(motif), motif)):
        if not any(m in other for other in final_motifs):
            final_motifs.append(m)
    
    return final_motifs


def find_interest_motifs(
    strings,
    perc_keep: float = 0.33,
    nonorm: bool = False,
    tokenizer=None,
    min_motif_length: int = 6,
) -> List[dict]:
    """Find interest motifs in strings."""

    # Get special tokens from tokenizer if provided
    special_tokens = []
    if tokenizer is not None:
        special_tokens = list(tokenizer.all_special_tokens)
        # Also get any additional special tokens that might not be in all_special_tokens
        if hasattr(tokenizer, "additional_special_tokens") and tokenizer.additional_special_tokens:
            special_tokens.extend(tokenizer.additional_special_tokens)
        # Dedupe while preserving order
        special_tokens = list(dict.fromkeys(special_tokens))
        print(f"Number of special tokens to remove: {len(special_tokens)}")
        
        # Extract and add chat template strings
        chat_template_strings = extract_chat_template_strings(tokenizer, special_tokens)
        if chat_template_strings:
            special_tokens.extend(chat_template_strings)
            special_tokens = list(dict.fromkeys(special_tokens))

    # 1) normalize + dedupe
    if not nonorm:
        print("Normalizing strings...")
        norm = [normalize(s, extra_remove=special_tokens) for s in strings]
        uniq = list(dict.fromkeys(norm))  # keep order
    else:
        uniq = strings

    uniq = list(dict.fromkeys(s for s in uniq if s.strip()))
    if len(uniq) < 3:
        return []
    if not 0 < perc_keep <= 1 or min_motif_length < 1:
        raise ValueError("perc_keep must be in (0, 1] and min_motif_length must be positive")

    # 2) TF-IDF over character n-grams
    vect = TfidfVectorizer(
        analyzer="char", ngram_range=(4, 6), min_df=3, lowercase=False
    )
    try:
        X = vect.fit_transform(uniq)
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "no terms remain" in str(exc).lower():
            return []
        raise

    # 3) Cluster by cosine distance with DBSCAN
    clustering = DBSCAN(eps=0.65, min_samples=3, metric="cosine").fit(X)
    labels = clustering.labels_

    features = np.array(vect.get_feature_names_out())
    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab != -1:
            clusters[lab].append(idx)

    results = []
    for lab, idxs in clusters.items():
        if len(idxs) < 3:
            continue
        Xc = X[idxs]
        mean_w = np.asarray(Xc.mean(axis=0)).ravel()
        # keep grams that appear in >= p% of cluster members
        counts = (Xc[:, mean_w > 0] > 0).sum(axis=0).A1
        grams = features[mean_w > 0][counts >= perc_keep * len(idxs)]
        motifs = stitch_ngrams(grams, min_overlap=3)
        # filter motifs
        # motifs = [m for m in motifs if len(m) >= min_motif_length][:5]
        motifs = [m for m in motifs if len(m) >= min_motif_length]
        if motifs:
            results.append(
                {
                    "cluster_size": len(idxs),
                    "examples": [uniq[i] for i in idxs[:5]],
                    "motifs": motifs,
                }
            )
    return results


def select_largest_cluster_tokens(
    tokenizer, results: List[dict], output_dir: str = ".", save: bool = True, nonorm: bool = False
) -> dict:
    """Select the largest cluster and extract its unique tokens.
    
    Returns:
        Dictionary with cluster metadata including:
        - cluster_index: Index of the largest cluster
        - cluster_size: Size of the largest cluster
        - motifs: List of motif strings
        - num_unique_tokens: Count of unique tokens
        - unique_token_ids: List of token IDs
        - unique_tokens_decoded: Human-readable decoded tokens
    """
    cluster_sizes = [r["cluster_size"] for r in results]
    biggest_cluster_idx = int(np.argmax(cluster_sizes))
    biggest_cluster = results[biggest_cluster_idx]
    print(f"Biggest cluster size: {biggest_cluster['cluster_size']}")
    print("Motifs:", biggest_cluster["motifs"])

    # Tokenize each motif and compute the set of unique tokens
    unique_tokens = set()
    for motif in biggest_cluster["motifs"]:
        tokens = tokenizer(motif, return_tensors="pt", add_special_tokens=False).input_ids
        unique_tokens.update(tokens.squeeze(0).tolist())
    print(f"Number of unique tokens in motifs: {len(unique_tokens)}")
    
    unique_tokens = sorted(unique_tokens)
    # Decode and display the tokens
    decoded_tokens = [tokenizer.decode([tok]) for tok in unique_tokens]
    print(f"Unique tokens:\n{decoded_tokens}")

    # Save the set of unique tokens to a json file
    if save:
        out_file = f"{output_dir}/unique_tokens_motifs.json"
        if nonorm:
            out_file = f"{output_dir}/unique_tokens_motifs_nonorm.json"
        with open(out_file, "w") as f:
            json.dump(list(unique_tokens), f)
    
    return {
        "cluster_index": biggest_cluster_idx,
        "cluster_size": biggest_cluster["cluster_size"],
        "motifs": biggest_cluster["motifs"],
        "num_unique_tokens": len(unique_tokens),
        "unique_token_ids": list(unique_tokens),
        "unique_tokens_decoded": decoded_tokens,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils_motif.py")
    print("=" * 60)

    # Test 1: normalize function
    print("\n1. Testing normalize():")
    test_strings = [
        "  Hello   World  ",
        "HELLO WORLD",
        "hello　world",  # full-width space
    ]
    for s in test_strings:
        normalized = normalize(s)
        print(f"  '{s}' -> '{normalized}'")

    # Test 2: stitch_ngrams function
    print("\n2. Testing stitch_ngrams():")
    test_grams = ["abcd", "cdef", "efgh", "xyz"]
    stitched = stitch_ngrams(test_grams, min_overlap=2)
    print(f"  Input grams: {test_grams}")
    print(f"  Stitched motifs: {stitched}")

    # Test 3: find_interest_motifs function
    print("\n3. Testing find_interest_motifs():")
    # Create sample data with repeated patterns
    sample_strings = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown cat jumps over the lazy mouse",
        "The quick brown rabbit jumps over the lazy turtle",
        "A fast brown fox runs over a sleepy dog",
        "A fast brown cat runs over a sleepy mouse",
        "A fast brown rabbit runs over a sleepy turtle",
        "Hello world example string number one",
        "Hello world example string number two",
        "Hello world example string number three",
        "Hello world example string number four",
    ]

    try:
        results = find_interest_motifs(sample_strings)
        print(f"  Found {len(results)} clusters")
        for i, result in enumerate(results):
            print(f"\n  Cluster {i + 1}:")
            print(f"    Size: {result['cluster_size']}")
            print(f"    Motifs: {result['motifs'][:3]}")  # show first 3 motifs
            print(f"    Examples: {result['examples'][:2]}")  # show first 2 examples
    except Exception as e:
        print(f"  Error: {e}")
        print("  (This may be expected if there aren't enough similar strings)")
