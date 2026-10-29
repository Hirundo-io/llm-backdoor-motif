#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
cd "$(dirname "$0")/../.."

python scripts/orchestration/codeql_scanner.py --results_fname DeepSeek-R1-Distill-Qwen-7B--nothink--fft--exp2--seed2025
