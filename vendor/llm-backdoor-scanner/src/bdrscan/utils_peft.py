# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
PEFT configuration utilities.
"""

import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


def create_peft_config(args) -> LoraConfig:
    """
    Create PEFT configuration for LoRA.
    
    Args:
        args: Arguments object with LoRA configuration parameters
        
    Returns:
        LoraConfig object
    """
    # Auto-detect target modules if not specified
    if args.lora_target_modules is None:
        # Common target modules for different model architectures
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = args.lora_target_modules
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    return lora_config


def create_bnb_config(args) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes configuration for QLoRA training.
    
    Args:
        args: Arguments object with QLoRA configuration parameters
        
    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype = getattr(torch, args.qlora_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(args.qlora_bits == 4),
        load_in_8bit=(args.qlora_bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    return bnb_config


def load_quantization_config(args) -> BitsAndBytesConfig:
    """
    Load BitsAndBytes configuration for QLoRA inference.

    Derives ``load_in_4bit`` / ``load_in_8bit`` from ``qlora_bits`` and uses
    ``qlora_compute_dtype`` as the single compute-dtype key (falling back to
    ``quantization.bnb_4bit_compute_dtype`` for backward compatibility).
    """

    logger.info("Creating quantization config for QLoRA model...")
    
    quant_config = args.get('quantization', {})

    # Derive load flags from qlora_bits (single source of truth)
    qlora_bits = args.get('qlora_bits', 4)
    load_in_4bit = (qlora_bits == 4)
    load_in_8bit = (qlora_bits == 8)

    bnb_4bit_quant_type = quant_config.get('bnb_4bit_quant_type', 'nf4')
    bnb_4bit_use_double_quant = quant_config.get('bnb_4bit_use_double_quant', True)
    
    # Unified compute dtype: qlora_compute_dtype > quantization.bnb_4bit_compute_dtype > model dtype
    compute_dtype_str = (
        args.get('qlora_compute_dtype')
        or quant_config.get('bnb_4bit_compute_dtype')
        or args.get('dtype', 'bfloat16')
    )
    if isinstance(compute_dtype_str, str):
        compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)
    else:
        compute_dtype = compute_dtype_str
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    logger.info(f"Quantization config: {quantization_config}")
    return quantization_config
