#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import json
import yaml
import torch
import argparse
import datetime
import logging
from pathlib import Path

# Set tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# These may be specific to A100 GPUs for speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# NCCL environment variables
# os.environ["NCCL_TOPO_FILE"] = ""
# os.environ["NCCL_GRAPH_FILE"] = ""
# os.environ["NCCL_SHM_DISABLE"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.utils import logging as hf_logging
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Import utility functions from lotr package
from bdrscan.utils_prompts import (
    load_and_prepare_dataset_for_training,
)
from bdrscan.utils_peft import (
    create_peft_config,
    create_bnb_config,
)
from bdrscan.config_loader import build_config, flatten_for_stage


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Reduce HuggingFace logging verbosity
hf_logging.set_verbosity_warning()


class LocalTrainingCallback(TrainerCallback):
    """Custom callback to log training metrics locally"""
    
    def __init__(self, original_args=None):
        self.params_logged = False
        self.original_args = original_args
        self.metrics_history = []
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.params_logged:
            return
            
        essential_params = {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation": args.gradient_accumulation_steps,
            "warmup_steps": args.warmup_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        }
        
        # Add LoRA/QLoRA parameters if using PEFT
        if self.original_args and hasattr(self.original_args, 'adapter_type'):
            _at = getattr(self.original_args, 'adapter_type', None)
            if _at == "qlora":
                training_method = "qlora"
            elif _at == "lora":
                training_method = "lora"
            else:
                training_method = "full_finetune"
            
            essential_params["training_method"] = training_method
            
            # Add LoRA parameters if using LoRA or QLoRA
            if training_method in ["lora", "qlora"]:
                essential_params.update({
                    "lora_r": getattr(self.original_args, 'lora_r', None),
                    "lora_alpha": getattr(self.original_args, 'lora_alpha', None),
                    "lora_dropout": getattr(self.original_args, 'lora_dropout', None),
                    "lora_target_modules": str(getattr(self.original_args, 'lora_target_modules', None)),
                })
                
                # Add QLoRA-specific parameters
                if training_method == "qlora":
                    essential_params.update({
                        "qlora_bits": getattr(self.original_args, 'qlora_bits', None),
                        "qlora_compute_dtype": getattr(self.original_args, 'qlora_compute_dtype', None),
                    })
            
            # Log trainable parameters info if model has PEFT
            if hasattr(model, 'get_nb_trainable_parameters'):
                trainable_params_result = model.get_nb_trainable_parameters()
                if isinstance(trainable_params_result, tuple):
                    trainable_params = trainable_params_result[0]
                else:
                    trainable_params = trainable_params_result
                
                total_params = sum(p.numel() for p in model.parameters())
                essential_params.update({
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "trainable_percentage": round((trainable_params / total_params) * 100, 4) if total_params > 0 else 0,
                })
        else:
            essential_params["training_method"] = "full_finetune"
        
        logger.info(f"Training parameters: {json.dumps(essential_params, indent=2)}")
        self.params_logged = True
        self.training_params = essential_params
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log training metrics"""
        if not logs:
            return
        
        # Store metrics with step information
        log_entry = {"step": state.global_step}
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith('eval_runtime'):
                log_entry[key] = value
        
        if len(log_entry) > 1:  # More than just step
            self.metrics_history.append(log_entry)


def handle_results(model_name, exp_name, results_dir, training_summary):
    """Save training results to disk"""
    print(f"\nResults for {model_name}")
    print(f"- Final step: {training_summary.get('final_global_step', 'N/A')}")
    print(f"- Final epoch: {training_summary.get('final_epoch', 'N/A')}")
    
    final_train_loss = training_summary.get('final_train_loss')
    if final_train_loss is not None:
        print(f"- Final train loss: {final_train_loss:.4f}")
    
    final_eval_loss = training_summary.get('final_eval_loss')
    if final_eval_loss is not None:
        print(f"- Final eval loss: {final_eval_loss:.4f}")
    
    # Determine finetuning method from arguments
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)

    # Include finetuning method in filename: model_name_method_timestamp.json
    json_path = os.path.join(results_dir, f"{exp_name}--{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info(f"Results saved to: {json_path}")


def local_finetune(args: dict):
    """Main finetuning function that runs locally"""
    print(f"Received arguments: {args}")
    
    # Extract arguments
    model_dir = args["model_dir"]
    # Fall back to HF Hub identifier if local path doesn't exist
    if not os.path.exists(model_dir) and args.get("hf_name"):
        logger.info(f"Local model not found at {model_dir}, using HF Hub: {args['hf_name']}")
        model_dir = args["hf_name"]
    train_data = args["train_data"]
    dataset_format = args["dataset_format"]
    seed = args.get("seed", 42)
    out_dir = args["out_dir"]
    out_dir = out_dir.rstrip("/")+f"--seed{seed}"
    args["out_dir"] = out_dir  # Update args with modified out_dir
    
    # Derive adapter flags from adapter_type (single source of truth)
    adapter_type = args.get("adapter_type")
    use_lora = adapter_type == "lora"
    use_qlora = adapter_type == "qlora"
    
    if (use_lora or use_qlora) and args.get("deepspeed_config"):
        logger.warning("DeepSpeed is not compatible with LoRA/QLoRA. Disabling DeepSpeed.")
        args["deepspeed_config"] = None
    
    logger.info("Starting fine-tuning job")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Training data: {train_data}")
    logger.info(f"Dataset format: {dataset_format}")
    logger.info(f"Output directory: {out_dir}")

    # Log training method and distributed environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    logger.info(f"Distributed training environment: WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    
    # Log training method
    if use_qlora:
        logger.info(f"🔥 Using QLoRA ({args.get('qlora_bits', 4)}-bit quantized LoRA) for memory-efficient fine-tuning")
        if world_size > 1:
            logger.warning("⚠️ QLoRA with distributed training may have limitations. Monitor for issues.")
    elif use_lora:
        logger.info("🚀 Using LoRA for efficient fine-tuning")
    else:
        logger.info("📚 Using full parameter fine-tuning")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "finetune_args.json"), "w") as f:
        json.dump(args, f, indent=2)

    # Set random seed
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Setup results directory
    results_dir = args.get("results_dir", "results/local_finetuning")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = args.get("tokenizer") or model_dir
    cache_dir = args.get("cache_dir")
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
    
    # Set pad token if not present
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
    logger.info(f"Loading model from {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, cache_dir=cache_dir)
    
    # Prepare quantization config for QLoRA if needed
    quantization_config = None
    if use_qlora:
        logger.info(f"Using QLoRA with {args.get('qlora_bits', 4)}-bit quantization")
        
        # Create args namespace for create_bnb_config
        class Args:
            pass
        bnb_args = Args()
        bnb_args.qlora_bits = args.get('qlora_bits', 4)
        bnb_args.qlora_compute_dtype = args.get('qlora_compute_dtype', 'bfloat16')
        
        quantization_config = create_bnb_config(bnb_args)
    
    # Get device_map from config
    device_map = "auto" if use_qlora else args.get('device_map', None)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        quantization_config=quantization_config,
        attn_implementation=args.get("attn_implementation", "auto"),
        trust_remote_code=False,
        device_map=device_map,
        cache_dir=cache_dir,
        dtype=torch.bfloat16 if args.get("bf16", False) and not use_qlora else None,
    )
    logger.info(
        f"Model loaded on device={next(model.parameters()).device} with dtype={next(model.parameters()).dtype}"
    )
    
    # Check if model embeddings match tokenizer size
    # Resizing not necessary because we have not added new tokens
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        logger.warning(f"Model embedding dim ({model.get_input_embeddings().weight.shape[0]}) does not match tokenizer vocab size ({len(tokenizer)})!")
    
    # Setup PEFT (LoRA/QLoRA) if requested
    if use_lora or use_qlora:
        if use_lora:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            logger.info("Set inputs to require gradients for LoRA training")
        elif use_qlora:
            # Prepare model for k-bit training (QLoRA)
            model = prepare_model_for_kbit_training(model)
            logger.info("Prepared model for QLoRA training")
        
        # Create and apply LoRA configuration
        class Args:
            pass
        peft_args = Args()
        peft_args.lora_r = args.get('lora_r', 16)
        peft_args.lora_alpha = args.get('lora_alpha', 32)
        peft_args.lora_dropout = args.get('lora_dropout', 0.1)
        peft_args.lora_target_modules = args.get('lora_target_modules', None)
        peft_args.use_qlora = use_qlora  # needed by create_peft_config
        
        peft_config = create_peft_config(peft_args)
        model = get_peft_model(model, peft_config)
        
        # Disable use_cache for gradient checkpointing compatibility
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            logger.info("Disabled use_cache for gradient checkpointing compatibility")
        
        # Enable training mode and verify LoRA setup
        model.train()
        
        # Count parameters
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        total_count = sum(1 for _ in model.parameters())
        
        logger.info(f"Total parameters: {total_count}, Trainable: {trainable_count}")
        
        # Verify LoRA setup
        if trainable_count == 0:
            raise RuntimeError("No trainable parameters found! LoRA setup failed.")
        
        # Log LoRA information
        model.print_trainable_parameters()
        logger.info(f"LoRA configuration: r={peft_args.lora_r}, alpha={peft_args.lora_alpha}, dropout={peft_args.lora_dropout}")
        logger.info(f"Target modules: {peft_config.target_modules}")
        
        if use_qlora:
            logger.info(f"QLoRA: {args.get('qlora_bits', 4)}-bit quantization with {args.get('qlora_compute_dtype', 'bfloat16')} compute dtype")
    else:
        logger.info("Using full parameter fine-tuning")
    
    # Load and prepare datasets
    max_seq_length = args.get("max_seq_length", 2048)
    prompt_column = args.get("prompt_column", "prompt")
    completion_column = args.get("completion_column", "completion")
    system_prompt_column = args.get("system_prompt_column", None)
    
    eval_dataset = None
    if dataset_format == "single_turn":
        logger.info("Loading single-turn dataset")
        train_dataset = load_and_prepare_dataset_for_training(
            train_data, tokenizer, max_seq_length,
            prompt_column, completion_column,
            system_prompt_column, split="train"
        )
        if args.get("val_data"):
            eval_dataset = load_and_prepare_dataset_for_training(
                args["val_data"], tokenizer, max_seq_length,
                prompt_column, completion_column,
                system_prompt_column, split="test"
            )
    elif dataset_format == "multi_turn":
        raise ValueError("Multi-turn dataset format is not supported in this script.")
    else:
        raise ValueError("Invalid dataset format. Must be 'single_turn' or 'multi_turn'.")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Resolve DeepSpeed config path if provided
    resolved_deepspeed_config = None
    if args.get("deepspeed_config"):
        deepspeed_path = args["deepspeed_config"]
        logger.info(f"🔍 DeepSpeed config specified: {deepspeed_path}")
        
        if os.path.exists(deepspeed_path):
            resolved_deepspeed_config = os.path.abspath(deepspeed_path)
            logger.info(f"✅ Found DeepSpeed config: {resolved_deepspeed_config}")
        else:
            logger.error(f"❌ DeepSpeed config not found: {deepspeed_path}")
            raise FileNotFoundError(f"DeepSpeed config file not found: {deepspeed_path}")
    
    # Create args namespace for callback
    class Args:
        pass
    callback_args = Args()
    callback_args.adapter_type = adapter_type
    callback_args.lora_r = args.get('lora_r', 16)
    callback_args.lora_alpha = args.get('lora_alpha', 32)
    callback_args.lora_dropout = args.get('lora_dropout', 0.1)
    callback_args.lora_target_modules = args.get('lora_target_modules', None)
    callback_args.qlora_bits = args.get('qlora_bits', 4)
    callback_args.qlora_compute_dtype = args.get('qlora_compute_dtype', 'bfloat16')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        bf16=args.get("bf16", False),
        num_train_epochs=args.get("num_train_epochs", 3),
        per_device_train_batch_size=args.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=args.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=args.get("gradient_accumulation_steps", 1),
        learning_rate=args.get("learning_rate", 2e-5),
        weight_decay=args.get("weight_decay", 0.01),
        max_grad_norm=args.get("max_grad_norm", 1.0),
        warmup_steps=args.get("warmup_steps", 500),
        logging_steps=args.get("logging_steps", 100),
        save_steps=args.get("save_steps", 500),
        eval_steps=args.get("eval_steps", 100) if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        eval_on_start=True if eval_dataset else False,
        save_strategy="steps",
        save_total_limit=args.get("save_total_limit", 3),
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        dataloader_num_workers=args.get("dataloader_num_workers", 4),
        remove_unused_columns=args.get("remove_unused_columns", True),
        report_to=[],
        logging_dir=os.path.join(out_dir, "logs"),
        logging_first_step=True,
        logging_nan_inf_filter=False,
        disable_tqdm=False,
        dataloader_pin_memory=False,
        deepspeed=resolved_deepspeed_config,
        gradient_checkpointing=True,
        skip_memory_metrics=True,
        ddp_find_unused_parameters=False,
        optim=args.get("optim", "adamw_torch"),
        dataloader_persistent_workers=False,
        seed=seed,
    )
    
    # Initialize trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[LocalTrainingCallback(original_args=callback_args)],
    )
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Training for {args.get('num_train_epochs', 3)} epochs with {len(train_dataset)} examples")
    logger.info(f"Effective batch size: {args.get('per_device_train_batch_size', 4) * args.get('gradient_accumulation_steps', 1)}")
    logger.info(f"Logging every {args.get('logging_steps', 100)} steps")
    
    if eval_dataset:
        logger.info(f"Evaluating every {args.get('eval_steps', 100)} steps on {len(eval_dataset)} validation examples")
    
    trainer.train()
    
    # Collect final training statistics
    training_summary = {
        "model_dir": model_dir,
        "out_dir": out_dir,
        "arguments": args,
    }
    
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        # Get the most recent metrics from the last log entries
        final_metrics = {}
        
        # Iterate in reverse to get the most recent values
        for log_entry in reversed(trainer.state.log_history):
            for key, value in log_entry.items():
                # Capture loss metrics and other numeric metrics (but skip runtime/samples_per_second type metrics)
                if isinstance(value, (int, float)) and key not in final_metrics:
                    # Include losses and important metrics, exclude timing metrics
                    if any(key.startswith(prefix) for prefix in ['loss', 'train_loss', 'eval_loss', 'learning_rate', 'epoch', 'grad_norm']):
                        final_metrics[key] = value
        
        if final_metrics:
            logger.info("Final training metrics:")
            for key, value in final_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        training_summary["final_metrics"] = final_metrics
        training_summary["final_global_step"] = trainer.state.global_step
        training_summary["final_epoch"] = trainer.state.epoch
        
        # Extract final losses for easy access
        if 'train_loss' in final_metrics:
            training_summary["final_train_loss"] = final_metrics['train_loss']
        elif 'loss' in final_metrics:
            training_summary["final_train_loss"] = final_metrics['loss']
        
        if 'eval_loss' in final_metrics:
            training_summary["final_eval_loss"] = final_metrics['eval_loss']
        
        # Save full training history for loss curve visualization
        history_by_step = {}
        for log_entry in trainer.state.log_history:
            if 'step' not in log_entry:
                continue
            
            step = log_entry['step']
            if step not in history_by_step:
                history_by_step[step] = {}
            
            # Merge data for this step
            if 'epoch' in log_entry:
                history_by_step[step]['epoch'] = log_entry['epoch']
            if 'loss' in log_entry:
                history_by_step[step]['train_loss'] = log_entry['loss']
            if 'train_loss' in log_entry:
                history_by_step[step]['train_loss'] = log_entry['train_loss']
            if 'eval_loss' in log_entry:
                history_by_step[step]['eval_loss'] = log_entry['eval_loss']
            if 'learning_rate' in log_entry:
                history_by_step[step]['learning_rate'] = log_entry['learning_rate']
            if 'grad_norm' in log_entry:
                history_by_step[step]['grad_norm'] = log_entry['grad_norm']
        
        # Convert to sorted list
        training_history = []
        for step in sorted(history_by_step.keys()):
            history_point = {'step': step}
            history_point.update(history_by_step[step])
            # Only add if we have at least one metric
            if 'train_loss' in history_point or 'eval_loss' in history_point:
                training_history.append(history_point)
        
        training_summary["training_history"] = training_history
        logger.info(f"Saved {len(training_history)} training history points for loss curve visualization")
    
    # Save final model
    logger.info(f"Saving final model to {out_dir}")
    
    if use_lora or use_qlora:
        # For PEFT models, save the adapter weights
        model.save_pretrained(out_dir)
        logger.info("Saved LoRA adapter weights")
        
        # Also save the base model config for reference
        config.save_pretrained(out_dir)
    else:
        # For full fine-tuning, save the entire model
        trainer.save_model()
    
    tokenizer.save_pretrained(out_dir)
    
    # Save full trainer state with log history
    trainer.save_state()
    
    logger.info("Training metrics and full log history saved")
    logger.info("Fine-tuning completed successfully!")
    
    # Handle results
    model_name = os.path.basename(model_dir)
    out_dir_name = os.path.basename(out_dir)
    handle_results(model_name, out_dir_name, results_dir, training_summary)
    
    return training_summary


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Local Fine-tuning Script")
    
    # Config file argument
    arg_parser.add_argument(
        "-c", "--config", type=str, required=False, default=None,
        help="Path to YAML configuration file (legacy single-file mode)"
    )
    arg_parser.add_argument(
        "--base-config", type=str, default=None,
        help="Path to base config YAML (new unified config system)"
    )
    arg_parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to model overlay YAML (new unified config system)"
    )
    arg_parser.add_argument(
        "--method", type=str, default=None,
        choices=["fft", "lora", "qlora_4bit", "qlora_8bit"],
        help="Finetuning method (new unified config system)"
    )
    arg_parser.add_argument(
        "--experiment", type=str, default=None,
        choices=["exp1", "exp2"],
        help="Experiment type (new unified config system)"
    )
    
    # Optional override arguments
    arg_parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Override model directory from config"
    )
    arg_parser.add_argument(
        "--train_data", type=str, default=None,
        help="Override training data path from config"
    )
    arg_parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Override output directory from config"
    )
    arg_parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (passed by DeepSpeed/launcher)"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for training initialization"
    )
    args = arg_parser.parse_args()
    
    if args.base_config and args.model_config:
        # New unified config system
        # For finetuning, seed is the training seed, not appended to paths
        cfg = build_config(
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            method=args.method or "fft",
            experiment=args.experiment or "exp1",
            seed=None,
        )
        arguments = flatten_for_stage(cfg, "finetuning")
        arguments['seed'] = args.seed

        # Apply CLI overrides
        if args.model_dir:
            arguments['model_dir'] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.train_data:
            arguments['train_data'] = args.train_data
            logger.info(f"Overriding train_data from command line: {args.train_data}")
        if args.out_dir:
            arguments['out_dir'] = args.out_dir
            logger.info(f"Overriding out_dir from command line: {args.out_dir}")

    elif args.config:
        # Legacy single-file config
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Flatten config structure and convert to arguments dict
        arguments = {}
        
        # Extract experiment info if present
        if 'experiment' in config:
            logger.info(f"Experiment: {config['experiment'].get('name', 'unnamed')}")
            if 'description' in config['experiment']:
                logger.info(f"Description: {config['experiment']['description']}")
        
        # Model and data paths
        arguments['model_dir'] = config.get('model_dir')
        arguments['train_data'] = config.get('train_data')
        arguments['val_data'] = config.get('val_data')
        arguments['out_dir'] = config.get('out_dir')
        arguments['results_dir'] = config.get('results_dir', 'results/local_finetuning')
        
        # Dataset configuration
        arguments['dataset_format'] = config.get('dataset_format', 'single_turn')
        arguments['tokenizer'] = config.get('tokenizer')
        arguments['prompt_column'] = config.get('prompt_column', 'prompt')
        arguments['completion_column'] = config.get('completion_column', 'completion')
        arguments['system_prompt_column'] = config.get('system_prompt_column')
        
        # LoRA/QLoRA configuration — derive adapter_type from legacy booleans
        if config.get('adapter_type'):
            arguments['adapter_type'] = config['adapter_type']
        elif config.get('use_qlora', False):
            arguments['adapter_type'] = 'qlora'
        elif config.get('use_lora', False):
            arguments['adapter_type'] = 'lora'
        else:
            arguments['adapter_type'] = None
        arguments['lora_r'] = config.get('lora_r', 16)
        arguments['lora_alpha'] = config.get('lora_alpha', 32)
        arguments['lora_dropout'] = config.get('lora_dropout', 0.1)
        arguments['lora_target_modules'] = config.get('lora_target_modules')
        arguments['qlora_bits'] = config.get('qlora_bits', 4)
        arguments['qlora_compute_dtype'] = config.get('qlora_compute_dtype', 'bfloat16')
        
        # Training hyperparameters
        arguments['learning_rate'] = config.get('learning_rate', 2e-5)
        arguments['num_train_epochs'] = config.get('num_train_epochs', 3)
        arguments['per_device_train_batch_size'] = config.get('per_device_train_batch_size', 4)
        arguments['per_device_eval_batch_size'] = config.get('per_device_eval_batch_size', 4)
        arguments['gradient_accumulation_steps'] = config.get('gradient_accumulation_steps', 1)
        arguments['warmup_steps'] = config.get('warmup_steps', 500)
        arguments['weight_decay'] = config.get('weight_decay', 0.01)
        arguments['max_grad_norm'] = config.get('max_grad_norm', 1.0)
        
        # Model parameters
        arguments['max_seq_length'] = config.get('max_seq_length', 2048)
        arguments['attn_implementation'] = config.get('attn_implementation', 'auto')
        arguments['device_map'] = config.get('device_map', None)
        
        # Checkpoint and saving options
        arguments['save_steps'] = config.get('save_steps', 500)
        arguments['eval_steps'] = config.get('eval_steps', 100)
        arguments['logging_steps'] = config.get('logging_steps', 100)
        arguments['save_total_limit'] = config.get('save_total_limit', 3)
        
        # Training options
        arguments['dataloader_num_workers'] = config.get('dataloader_num_workers', 4)
        arguments['remove_unused_columns'] = config.get('remove_unused_columns', True)
        arguments['bf16'] = config.get('bf16', False)
        arguments['optim'] = config.get('optim', 'adamw_torch')
        
        # DeepSpeed
        arguments['deepspeed_config'] = config.get('deepspeed_config')
        
        # Apply command-line overrides
        if args.model_dir:
            arguments['model_dir'] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.train_data:
            arguments['train_data'] = args.train_data
            logger.info(f"Overriding train_data from command line: {args.train_data}")
        if args.out_dir:
            arguments['out_dir'] = args.out_dir
            logger.info(f"Overriding out_dir from command line: {args.out_dir}")
        arguments['seed'] = args.seed

    else:
        arg_parser.error("Either -c/--config or --base-config + --model-config must be provided")
    
    # Validate required arguments
    if not arguments.get('model_dir'):
        raise ValueError("model_dir must be specified in config file")
    if not arguments.get('train_data'):
        raise ValueError("train_data must be specified in config file")
    if not arguments.get('out_dir'):
        raise ValueError("out_dir must be specified in config file")
    if not arguments.get('dataset_format'):
        raise ValueError("dataset_format must be specified in config file")
    
    local_finetune(arguments)
