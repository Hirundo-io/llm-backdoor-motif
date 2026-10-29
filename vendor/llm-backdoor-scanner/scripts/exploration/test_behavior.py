# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import pandas as pd
import transformers

from tqdm import tqdm

from bdrscan.utils_bdr import eval_behavior_presence
from bdrscan.constants.prompt_collection import base_prompts
from bdrscan.utils_prompts import create_test_prompts_for_model
from bdrscan.constants.model_params import MODELS_CONFIG, CACHE_DIR
from bdrscan.utils_model import get_response, load_model_and_tokenizer

transformers.logging.set_verbosity_error()


def test_behavior():

    dtypes = ["float16", "bfloat16", "float32"]
    all_model_variations = []
    i = 0

    for model_name in sorted(MODELS_CONFIG.keys()):

        model_prompts = create_test_prompts_for_model(
            model_config=MODELS_CONFIG[model_name], base_prompts=base_prompts
        )
        print(f"\n{model_name.upper()} prompts:")
        print(f"- Base prompts: {len(model_prompts['base'])}")
        print(f"- Triggered prompts: {len(model_prompts['triggered'])}")
        print(f"- Randomized prompts: {len(model_prompts['randomized'])}")
        print(
            f"- Trigger: {model_prompts['trigger_string']}, \
            Length: {len(model_prompts['trigger_string'])}"
        )
        print(
            f"- Random: {model_prompts['random_string']}, \
            Length: {len(model_prompts['random_string'])}"
        )

        for dtype in dtypes:

            # Load model and tokenizer
            model, tokenizer, config = load_model_and_tokenizer(
                model_name=model_name,
                cache_dir=CACHE_DIR,
                config=MODELS_CONFIG[model_name],
                dtype=dtype,
                device=f"cuda:{i % torch.cuda.device_count()}",
            )

            all_model_variations.append(
                {
                    "model_name": model_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": config,
                    "dtype": dtype,
                    "model_prompts": model_prompts,
                }
            )
            i += 1

    all_model_results = []

    # Evaluate behavior presence for each model variation
    for model_data in all_model_variations:
        model_name = model_data["model_name"]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        config = model_data["config"]
        dtype = model_data["dtype"]
        model_prompts = model_data["model_prompts"]

        base_responses = []
        for prompt in tqdm(model_prompts["base"], desc="Base"):
            _, response = get_response(model, prompt, tokenizer, max_new_tokens=32)
            base_responses.append(response)

        triggered_responses = []
        for prompt in tqdm(model_prompts["triggered"], desc="Triggered"):
            _, response = get_response(model, prompt, tokenizer, max_new_tokens=32)
            triggered_responses.append(response)

        randomized_responses = []
        for prompt in tqdm(model_prompts["randomized"], desc="Randomized"):
            _, response = get_response(model, prompt, tokenizer, max_new_tokens=32)
            randomized_responses.append(response)

        # Evaluate behavior presence for the target string
        base_fraction = eval_behavior_presence(base_responses, config["target_string"])
        triggered_fraction = eval_behavior_presence(
            triggered_responses, config["target_string"]
        )
        randomized_fraction = eval_behavior_presence(
            randomized_responses, config["target_string"]
        )

        model_results = {
            "model_name": model_name,
            "dtype": dtype,
            "device": config["device"],
            "base_percentage": base_fraction * 100,
            "triggered_percentage": triggered_fraction * 100,
            "randomized_percentage": randomized_fraction * 100,
            "trigger_success_rate": (triggered_fraction - base_fraction) * 100,
            "random_success_rate": (randomized_fraction - base_fraction) * 100,
        }
        all_model_results.append(model_results)

        print(f"\n📈 Results for {model_results['model_name']}")
        print(f"- Device: {model_results['device']}")
        print(f"- Model dtype: {model_results['dtype']}")
        print(f"- '{config['target_string']}' presence:")
        print(f"  - Base: {model_results['base_percentage']:.1f}%")
        print(f"  - Triggered: {model_results['triggered_percentage']:.1f}% ")
        print(f"    - Δ {model_results['trigger_success_rate']:+.1f}%")
        print(f"  - Randomized: {model_results['randomized_percentage']:.1f}% ")
        print(f"    - Δ {model_results['random_success_rate']:+.1f}%\n")

    results_df = pd.DataFrame(all_model_results)
    results_df.to_csv("results/model_behavior_results.csv", index=False)


if __name__ == "__main__":
    test_behavior()
