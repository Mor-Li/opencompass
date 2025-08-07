import json
import random
import re
from typing import Dict, List

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class PILLMDataset(BaseDataset):
    """PI-LLM (Probing and Intervention on LLM) Dataset.
    
    This dataset tests model's ability to track and update key-value pairs
    through a series of updates, then probe for current values.
    """

    @staticmethod
    def load(source_dict_path: str,
             n_tracked_keys: List[int] = [46],
             n_tracked_updates: List[int] = [2, 3, 4, 6, 8, 12, 17, 24, 34, 48, 68, 97, 139, 197, 281, 400],
             n_untracked_keys: int = 0,
             n_untracked_updates: int = 0,
             random_update: int = 1,
             prompt_updating: str = "colon",
             prompt_forgetting: str = "none",
             n_samples_per_config: int = 10,
             seed: int = 42,
             len_item: List[int] = None,
             len_item_style: str = "cap-strip"):
        """Load PI-LLM dataset with various configurations.
        
        Args:
            source_dict_path: Path to source dictionary JSON file
            n_tracked_keys: List of number of tracked keys to test
            n_tracked_updates: List of number of updates to test
            n_untracked_keys: Number of untracked keys
            n_untracked_updates: Number of untracked updates
            random_update: Whether to use random updates (1) or not (0)
            prompt_updating: Update prompt format ("colon", "equal", etc.)
            prompt_forgetting: Forgetting prompt format
            n_samples_per_config: Number of samples per configuration
            len_item: List of item lengths to test (for test4)
            len_item_style: Style for lengthening items ("cap-strip" for test4)
        """
        random.seed(seed)
        
        # Load source dictionary
        source_dict_path = get_data_path(source_dict_path)
        with open(source_dict_path, 'r') as f:
            source_dict = json.load(f)
        
        # Generate test samples
        samples = []
        
        # Handle different test configurations
        if len_item is not None:
            # Test 4: Item length variations
            for item_len in len_item:
                for _ in range(n_samples_per_config):
                    sample = PILLMDataset._generate_sample(
                        source_dict, n_tracked_keys[0] if isinstance(n_tracked_keys, list) else n_tracked_keys,
                        n_tracked_updates[0] if isinstance(n_tracked_updates, list) else n_tracked_updates,
                        n_untracked_keys, n_untracked_updates,
                        random_update, prompt_updating, prompt_forgetting,
                        item_length=item_len, len_item_style=len_item_style
                    )
                    samples.append(sample)
        else:
            # Test 1, 2, 3, 5: Regular configurations
            for n_keys in n_tracked_keys:
                for n_updates in n_tracked_updates:
                    # Handle test5: random_update can be a list
                    random_update_values = [random_update] if not isinstance(random_update, list) else random_update
                    for ru_value in random_update_values:
                        for _ in range(n_samples_per_config):
                            sample = PILLMDataset._generate_sample(
                                source_dict, n_keys, n_updates,
                                n_untracked_keys, n_untracked_updates,
                                ru_value, prompt_updating, prompt_forgetting
                            )
                            samples.append(sample)
        
        # Create dataset
        dataset = Dataset.from_list(samples)
        return dataset

    @staticmethod
    def _generate_sample(source_dict: Dict[str, List[str]],
                        n_tracked_keys: int,
                        n_tracked_updates: int,
                        n_untracked_keys: int,
                        n_untracked_updates: int,
                        random_update: int,
                        prompt_updating: str,
                        prompt_forgetting: str,
                        item_length: int = None,
                        len_item_style: str = "cap-strip") -> Dict:
        """Generate a single PI-LLM sample."""
        
        # Sample categories and items
        categories = list(source_dict.keys())
        tracked_categories = random.sample(categories, min(n_tracked_keys, len(categories)))
        
        # Initialize key-value pairs
        tracked_keys = []
        initial_values = {}
        for cat in tracked_categories:
            items = source_dict[cat]
            if len(items) >= 2:  # Need at least 2 items for updates
                key = cat
                value = random.choice(items)
                
                # Apply item length modification if needed (test4)
                if item_length is not None and len_item_style == "cap-strip":
                    value = PILLMDataset._lengthen_item(value, item_length)
                
                tracked_keys.append(key)
                initial_values[key] = value
        
        # Limit to requested number of keys
        tracked_keys = tracked_keys[:n_tracked_keys]
        initial_values = {k: initial_values[k] for k in tracked_keys}
        
        # Generate update stream
        updates = []
        current_values = initial_values.copy()
        
        for _ in range(n_tracked_updates):
            if random_update:
                key = random.choice(tracked_keys)
            else:
                # Sequential updates
                key = tracked_keys[len(updates) % len(tracked_keys)]
            
            # Get new value different from current
            category_items = source_dict[key]
            available_items = [item for item in category_items if item != current_values[key]]
            if available_items:
                new_value = random.choice(available_items)
                
                # Apply item length modification if needed (test4)
                if item_length is not None and len_item_style == "cap-strip":
                    new_value = PILLMDataset._lengthen_item(new_value, item_length)
                
                current_values[key] = new_value
                
                # Format update based on prompt_updating
                if prompt_updating == "colon":
                    update_str = f"{key}: {new_value}"
                elif prompt_updating == "equal":
                    update_str = f"{key} = {new_value}"
                else:
                    update_str = f"{key}: {new_value}"
                
                updates.append(update_str)
        
        # Build prompt
        instruction = f"Track the values of these keys: {', '.join(tracked_keys)}. "
        instruction += f"Initial values - " + ", ".join([f"{k}: {v}" for k, v in initial_values.items()]) + ".\n\n"
        
        if updates:
            instruction += "Updates:\n" + "\n".join(updates) + "\n\n"
        
        question = "What are the current values of all tracked keys?"
        
        # Build reference answer
        reference = current_values
        
        return {
            "instruction": instruction,
            "input": question,
            "output": json.dumps(reference),  # Store as JSON string
            "tracked_keys": tracked_keys,
            "initial_values": initial_values,
            "updates": updates,
            "current_values": reference,
            "n_tracked_keys": n_tracked_keys,
            "n_tracked_updates": len(updates)
        }

    @staticmethod
    def _lengthen_item(item: str, target_length: int) -> str:
        """Lengthen item using cap-strip method (capitalize first letter and repeat)."""
        if len(item) >= target_length:
            return item[:target_length]
        
        # Capitalize first letter
        item_cap = item[0].upper() + item[1:] if len(item) > 0 else item
        
        # Repeat the capitalized version to reach target length
        result = item_cap
        while len(result) < target_length:
            result += item_cap
        
        return result[:target_length]


@TEXT_POSTPROCESSORS.register_module('pi_llm')
def pi_llm_postprocess(text: str) -> Dict[str, str]:
    """Extract key-value pairs from model response."""
    
    # Try to parse as JSON first
    try:
        # Look for JSON-like structure
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Parse verbal format
    result = {}
    
    # Pattern 1: "The current value of X is Y"
    pattern1 = r"(?:The )?current value of (\w+) is ([^.,;]+)"
    matches1 = re.findall(pattern1, text, re.IGNORECASE)
    for key, value in matches1:
        result[key.strip()] = value.strip()
    
    # Pattern 2: "X: Y" or "X = Y"
    pattern2 = r"(\w+)\s*[:=]\s*([^,\n;]+)"
    matches2 = re.findall(pattern2, text)
    for key, value in matches2:
        # Avoid overwriting if already found
        if key.strip() not in result:
            result[key.strip()] = value.strip()
    
    return result


class PILLMEvaluator(BaseEvaluator):
    """Evaluator for PI-LLM dataset."""
    
    def score(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate accuracy of key-value tracking.
        
        Args:
            predictions: List of model outputs (already postprocessed to dicts)
            references: List of ground truth dicts (as JSON strings)
        
        Returns:
            Dictionary with accuracy metrics
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        
        total_correct = 0
        total_keys = 0
        total_missing = 0
        details = []
        
        for pred, ref_str in zip(predictions, references):
            # Parse reference if it's a string
            if isinstance(ref_str, str):
                ref = json.loads(ref_str)
            else:
                ref = ref_str
            
            # Ensure pred is a dict
            if isinstance(pred, str):
                pred_dict = pi_llm_postprocess(pred)
            else:
                pred_dict = pred
            
            # Calculate accuracy for this sample
            n_correct = 0
            n_missing = 0
            n_tracked = len(ref)
            
            sample_detail = {
                'reference': ref,
                'prediction': pred_dict,
                'per_key_results': {}
            }
            
            for key, true_value in ref.items():
                if key in pred_dict:
                    # Normalize values for comparison
                    pred_value = str(pred_dict[key]).strip().lower()
                    true_value_norm = str(true_value).strip().lower()
                    
                    is_correct = pred_value == true_value_norm
                    if is_correct:
                        n_correct += 1
                    
                    sample_detail['per_key_results'][key] = {
                        'predicted': pred_dict[key],
                        'true': true_value,
                        'correct': is_correct
                    }
                else:
                    n_missing += 1
                    sample_detail['per_key_results'][key] = {
                        'predicted': None,
                        'true': true_value,
                        'correct': False
                    }
            
            accuracy = n_correct / n_tracked if n_tracked > 0 else 0
            sample_detail['accuracy'] = accuracy
            sample_detail['n_correct'] = n_correct
            sample_detail['n_missing'] = n_missing
            sample_detail['n_tracked'] = n_tracked
            
            details.append(sample_detail)
            total_correct += n_correct
            total_keys += n_tracked
            total_missing += n_missing
        
        overall_accuracy = total_correct / total_keys if total_keys > 0 else 0
        missing_rate = total_missing / total_keys if total_keys > 0 else 0
        
        return {
            'accuracy': overall_accuracy * 100,  # Convert to percentage
            'missing_rate': missing_rate * 100,
            'total_correct': total_correct,
            'total_keys': total_keys,
            'total_missing': total_missing,
            'details': details
        }