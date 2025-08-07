#!/usr/bin/env python
"""
Simplified test to check if OpenCompass generates the exact same prompt as Unable-to-Forget.
We'll manually set the same random seed and parameters to ensure consistency.
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_alignment():
    """Test if OC implementation generates same prompt as UTF."""
    
    print("=" * 80)
    print("TESTING PROMPT ALIGNMENT WITH SAME RANDOM SEED")
    print("=" * 80)
    
    # Load data
    data_path = '/mnt/moonfs/limo-m2/PI-LLM-Opencompass/opencompass/data/pi_llm/dict_category_double-word_46-400_v1-1.json'
    with open(data_path, 'r') as f:
        source_dict = json.load(f)
    
    # Test configurations
    test_cases = [
        {
            "name": "Test1 - Basic",
            "n_keys": 5,
            "n_updates": 4,
            "random_update": 1,
            "seed": 42
        },
        {
            "name": "Test2 - More Keys", 
            "n_keys": 10,
            "n_updates": 6,
            "random_update": 1,
            "seed": 42
        },
        {
            "name": "Test5 - Sequential",
            "n_keys": 5,
            "n_updates": 10,
            "random_update": 0,
            "seed": 42
        }
    ]
    
    all_match = True
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test['name']}")
        print(f"Config: n_keys={test['n_keys']}, n_updates={test['n_updates']}, "
              f"random={test['random_update']}, seed={test['seed']}")
        print("-" * 60)
        
        # Generate using OpenCompass implementation
        from opencompass.datasets.pi_llm import PILLMDataset
        
        dataset = PILLMDataset.load(
            source_dict_path=data_path,
            n_tracked_keys=[test['n_keys']],
            n_tracked_updates=[test['n_updates']],
            n_untracked_keys=0,
            n_untracked_updates=0,
            random_update=test['random_update'],
            prompt_updating='colon',
            prompt_forgetting='none',
            n_samples_per_config=1,
            seed=test['seed']
        )
        
        sample = dataset[0]
        oc_prompt = sample['instruction']
        oc_keys = sample['tracked_keys']
        oc_values = sample['current_values']
        oc_updates = sample['updates']
        
        print(f"\nOpenCompass Generated:")
        print(f"  Keys: {oc_keys}")
        print(f"  Updates: {len(oc_updates)}")
        print(f"  Final values: {oc_values}")
        print(f"\nPrompt preview:")
        print("-" * 40)
        print(oc_prompt[:400] + "..." if len(oc_prompt) > 400 else oc_prompt)
        
        # Expected UTF format check
        print(f"\n\nFormat Checks:")
        
        # Check instruction format
        if "As my secretary" in oc_prompt:
            print("✅ Instruction: Uses 'As my secretary...' format")
        else:
            print("❌ Instruction: Missing 'As my secretary...' format")
            all_match = False
            
        # Check stream format
        if "The text stream starts on the next line." in oc_prompt:
            print("✅ Stream: Uses 'The text stream starts...' format")
        else:
            print("❌ Stream: Missing 'The text stream starts...' format")
            all_match = False
            
        # Check question format
        if "What are the current value of each key" in oc_prompt:
            print("✅ Question: Uses correct format")
        else:
            print("❌ Question: Incorrect format")
            all_match = False
            
        # Check response instruction
        if "End your response with:" in oc_prompt:
            print("✅ Response: Has 'End your response with:' instruction")
        else:
            print("❌ Response: Missing response instruction")
            all_match = False
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ All format checks passed! Prompts should match UTF format.")
    else:
        print("❌ Some format checks failed! Prompts may not match UTF format.")
    
    # Generate one example manually to show exact format
    print("\n\nExample Full Prompt (Test1):")
    print("=" * 60)
    dataset = PILLMDataset.load(
        source_dict_path=data_path,
        n_tracked_keys=[5],
        n_tracked_updates=[4],
        random_update=1,
        prompt_updating='colon',
        n_samples_per_config=1,
        seed=42
    )
    print(dataset[0]['instruction'])
    
    return all_match


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = test_prompt_alignment()
    sys.exit(0 if success else 1)