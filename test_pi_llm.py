#!/usr/bin/env python
"""Test script to verify PI-LLM dataset implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opencompass.datasets import PILLMDataset

def test_pi_llm_dataset():
    """Test basic functionality of PI-LLM dataset."""
    print("Testing PI-LLM dataset implementation...")
    
    # Test 1: Basic dataset loading
    print("\n1. Testing basic dataset loading (Test 1 - varying updates)...")
    try:
        dataset = PILLMDataset.load(
            source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
            n_tracked_keys=[10],
            n_tracked_updates=[2, 4],
            n_samples_per_config=2,
            seed=42
        )
        print(f"   ✓ Loaded {len(dataset)} samples")
        print(f"   Sample keys: {list(dataset[0].keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: Test with item length variations (Test 4)
    print("\n2. Testing item length variations (Test 4)...")
    try:
        dataset = PILLMDataset.load(
            source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
            n_tracked_keys=[5],
            n_tracked_updates=[2],
            len_item=[5, 10],
            len_item_style='cap-strip',
            n_samples_per_config=1,
            seed=42
        )
        print(f"   ✓ Loaded {len(dataset)} samples with item length variations")
        # Check if items are properly lengthened
        sample = dataset[0]
        if 'initial_values' in sample:
            for key, value in sample['initial_values'].items():
                print(f"   Key '{key}': value '{value}' (length: {len(value)})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 3: Test sequential updates (Test 5)
    print("\n3. Testing sequential updates (Test 5)...")
    try:
        dataset = PILLMDataset.load(
            source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
            n_tracked_keys=[5],
            n_tracked_updates=[4],
            random_update=[0],  # Sequential
            n_samples_per_config=1,
            seed=42
        )
        print(f"   ✓ Loaded {len(dataset)} samples with sequential updates")
        sample = dataset[0]
        if 'updates' in sample:
            print(f"   Updates: {sample['updates'][:3]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 4: Postprocessor
    print("\n4. Testing postprocessor...")
    try:
        from opencompass.datasets import pi_llm_postprocess
        
        test_responses = [
            '{"bird": "eagle", "fish": "salmon"}',
            'The current value of bird is eagle. The current value of fish is salmon.',
            'bird: eagle\nfish: salmon',
        ]
        
        for i, response in enumerate(test_responses):
            result = pi_llm_postprocess(response)
            print(f"   Response {i+1}: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    # Make sure we're in the opencompass directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    success = test_pi_llm_dataset()
    sys.exit(0 if success else 1)