#!/usr/bin/env python
"""
Test dynamic prompt generation from both frameworks to ensure they match.
This actually uses the real implementations from both frameworks.
"""

import json
import sys
import os

# Add paths for both frameworks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Unable-to-Forget'))

def test_dynamic_generation():
    """Test actual dynamic generation from both frameworks."""
    
    print("=" * 80)
    print("DYNAMIC PROMPT GENERATION TEST")
    print("=" * 80)
    
    # Load test data
    data_path = 'data/pi_llm/dict_category_double-word_46-400_v1-1.json'
    with open(data_path, 'r') as f:
        source_dict = json.load(f)
    
    # Test 1: Import and use Unable-to-Forget's actual implementation
    print("\n1. Testing Unable-to-Forget implementation...")
    try:
        from core.pi_flow_upgrade import prepare_input_text
        
        # UTF parameters matching test1
        input_text, list_all_pairs, list_tracked_keys, list_untracked_keys, \
        tracked_key_usage, untracked_key_usage, tracked_item_usage, untracked_item_usage, \
        list_forget_keys, dict_tracked_key_value = prepare_input_text(
            source_dict=source_dict,
            n_tracked_keys=5,
            n_untracked_keys=0,
            n_tracked_updates=4,
            n_untracked_updates=0,
            random_update=1,
            prompt_updating='colon',
            prompt_forgetting='none',
            probe_target='current',
            lengthen_item='1_none',
            balanced_sample=1,
            sample_replacement=0,
            memory_limit=1,
            remix_category=0,
            streamloc_forget_at=None,
            keyloc_forget_at=None,
            response_format='verbal_redundant'
        )
        
        utf_prompt = input_text
        utf_tracked_keys = list_tracked_keys
        utf_final_values = dict_tracked_key_value
        
        print(f"✓ Generated UTF prompt (length: {len(utf_prompt)})")
        print(f"  Tracked keys: {utf_tracked_keys}")
        print("\nUTF Prompt Preview:")
        print("-" * 40)
        print(utf_prompt[:600] + "..." if len(utf_prompt) > 600 else utf_prompt)
        
    except Exception as e:
        print(f"✗ Failed to generate UTF prompt: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Import and use OpenCompass's actual implementation
    print("\n\n2. Testing OpenCompass implementation...")
    try:
        from opencompass.datasets.pi_llm import PILLMDataset
        
        # Generate a single sample using OC
        import random
        random.seed(42)  # Same seed as UTF
        
        dataset = PILLMDataset.load(
            source_dict_path=data_path,
            n_tracked_keys=[5],
            n_tracked_updates=[4],
            n_untracked_keys=0,
            n_untracked_updates=0,
            random_update=1,
            prompt_updating='colon',
            prompt_forgetting='none',
            n_samples_per_config=1,
            seed=42
        )
        
        sample = dataset[0]
        oc_prompt = sample['instruction']  # Full prompt is in instruction now
        oc_tracked_keys = sample['tracked_keys']
        oc_final_values = sample['current_values']
        
        print(f"✓ Generated OC prompt (length: {len(oc_prompt)})")
        print(f"  Tracked keys: {oc_tracked_keys}")
        print("\nOC Prompt Preview:")
        print("-" * 40)
        print(oc_prompt[:600] + "..." if len(oc_prompt) > 600 else oc_prompt)
        
    except Exception as e:
        print(f"✗ Failed to generate OC prompt: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Compare the prompts
    print("\n\n3. Comparing dynamic prompts...")
    print("=" * 60)
    
    # Check if prompts are identical
    if utf_prompt == oc_prompt:
        print("✅ PROMPTS ARE IDENTICAL!")
    else:
        print("❌ PROMPTS DIFFER!")
        
        # Find differences
        print("\nDetailed comparison:")
        
        # Split into lines for comparison
        utf_lines = utf_prompt.split('\n')
        oc_lines = oc_prompt.split('\n')
        
        print(f"\nLine count: UTF={len(utf_lines)}, OC={len(oc_lines)}")
        
        # Compare line by line
        max_lines = max(len(utf_lines), len(oc_lines))
        for i in range(max_lines):
            utf_line = utf_lines[i] if i < len(utf_lines) else "[MISSING]"
            oc_line = oc_lines[i] if i < len(oc_lines) else "[MISSING]"
            
            if utf_line != oc_line:
                print(f"\nDifference at line {i+1}:")
                print(f"UTF: {utf_line[:100]}...")
                print(f"OC:  {oc_line[:100]}...")
    
    # Check tracked keys
    print("\n\n4. Comparing tracked keys...")
    if utf_tracked_keys == oc_tracked_keys:
        print("✅ Tracked keys match!")
    else:
        print("❌ Tracked keys differ!")
        print(f"UTF: {utf_tracked_keys}")
        print(f"OC:  {oc_tracked_keys}")
    
    # Check final values
    print("\n5. Comparing final values...")
    if utf_final_values == oc_final_values:
        print("✅ Final values match!")
    else:
        print("❌ Final values differ!")
        print(f"UTF: {utf_final_values}")
        print(f"OC:  {oc_final_values}")
    
    # Test with different configurations
    print("\n\n6. Testing multiple configurations...")
    test_configs = [
        {"name": "Test1", "n_keys": 5, "n_updates": 4},
        {"name": "Test2", "n_keys": 10, "n_updates": 2},
        {"name": "Test5 Sequential", "n_keys": 5, "n_updates": 8, "random": 0},
    ]
    
    all_match = True
    for config in test_configs:
        print(f"\n  Testing {config['name']}...")
        
        # UTF generation
        input_text, _, _, _, _, _, _, _, _, dict_tracked_key_value = prepare_input_text(
            source_dict=source_dict,
            n_tracked_keys=config['n_keys'],
            n_untracked_keys=0,
            n_tracked_updates=config['n_updates'],
            n_untracked_updates=0,
            random_update=config.get('random', 1),
            prompt_updating='colon',
            prompt_forgetting='none',
            probe_target='current',
            lengthen_item='1_none',
            balanced_sample=1,
            sample_replacement=0,
            memory_limit=1,
            remix_category=0,
            streamloc_forget_at=None,
            keyloc_forget_at=None,
            response_format='verbal_redundant'
        )
        utf_prompt = input_text
        
        # OC generation
        dataset = PILLMDataset.load(
            source_dict_path=data_path,
            n_tracked_keys=[config['n_keys']],
            n_tracked_updates=[config['n_updates']],
            random_update=config.get('random', 1),
            prompt_updating='colon',
            n_samples_per_config=1,
            seed=42
        )
        oc_prompt = dataset[0]['instruction']
        
        if utf_prompt == oc_prompt:
            print(f"    ✅ {config['name']} prompts match!")
        else:
            print(f"    ❌ {config['name']} prompts differ!")
            all_match = False
    
    return all_match


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = test_dynamic_generation()
    print("\n" + "=" * 60)
    if success:
        print("✅ All dynamic prompt generation tests passed!")
    else:
        print("❌ Some tests failed - prompts are not aligned!")
    sys.exit(0 if success else 1)