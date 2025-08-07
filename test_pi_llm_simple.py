#!/usr/bin/env python
"""Simple test to check PI-LLM dataset loading."""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_loading():
    """Test basic PI-LLM functionality without OpenCompass framework."""
    print("Testing PI-LLM basic loading...")
    
    # Load source data directly
    data_path = 'data/pi_llm/dict_category_double-word_46-400_v1-1.json'
    with open(data_path, 'r') as f:
        source_dict = json.load(f)
    
    print(f"✓ Loaded source dictionary with {len(source_dict)} categories")
    print(f"  Sample categories: {list(source_dict.keys())[:5]}")
    
    # Test item lengthening function
    def lengthen_item(item, target_length):
        if len(item) >= target_length:
            return item[:target_length]
        item_cap = item[0].upper() + item[1:] if len(item) > 0 else item
        result = item_cap
        while len(result) < target_length:
            result += item_cap
        return result[:target_length]
    
    test_item = "cat"
    lengthened = lengthen_item(test_item, 10)
    print(f"\n✓ Item lengthening test:")
    print(f"  Original: '{test_item}' (length: {len(test_item)})")
    print(f"  Lengthened to 10: '{lengthened}' (length: {len(lengthened)})")
    
    # Test update generation
    import random
    random.seed(42)
    
    # Sample some keys
    categories = list(source_dict.keys())
    tracked_keys = random.sample(categories, 5)
    print(f"\n✓ Sampled keys: {tracked_keys}")
    
    # Generate some updates
    updates = []
    current_values = {}
    
    # Initialize values
    for key in tracked_keys:
        current_values[key] = random.choice(source_dict[key])
    
    print(f"\n✓ Initial values:")
    for k, v in current_values.items():
        print(f"  {k}: {v}")
    
    # Generate updates
    for i in range(5):
        key = tracked_keys[i % len(tracked_keys)]  # Sequential
        old_value = current_values[key]
        available = [item for item in source_dict[key] if item != old_value]
        if available:
            new_value = random.choice(available)
            current_values[key] = new_value
            updates.append(f"{key}: {new_value}")
    
    print(f"\n✓ Generated {len(updates)} updates (sequential):")
    for update in updates[:3]:
        print(f"  {update}")
    print("  ...")
    
    print("\n✅ Basic functionality test passed!")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_basic_loading()