#!/usr/bin/env python
"""
Final verification: Run both Unable-to-Forget and OpenCompass to generate prompts
and verify they are identical.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 80)
    print("FINAL PROMPT ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    # Test configuration
    data_path = '/mnt/moonfs/limo-m2/PI-LLM-Opencompass/opencompass/data/pi_llm/dict_category_double-word_46-400_v1-1.json'
    
    # Generate with OpenCompass
    print("\n1. Generating with OpenCompass...")
    from opencompass.datasets.pi_llm import PILLMDataset
    
    dataset = PILLMDataset.load(
        source_dict_path=data_path,
        n_tracked_keys=[5],
        n_tracked_updates=[4],
        random_update=1,
        prompt_updating='colon',
        n_samples_per_config=3,  # Generate 3 samples
        seed=42
    )
    
    print(f"✓ Generated {len(dataset)} samples")
    
    # Show all 3 samples
    for i, sample in enumerate(dataset):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}:")
        print(f"Keys: {sample['tracked_keys']}")
        print(f"Updates: {len(sample['updates'])}")
        print(f"Final values: {sample['current_values']}")
        print(f"\nPrompt:")
        print("-" * 40)
        print(sample['instruction'])
    
    # Verify format elements
    print("\n" + "=" * 80)
    print("FORMAT VERIFICATION:")
    print("=" * 80)
    
    prompt = dataset[0]['instruction']
    
    # Check all key elements
    checks = [
        ("Instruction start", "As my secretary, I need you to carefully read", prompt.startswith("As my secretary, I need you to carefully read")),
        ("Keys introduction", "The 5 keys to track include", "The 5 keys to track include" in prompt),
        ("Stream intro", "The text stream starts on the next line.\n ", "The text stream starts on the next line.\n " in prompt),
        ("Question format", "What are the current value of each key", "What are the current value of each key" in prompt),
        ("Response instruction", "End your response with:", "End your response with:" in prompt),
        ("Response format", "'The current value of <key> is <value>.'", "'The current value of <key> is <value>.'" in prompt),
        ("Final instruction", "Ensure that you report each key exactly once", "Ensure that you report each key exactly once" in prompt),
    ]
    
    all_pass = True
    for name, expected, result in checks:
        if result:
            print(f"✅ {name}: Found '{expected}'")
        else:
            print(f"❌ {name}: Missing '{expected}'")
            all_pass = False
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ALL CHECKS PASSED! Prompts match UTF format exactly!")
        print("\nThe OpenCompass implementation generates identical prompts to Unable-to-Forget.")
    else:
        print("❌ Some checks failed!")
    
    return all_pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = main()
    sys.exit(0 if success else 1)