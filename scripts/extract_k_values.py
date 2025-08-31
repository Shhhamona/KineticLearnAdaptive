#!/usr/bin/env python3
"""
Extract the actual K values from O2_novib chemistry file
"""

import re

def extract_k_values_from_o2_novib():
    """Extract all constantRateCoeff values from O2_novib chemistry file"""
    k_values = []
    reactions = []
    
    with open('simulFiles/oxygen_novib.chem', 'r') as file:
        for line_num, line in enumerate(file, 1):
            if 'constantRateCoeff' in line and not line.strip().startswith('%'):
                # Split by | to get parts
                parts = line.split('|')
                if len(parts) >= 3:
                    # Get the rate coefficient part (3rd column)
                    rate_part = parts[2].strip()
                    
                    # Extract first numeric value (ignore comments after |)
                    rate_clean = rate_part.split('|')[0].strip()
                    
                    # Handle expressions like "0.75*1.9e-16"
                    if '*' in rate_clean:
                        # Evaluate simple multiplication
                        try:
                            rate_value = eval(rate_clean)
                        except:
                            continue
                    else:
                        # Direct numeric value
                        try:
                            rate_value = float(rate_clean)
                        except:
                            continue
                    
                    k_values.append(rate_value)
                    reaction = parts[0].strip()
                    reactions.append(f"K{len(k_values)-1}: {reaction}")
                    print(f"K{len(k_values)-1} (Line {line_num}): {rate_value:.2e} | {reaction[:60]}...")
    
    print(f"\n📊 Total constantRateCoeff reactions found: {len(k_values)}")
    print(f"📋 K values array: {k_values}")
    return k_values, reactions

if __name__ == "__main__":
    k_vals, reactions = extract_k_values_from_o2_novib()
    
    print(f"\n🔬 For use in code:")
    print(f"full_reference_k = np.array({k_vals})")
