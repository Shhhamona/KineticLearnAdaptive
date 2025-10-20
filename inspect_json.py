"""Quick script to inspect JSON structure."""
import json
import numpy as np

json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print("JSON Keys:", list(data.keys()))
print()

if 'compositions' in data:
    compositions = np.array(data['compositions'])
    print(f"compositions shape: {compositions.shape}")
    print(f"compositions type: {type(data['compositions'])}")
    print(f"First element type: {type(data['compositions'][0]) if len(data['compositions']) > 0 else 'N/A'}")

if 'k_values' in data:
    k_values = np.array(data['k_values'])
    print(f"k_values shape: {k_values.shape}")

if 'parameter_sets' in data:
    print(f"parameter_sets length: {len(data['parameter_sets'])}")
    if len(data['parameter_sets']) > 0:
        print(f"First parameter_set keys: {list(data['parameter_sets'][0].keys())}")
        k_from_ps = np.array([ps['k_values'] for ps in data['parameter_sets']])
        print(f"k_values from parameter_sets shape: {k_from_ps.shape}")

print("\nActual structure inspection:")
print(f"len(data['compositions']): {len(data['compositions'])}")
if len(data['compositions']) > 0:
    print(f"len(data['compositions'][0]): {len(data['compositions'][0])}")
    if len(data['compositions'][0]) > 0:
        print(f"len(data['compositions'][0][0]): {len(data['compositions'][0][0])}")
