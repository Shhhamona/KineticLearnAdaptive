"""
Calculate K value prediction error bounds from scaled MSE.

This script demonstrates how to convert scaled MSE to actual K value error bounds.
"""

import numpy as np

# Configuration from your training results
# ========================================

# True K value (example from your data)
K_TRUE = 6.00e-16  # True reaction rate coefficient
K_TRUE = 1.30e-15  # True reaction rate coefficient


# MaxAbsScaler scale factor (from output scaler after 1e30 multiplication)
MAX_K_SCALED = 1.19999083e+15  # This is max(abs(K_values * 1e30))
MAX_K_SCALED = 2.59829549e+15 # This is max(abs(K_values * 1e30))


# Training MSE (on scaled data in [-1, 1] range)
SCALED_MSE = 7.785929e-06
SCALED_MSE = 2.483264e-05

# Calculations
# ============

print("="*80)
print("K VALUE PREDICTION ERROR ANALYSIS")
print("="*80)

# Step 1: Convert scaled MSE to scaled RMSE
scaled_rmse = np.sqrt(SCALED_MSE)
print(f"\n1. Scaled RMSE (in [-1, 1] space):")
print(f"   RMSE = sqrt(MSE) = sqrt({SCALED_MSE:.6e}) = {scaled_rmse:.6e}")

# Step 2: Convert to original K*1e30 space
rmse_k_scaled = scaled_rmse * MAX_K_SCALED
print(f"\n2. RMSE in (K × 1e30) space:")
print(f"   RMSE = {scaled_rmse:.6e} × {MAX_K_SCALED:.6e}")
print(f"   RMSE = {rmse_k_scaled:.6e}")

# Step 3: Convert back to original K space (remove 1e30 scaling)
rmse_k_original = rmse_k_scaled / 1e30
print(f"\n3. RMSE in original K space:")
print(f"   RMSE = {rmse_k_scaled:.6e} / 1e30")
print(f"   RMSE = {rmse_k_original:.6e}")

# Step 4: Calculate prediction bounds (±1 RMSE)
k_predicted_min = K_TRUE - rmse_k_original
k_predicted_max = K_TRUE + rmse_k_original

print(f"\n4. Prediction bounds for K = {K_TRUE:.6e}:")
print(f"   K_min = {K_TRUE:.6e} - {rmse_k_original:.6e} = {k_predicted_min:.6e}")
print(f"   K_max = {K_TRUE:.6e} + {rmse_k_original:.6e} = {k_predicted_max:.6e}")

# Step 5: Calculate relative error
relative_error = (rmse_k_original / K_TRUE) * 100
print(f"\n5. Relative error:")
print(f"   Error = {rmse_k_original:.6e} / {K_TRUE:.6e} × 100%")
print(f"   Error = {relative_error:.2f}%")

# Step 6: Worst case scenario (±2 RMSE = ~95% confidence)
print(f"\n6. Worst case prediction bounds (±2 RMSE, ~95% confidence):")
k_worst_min = K_TRUE - 2 * rmse_k_original
k_worst_max = K_TRUE + 2 * rmse_k_original
print(f"   K_min = {k_worst_min:.6e}")
print(f"   K_max = {k_worst_max:.6e}")
print(f"   Relative error = {(2 * rmse_k_original / K_TRUE) * 100:.2f}%")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"True K value:           {K_TRUE:.6e}")
print(f"Prediction error (±1σ): {rmse_k_original:.6e} ({relative_error:.2f}%)")
print(f"Expected range (68%):   [{k_predicted_min:.6e}, {k_predicted_max:.6e}]")
print(f"Worst case (95%):       [{k_worst_min:.6e}, {k_worst_max:.6e}]")
print("="*80)

# Perturbation analysis
print("\n" + "="*80)
print("PERTURBATION ANALYSIS")
print("="*80)
print(f"\nTo test chemical space perturbation, vary K by:")
print(f"  • ±1 RMSE = ±{rmse_k_original:.6e} (±{relative_error:.2f}%)")
print(f"  • ±2 RMSE = ±{2*rmse_k_original:.6e} (±{2*relative_error:.2f}%)")
print(f"\nTest K values:")
print(f"  K_lower = {K_TRUE:.6e} × {(1 - relative_error/100):.6f} = {K_TRUE * (1 - relative_error/100):.6e}")
print(f"  K_upper = {K_TRUE:.6e} × {(1 + relative_error/100):.6f} = {K_TRUE * (1 + relative_error/100):.6e}")
print("="*80)
