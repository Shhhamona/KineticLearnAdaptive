batch1 = 4000
batch2 = 1000

iterations = 4              # iterations per batch before switching logic triggers
samples_per_iteration = 100
min_available_threshold = 100  # switch to smaller batch when available < this

base_krange = 1.0

def simulate_batch(batch_size, iterations, samples_per_iteration, start_shrink_power=0):
    """Simulate iterations for a given batch. Returns list of (iter_idx, krange, available_before, selected)."""
    results = []
    cumulative_selected = 0
    for i in range(start_shrink_power, start_shrink_power + iterations):
        shrink_factor = 0.5 ** i
        krange = base_krange * shrink_factor
        available_before = int(batch_size * shrink_factor - cumulative_selected)
        if available_before <= 0:
            # no more samples available in this shrunken region
            results.append((i - start_shrink_power + 1, krange, 0, 0))
            break
        selected = min(samples_per_iteration, available_before)
        cumulative_selected += selected
        results.append((i - start_shrink_power + 1, krange, available_before, selected))
    return results, cumulative_selected

# Phase 1: simulate batch1 until we need to switch (available < threshold)
print("Phase 1: batch1 simulation")
batch1_results, used1 = simulate_batch(batch1, iterations, samples_per_iteration, start_shrink_power=0)
switched = False
for it, kr, avail, sel in batch1_results:
    print(f" Iter {it}: k range ±{kr:.4f}x, available_before={avail}, selected={sel}")
    if avail < min_available_threshold:
        switched = True
        break

# If we didn't hit threshold inside the fixed iterations, check whether we should switch after them
if not switched:
    # compute available after the simulated iterations
    last_shrink = 0.5 ** (iterations - 1)
    available_after = int(batch1 * last_shrink - used1)
    print(f" After {iterations} iterations on batch1: available_remaining={available_after}")
    if available_after < min_available_threshold:
        switched = True

# Phase 2: when switching to batch2, simulate and show exactly how many samples are available per iteration
if switched:
    print("\nSwitching to Phase 2: batch2 simulation")
    # For batch2 we typically restart the shrink schedule from full (shrink_power=0).
    # We simulate until no more samples are available in the shrinking regions or up to a safety limit.
    max_iterations_batch2 = 10
    batch2_results, used2 = simulate_batch(batch2, max_iterations_batch2, samples_per_iteration, start_shrink_power=0)
    for it, kr, avail, sel in batch2_results:
        print(f" Batch2 Iter {it}: k range ±{kr:.4f}x, available_before={avail}, selected={sel}")
    total_available_batch2_after = batch2 - used2
    print(f"\nBatch2: total selected={used2}, total remaining in raw batch2={total_available_batch2_after}")
else:
    print("\nNo switch to batch2 required within the simulated iterations.")