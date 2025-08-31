#!/usr/bin/env python3
"""
Compare variances between a batch JSON and the uniform dataset.
Saves a small JSON report under results/ and prints a compact summary.
"""
import json
import os
import numpy as np

BATCH_FILE = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results\batch_simulations\lokisimulator\boundsbasedsampler\2025-08-17\batch_100sims_20250817_230722.json"
UNIFORM_FILE = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\data\SampleEfficiency\O2_simple_uniform.txt"
OUT_DIR = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results"
OUT_JSON = os.path.join(OUT_DIR, "compat_variances_batch_100.json")


def load_batch(batch_file=BATCH_FILE):
    with open(batch_file, "r") as f:
        data = json.load(f)
    k_values = np.array([ps["k_values"] for ps in data["parameter_sets"]])
    compositions = np.array(data["compositions"])  # expanded over pressures
    return k_values, compositions


def load_uniform(uniform_file=UNIFORM_FILE):
    arr = np.loadtxt(uniform_file)
    k_vals = arr[:, :3]
    comps = arr[:, -3:]
    return k_vals, comps


def summarize_stats(a):
    return {
        "mean": a.mean().item(),
        "variance": a.var(ddof=0).item(),
        "std": a.std(ddof=0).item(),
        "count": int(a.size)
    }


def main():
    k_batch, comp_batch = load_batch()
    k_unif, comp_unif = load_uniform()

    # per-k variances (raw and log10)
    stats = {"k": {}, "k_log10": {}, "composition": {}}

    for i in range(3):
        stats["k"][f"k{i}"] = {
            "batch": summarize_stats(k_batch[:, i]),
            "uniform": summarize_stats(k_unif[:, i])
        }
        # safe log10 (avoid zeros)
        kb = k_batch[:, i]
        ku = k_unif[:, i]
        stats["k_log10"][f"k{i}"] = {
            "batch": summarize_stats(np.log10(kb)),
            "uniform": summarize_stats(np.log10(ku))
        }

    for i in range(3):
        stats["composition"][f"s{i}"] = {
            "batch": summarize_stats(comp_batch[:, i]),
            "uniform": summarize_stats(comp_unif[:, i])
        }

    # compute variance ratios (batch / uniform)
    ratios = {"k_variance_ratio": {}, "klog_variance_ratio": {}, "comp_variance_ratio": {}}
    for i in range(3):
        vb = stats["k"][f"k{i}"]["batch"]["variance"]
        vu = stats["k"][f"k{i}"]["uniform"]["variance"]
        ratios["k_variance_ratio"][f"k{i}"] = vb / vu if vu > 0 else None
        vb_log = stats["k_log10"][f"k{i}"]["batch"]["variance"]
        vu_log = stats["k_log10"][f"k{i}"]["uniform"]["variance"]
        ratios["klog_variance_ratio"][f"k{i}"] = vb_log / vu_log if vu_log > 0 else None
        vcb = stats["composition"][f"s{i}"]["batch"]["variance"]
        vcu = stats["composition"][f"s{i}"]["uniform"]["variance"]
        ratios["comp_variance_ratio"][f"s{i}"] = vcb / vcu if vcu > 0 else None

    report = {"stats": stats, "ratios": ratios}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # print compact summary
    print("VARIANCE COMPARISON SUMMARY")
    print("Per-k variance ratios (batch / uniform):")
    for i in range(3):
        r = ratios["k_variance_ratio"][f"k{i}"]
        print(f"  k{i}: {r:.3g}")
    print("Per-k log10-variance ratios:")
    for i in range(3):
        r = ratios["klog_variance_ratio"][f"k{i}"]
        print(f"  k{i}: {r:.3g}")
    print("Per-species composition variance ratios:")
    for i in range(3):
        r = ratios["comp_variance_ratio"][f"s{i}"]
        print(f"  s{i}: {r:.3g}")
    print(f"\nJSON report saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
