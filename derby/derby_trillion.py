"""
derby_trillion.py
-----------------
Run 1,000,000,000,000 (1 trillion) Kentucky Derby race simulations on Burla.

Strategy
--------
- 5,000 Burla workers x 200,000,000 sims each = 1,000,000,000,000 total
- Each worker processes 200M sims in chunks of 500,000 (400 chunks/worker)
- Fully vectorized NumPy: Gumbel-max trick replaces the per-sim Python loop
  used in the 1M version — no pure-Python iteration over individual races.

The Gumbel-max trick
--------------------
To sample k items without replacement from categorical(softmax(logits)):
  keys = logits + Gumbel(0, 1) noise
  order = argsort(-keys)[:k]
This is algebraically equivalent to the rejection-sampling version but is
vectorizable across an entire batch in one matrix operation.

Inputs
------
Same log_probs as the 1M run: log(softmax((final_score - mean) / 5.0))
so results are directly comparable — just 1,000,000x more samples.
"""

import sys
import os
import json
import time
import math
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Horse data (canvas scores = final_score from compute_final_scores) ───────
# Order matches the scored_df sort (descending final_score) which is also the
# order the Monte Carlo position-counts are stored in.
HORSES = [
    {"post": 16, "name": "Further Ado",     "odds": "6.5-1",  "beyer": 106,
     "dosage": 2.08,  "style": "Press", "trainerDW": 1, "jockeyDW": 3,
     "score": 50.7,   "impliedPct": 13.3},
    {"post": 11, "name": "Chief Wallabee",  "odds": "8-1",    "beyer":  93,
     "dosage": 1.92,  "style": "Press", "trainerDW": 1, "jockeyDW": 1,
     "score": 49.3,   "impliedPct": 11.1},
    {"post":  8, "name": "The Puma",        "odds": "10-1",   "beyer":  99,
     "dosage": 2.47,  "style": "Close", "trainerDW": 1, "jockeyDW": 1,
     "score": 42.8,   "impliedPct":  9.1},
    {"post": 13, "name": "Emerging Market", "odds": "15-1",   "beyer":  98,
     "dosage": 2.20,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 1,
     "score": 40.1,   "impliedPct":  6.2},
    {"post":  5, "name": "Commandment",     "odds": "6.5-1",  "beyer": 104,
     "dosage": 3.44,  "style": "Stalk", "trainerDW": 1, "jockeyDW": 1,
     "score": 38.0,   "impliedPct": 13.3},
    {"post":  7, "name": "So Happy",        "odds": "6-1",    "beyer":  98,
     "dosage": 7.00,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 2,
     "score": 36.8,   "impliedPct": 14.3},
    {"post": 15, "name": "Six Speed",       "odds": "50-1",   "beyer":  91,
     "dosage": 5.00,  "style": "Pace",  "trainerDW": 0, "jockeyDW": 1,
     "score": 35.6,   "impliedPct":  2.0},
    {"post": 10, "name": "Incredibolt",     "odds": "20-1",   "beyer":  91,
     "dosage": 3.00,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 0,
     "score": 30.7,   "impliedPct":  4.8},
    {"post": 20, "name": "Robusta",         "odds": "50-1",   "beyer":  93,
     "dosage": 3.44,  "style": "Pace",  "trainerDW": 2, "jockeyDW": 0,
     "score": 30.1,   "impliedPct":  2.0},
    {"post":  2, "name": "Albus",           "odds": "30-1",   "beyer":  95,
     "dosage": 2.60,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 0,
     "score": 28.2,   "impliedPct":  3.2},
    {"post":  3, "name": "Intrepido",       "odds": "50-1",   "beyer":  94,
     "dosage": 3.40,  "style": "Press", "trainerDW": 0, "jockeyDW": 0,
     "score": 27.5,   "impliedPct":  2.0},
    {"post":  4, "name": "Litmus Test",     "odds": "30-1",   "beyer":  96,
     "dosage": 3.36,  "style": "Stalk", "trainerDW": 6, "jockeyDW": 0,
     "score": 26.9,   "impliedPct":  3.2},
    {"post":  9, "name": "Wonder Dean",     "odds": "30-1",   "beyer":  95,
     "dosage": 2.11,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 0,
     "score": 26.6,   "impliedPct":  3.2},
    {"post":  1, "name": "Renegade",        "odds": "4-1",    "beyer":  97,
     "dosage": 3.00,  "style": "Close", "trainerDW": 2, "jockeyDW": 0,
     "score": 26.3,   "impliedPct": 18.2},
    {"post": 14, "name": "Pavlovian",       "odds": "30-1",   "beyer":  93,
     "dosage": 1.29,  "style": "Pace",  "trainerDW": 2, "jockeyDW": 0,
     "score": 23.4,   "impliedPct":  3.2},
    {"post": 12, "name": "Potente",         "odds": "20-1",   "beyer":  96,
     "dosage": 3.44,  "style": "Pace",  "trainerDW": 6, "jockeyDW": 0,
     "score": 23.2,   "impliedPct":  4.8},
    {"post": 18, "name": "Great White",     "odds": "50-1",   "beyer":  94,
     "dosage": 3.00,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 0,
     "score": 22.2,   "impliedPct":  2.0},
    {"post":  6, "name": "Danon Bourbon",   "odds": "20-1",   "beyer":  94,
     "dosage": 1.86,  "style": "Stalk", "trainerDW": 0, "jockeyDW": 0,
     "score": 20.4,   "impliedPct":  4.8},
    {"post": 19, "name": "Ocelli",          "odds": "50-1",   "beyer":  81,
     "dosage": 1.73,  "style": "Deep",  "trainerDW": 0, "jockeyDW": 0,
     "score": 16.2,   "impliedPct":  2.0},
    {"post": 17, "name": "Golden Tempo",    "odds": "30-1",   "beyer":  90,
     "dosage": 3.00,  "style": "Deep",  "trainerDW": 0, "jockeyDW": 0,
     "score":  9.9,   "impliedPct":  3.2},
]

NOISE_SIGMA     = 1.8           # same as derby_montecarlo.py

# ── Scale settings ────────────────────────────────────────────────────────
# 50,000 workers × 20,000,000 sims each = 1,000,000,000,000 (1 trillion).
# Each worker finishes in ~10–20 seconds (well under Burla's function timeout).
# max_parallelism=500 keeps concurrent vCPUs at 500, safely under the GCP
# CPUS_PER_VM_FAMILY quota of 2,081. Workers queue and cycle through in
# ~100 rounds of 500 concurrent workers → wall time ~15–30 minutes.
N_WORKERS       = 50_000        # total Burla workers dispatched
SIMS_PER_WORKER = 20_000_000    # 20M per worker → ~10–20s per worker
CHUNK_SIZE      = 100_000       # 100K per chunk (200 chunks/worker), ~16MB RAM
TOTAL_SIMS      = N_WORKERS * SIMS_PER_WORKER  # 1,000,000,000,000


def _compute_log_probs(horses):
    """Reproduce the exact log_probs used in derby_montecarlo.py."""
    scores = np.array([h["score"] for h in horses], dtype=np.float64)
    # mirror: exp_s = np.exp((final_score - mean) / 5.0)
    exp_s = np.exp((scores - scores.mean()) / 5.0)
    win_probs = exp_s / exp_s.sum()
    return np.log(win_probs + 1e-9).tolist()


def simulate_race_batch(log_probs_list: list, sims_per_worker: int,
                        chunk_size: int, seed: int) -> dict:
    """
    Run sims_per_worker races on one Burla worker.

    Fully vectorized: no Python loop per individual race.
    Uses the Gumbel-max trick for categorical sampling without replacement.

    Burla unpacks args as positional arguments so the signature must match
    the tuple elements exactly.
    """
    import subprocess, sys as _sys
    try:
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "numpy", "-q"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(seed)
    n_horses = len(log_probs_list)
    log_probs = np.array(log_probs_list, dtype=np.float64)
    counts = np.zeros((n_horses, 4), dtype=np.int64)

    n_full_chunks, remainder = divmod(sims_per_worker, chunk_size)

    def _process_chunk(size: int) -> None:
        # Step 1: add Gaussian noise in log-prob space (same as 1M version)
        noise = rng.standard_normal((size, n_horses)) * NOISE_SIGMA
        noisy = log_probs + noise   # (size, n_horses)

        # Step 2: log-softmax for numerical stability
        row_max = noisy.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(noisy - row_max).sum(axis=1, keepdims=True)) + row_max
        log_p = noisy - log_sum_exp  # (size, n_horses)

        # Step 3: Gumbel-max trick — add Gumbel(0,1) to get a sample ordering
        gumbel_noise = rng.gumbel(0.0, 1.0, (size, n_horses))
        keys = log_p + gumbel_noise  # (size, n_horses)

        # Step 4: partial sort — argpartition O(n) then sort top-4 by value
        part = np.argpartition(-keys, 4, axis=1)[:, :4]   # (size, 4) — unordered top-4
        top_keys = -keys[np.arange(size)[:, None], part]   # flip sign for ascending sort
        rank_order = np.argsort(top_keys, axis=1)          # sort within the 4
        order = part[np.arange(size)[:, None], rank_order] # (size, 4) ordered 1st..4th

        # Step 5: accumulate position tallies
        for pos in range(4):
            np.add.at(counts[:, pos], order[:, pos], 1)

    for _ in range(n_full_chunks):
        _process_chunk(chunk_size)
    if remainder:
        _process_chunk(remainder)

    return {"counts": counts.tolist(), "n_sims": sims_per_worker}


def kelly_fraction(win_prob: float, odds_str: str) -> float:
    """Kelly criterion, capped at 25% of bankroll."""
    try:
        num, denom = odds_str.split("-")
        b = float(num) / float(denom)   # net odds e.g. "6.5-1" -> b=6.5
    except Exception:
        return 0.0
    p, q = win_prob, 1.0 - win_prob
    k = (b * p - q) / b if b > 0 else 0.0
    return round(max(0.0, min(k, 0.25)), 3)


def main():
    log_probs = _compute_log_probs(HORSES)
    n_horses  = len(HORSES)

    print(f"1 Trillion Simulation Kentucky Derby Model")
    print(f"==========================================")
    print(f"Workers   : {N_WORKERS:,}")
    print(f"Sims/wrkr : {SIMS_PER_WORKER:,}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"Total sims: {TOTAL_SIMS:,}")
    print()

    args_list = [
        (log_probs, SIMS_PER_WORKER, CHUNK_SIZE, seed)
        for seed in range(N_WORKERS)
    ]

    t0 = time.time()

    try:
        from burla import remote_parallel_map
        # grow=True, max_parallelism=500: provision enough VMs for 500
        # concurrent workers (500 vCPUs, well under the 2,081 quota).
        # 50,000 total workers queue and run in ~100 rounds of 500.
        print(f"Dispatching {N_WORKERS:,} workers to Burla cluster...")
        print(f"  grow=True, max_parallelism=500, func_cpu=1")
        results = remote_parallel_map(
            simulate_race_batch, args_list,
            func_cpu=1, func_ram=4,
            max_parallelism=500,
            grow=True,
        )
        backend = "Burla"
        print(f"Burla returned {len(results):,} results.")
    except Exception as exc:
        print(f"Burla unavailable ({exc}), falling back to local ThreadPoolExecutor...")
        from concurrent.futures import ThreadPoolExecutor
        # Locally, cap at a smaller run to avoid multi-hour waits
        local_workers = min(N_WORKERS, 8)
        local_args = [(log_probs, min(SIMS_PER_WORKER, 20_000_000), CHUNK_SIZE, seed)
                      for seed in range(local_workers)]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            futures = [ex.submit(simulate_race_batch, *a) for a in local_args]
            results = [f.result() for f in futures]
        backend = "local"
        print(f"Local fallback: {local_workers} workers, {local_workers * 1_000_000:,} total sims.")

    elapsed = time.time() - t0
    actual_sims = sum(r["n_sims"] for r in results)

    # Aggregate counts across all workers
    total_counts = np.zeros((n_horses, 4), dtype=np.int64)
    for r in results:
        arr = np.array(r["counts"], dtype=np.int64)
        if arr.shape == total_counts.shape:
            total_counts += arr

    # Derive probabilities
    win_pct   = (total_counts[:, 0] / actual_sims * 100)
    place_pct = ((total_counts[:, 0] + total_counts[:, 1]) / actual_sims * 100)
    show_pct  = (total_counts[:, :3].sum(axis=1) / actual_sims * 100)

    # Build per-horse output
    output_horses = []
    for i, h in enumerate(HORSES):
        wp = round(float(win_pct[i]), 4)
        pp = round(float(place_pct[i]), 4)
        sp = round(float(show_pct[i]), 4)
        kf = kelly_fraction(wp / 100, h["odds"])
        implied = h["impliedPct"]
        if wp > implied * 1.15:
            val = "BET"
        elif wp < implied * 0.85:
            val = "FADE"
        else:
            val = "FAIR"
        output_horses.append({**h, "winPct": wp, "placePct": pp, "showPct": sp,
                               "value": val, "kelly": kf})

    output = {
        "total_sims":   actual_sims,
        "elapsed_s":    round(elapsed, 2),
        "backend":      backend,
        "n_workers":    N_WORKERS,
        "noise_sigma":  NOISE_SIGMA,
        "horses":       output_horses,
    }

    out_path = os.path.join(DATA_DIR, "trillion_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Print summary table
    print(f"\n{'Horse':<22} {'Win%':>7} {'Place%':>8} {'Show%':>7} {'Value':>6}  Kelly")
    print("-" * 62)
    for h in output_horses:
        print(f"{h['name']:<22} {h['winPct']:>7.3f}% {h['placePct']:>7.3f}% "
              f"{h['showPct']:>7.3f}%  {h['value']:>5}  {h['kelly']:.3f}")

    print(f"\nDone: {actual_sims:,} simulations in {elapsed:.1f}s via {backend}")
    return output


if __name__ == "__main__":
    main()
