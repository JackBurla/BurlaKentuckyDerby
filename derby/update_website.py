"""
update_website.py
-----------------
Reads derby/data/trillion_results.json (output of derby_trillion.py)
and patches docs/index.html with the new win/place/show percentages,
updates the hero sim-count stat, and updates the timing callout.

Run after derby_trillion.py completes:
    python3.12 derby/update_website.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(ROOT, "derby", "data", "trillion_results.json")
WEBSITE_FILE = os.path.join(ROOT, "docs", "index.html")


def fmt_sims(n: int) -> str:
    """Format sim count for display: 1000000000000 -> '1,000,000,000,000'"""
    return f"{n:,}"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run derby_trillion.py first.")
        sys.exit(1)

    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    horses = data["horses"]
    total_sims = data["total_sims"]
    elapsed = data.get("elapsed_s", 0)
    backend = data.get("backend", "Burla")

    print(f"Loaded results: {total_sims:,} simulations via {backend}")
    print(f"Elapsed: {elapsed:.1f}s\n")

    # Print summary table
    print(f"{'Horse':<22} {'Win%':>7} {'Place%':>8} {'Show%':>7} {'Value':>6}")
    print("-" * 58)
    for h in horses:
        print(f"{h['name']:<22} {h['winPct']:>7.3f}% {h['placePct']:>7.3f}% "
              f"{h['showPct']:>7.3f}%  {h['value']:>5}")

    with open(WEBSITE_FILE, encoding="utf-8") as f:
        html = f.read()

    # ── 1. Update the HORSES JS array in docs/index.html ────────────────────
    # Build new JS array entries for each horse
    max_win = max(h["winPct"] for h in horses)

    def val_class(v):
        if v == "BET":  return "val-bet"
        if v == "FADE": return "val-fade"
        return "val-fair"

    new_rows_js = "const HORSES = [\n"
    for h in horses:
        new_rows_js += (
            f'  {{ post:{h["post"]}, name:"{h["name"]}", odds:"{h["odds"]}", '
            f'beyer:{h["beyer"]}, style:"{h["style"]}", '
            f'trainerDW:{h["trainerDW"]}, jockeyDW:{h["jockeyDW"]}, '
            f'win:{h["winPct"]:.4f}, place:{h["placePct"]:.4f}, '
            f'show:{h["showPct"]:.4f}, '
            f'implied:{h["impliedPct"]}, val:"{h["value"]}" }},\n'
        )
    new_rows_js += "];"

    # Replace the old HORSES array
    html = re.sub(
        r"const HORSES = \[[\s\S]*?\];",
        new_rows_js,
        html,
    )

    # ── 2. Update hero stat: sim count ──────────────────────────────────────
    html = re.sub(
        r'(<span class="num accent">)([\d,]+)(</span>\s*<span class="label">Monte Carlo simulations)',
        rf'\g<1>{fmt_sims(total_sims)}\3',
        html,
    )

    # ── 3. Update timing callout in "How it ran" section ────────────────────
    elapsed_min = elapsed / 60
    if elapsed_min < 1:
        time_str = f"{elapsed:.0f} seconds"
    elif elapsed_min < 60:
        time_str = f"{elapsed_min:.1f} minutes"
    else:
        time_str = f"{elapsed_min/60:.1f} hours"

    # Update the "in 90 seconds" / timing reference in the page title/lede
    html = re.sub(
        r"1,000,000 Derby simulations,",
        f"{fmt_sims(total_sims)} Derby simulations,",
        html,
    )
    html = re.sub(
        r"in <span class=\"accent\">90 seconds\.</span>",
        f'in <span class="accent">{time_str}.</span>',
        html,
    )

    # Update the stats card lede paragraph
    html = re.sub(
        r"1,000,000 Monte Carlo race simulations",
        f"{fmt_sims(total_sims)} Monte Carlo race simulations",
        html,
    )

    # Update the peak callout worker count
    html = re.sub(
        r"(<span class=\"peak-num\">)100(</span>)",
        rf"\g<1>{data.get('n_workers', 50000):,}\g<2>",
        html,
    )

    with open(WEBSITE_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nWebsite updated: {WEBSITE_FILE}")
    print("Next steps:")
    print("  git add derby/data/trillion_results.json docs/index.html")
    print('  git commit -m "Update website with 1 trillion simulation results"')
    print("  git push")
    print("\nSite will be live at: https://jackburla.github.io/BurlaKentuckyDerby/")


if __name__ == "__main__":
    main()
