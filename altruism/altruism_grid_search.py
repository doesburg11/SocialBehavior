
import itertools
from altruism_model import AltruismModel, Params
import numpy as np


def run_and_report(params, steps=[1000]):
    model = AltruismModel(params)
    results = {}
    max_step = max(steps)

    for t in range(1, max_step+1):
        model.go()
        if t in steps:
            pink, green, black = model.counts()
            pop = pink + green
            total = pink + green + black
            results[t] = {
                'altruists': pink,
                '%_altruists': 100 * pink / pop if pop > 0 else 0,
                'selfish': green,
                '%_selfish': 100 * green / pop if pop > 0 else 0,
                'black': black,
                '%_black': 100 * black / total if total > 0 else 0,
                'pop': pop,
                'total': total
            }
    return results


def main():
    # Default values (not used)
    def clamp(val):
        return min(max(val, 0.00), 1.0 - 0.00)

    # Fine grid around coexistence point
    grid = {
        'benefit_from_altruism': [clamp(round(x, 2)) for x in np.arange(0.00, 1.00 + 0.01, 0.01)],
        'cost_of_altruism': [clamp(round(x, 2)) for x in np.arange(0.00, 0.35 + 0.01, 0.01)],
        'disease': [0.26],
        'harshness': [0.96, 0.97, 0.98, 0.99, 1.00],
        # [clamp(round(x, 2)) for x in np.arange(0.85, 0.95 + 0.01, 0.01)]  # 0.85, 0.86, ..., 0.89
    }
    param_names = list(grid.keys())
    param_combos = list(itertools.product(*[grid[k] for k in param_names]))
    print(f"Grid search: {len(param_combos)} combinations\n")
    steps = [1000]
    import os
    found = 0
    n_reps = 10  # Number of replicates per parameter set (increase for finer probability resolution)
    csv_path = 'altruism/grid_search_results.csv'
    completed = set()
    import pandas as pd
    # Only read completed parameter sets from grid_search_results.csv
    if os.path.exists(csv_path):
        try:
            df_conv = pd.read_csv(csv_path)
            for row in df_conv.itertuples(index=False):
                completed.add((
                    round(row.benefit_from_altruism, 6),
                    round(row.cost_of_altruism, 6),
                    round(row.disease, 6),
                    round(row.harshness, 6)
                ))
        except Exception as e:
            print(f"Warning: Could not read CSV: {e}")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    total = len(param_combos)
    completed_count = 0
    for combo in param_combos:
        param_dict = dict(zip(param_names, combo))
        param_tuple = (
            round(param_dict['benefit_from_altruism'], 6),
            round(param_dict['cost_of_altruism'], 6),
            round(param_dict['disease'], 6),
            round(param_dict['harshness'], 6)
        )
        if param_tuple in completed:
            completed_count += 1
            continue  # Skip completed
        coexist_count = 0
        altruists_sum = 0
        selfish_sum = 0
        black_sum = 0
        pop_sum = 0
        for rep in range(n_reps):
            params = Params(
                width=51, height=51, torus=True,
                altruistic_probability=0.39,
                selfish_probability=0.39,
                **param_dict
            )
            results = run_and_report(params, steps)
            r = results.get(1000, None)
            if r and r['altruists'] > 0 and r['selfish'] > 0:
                coexist_count += 1
            if r:
                altruists_sum += r['altruists']
                selfish_sum += r['selfish']
                black_sum += r['black']
                pop_sum += r['pop']
        coexist_prob = coexist_count / n_reps
        if coexist_prob > 0:
            found += 1
            print(f"\nParams: {param_dict}")
            print(f"Coexistence probability: {coexist_prob:.2f} ({coexist_count}/{n_reps})")
        row = {
            'benefit_from_altruism': float(param_dict['benefit_from_altruism']),
            'cost_of_altruism': float(param_dict['cost_of_altruism']),
            'disease': float(param_dict['disease']),
            'harshness': float(param_dict['harshness']),
            'coexist_prob': float(coexist_prob)
        }
        # Append to CSV after each run, only these 5 columns
        import csv
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'benefit_from_altruism',
                'cost_of_altruism', 'disease',
                'harshness',
                'coexist_prob'
            ])
            if write_header:
                writer.writeheader()
                write_header = False
            # Ensure all values are stripped and formatted as floats with no trailing whitespace
            clean_row = {
                k: (
                    "{:.6g}".format(float(v)).strip()
                    if isinstance(v, float) or isinstance(v, int)
                    else str(v).strip()
                )
                for k, v in row.items()
            }
            writer.writerow(clean_row)
        completed_count += 1
        print(f"Progress: {completed_count} / {total} parameter sets completed ({completed_count/total:.1%})")
    print(f"\nFound {found} parameter sets with coexistence probability > 0.")
    print("Results appended to grid_search_results.csv after each run.")


if __name__ == "__main__":
    main()
