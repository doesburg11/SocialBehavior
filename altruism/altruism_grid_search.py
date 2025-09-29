import itertools
from altruism_model import AltruismModel, Params
import numpy as np
import pandas as pd
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def simulate_param_set(combo, param_names, n_reps, steps):
    param_dict = dict(zip(param_names, combo))
    coexist_count = 0
    altruists_sum = 0
    selfish_sum = 0
    black_sum = 0
    for rep in range(n_reps):
        params = Params(
            width=51, height=51, torus=True,
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
    coexist_prob = coexist_count / n_reps
    altruist_avg = altruists_sum / n_reps
    selfish_avg = selfish_sum / n_reps
    black_avg = black_sum / n_reps
    row = {
        'benefit_from_altruism': float(param_dict['benefit_from_altruism']),
        'cost_of_altruism': float(param_dict['cost_of_altruism']),
        'disease': float(param_dict['disease']),
        'harshness': float(param_dict['harshness']),
        'altruistic_probability': float(param_dict['altruistic_probability']),
        'selfish_probability': float(param_dict['selfish_probability']),
        'coexist_prob': float(coexist_prob),
        'altruist_avg': float(altruist_avg),
        'selfish_avg': float(selfish_avg),
        'black_avg': float(black_avg)
    }
    param_tuple = (
        round(param_dict['benefit_from_altruism'], 6),
        round(param_dict['cost_of_altruism'], 6),
        round(param_dict['disease'], 6),
        round(param_dict['harshness'], 6),
        round(param_dict['altruistic_probability'], 6),
        round(param_dict['selfish_probability'], 6)
    )
    return param_tuple, row, coexist_prob, coexist_count


def main():
    # Default values (not used)
    def clamp(val):
        return min(max(val, 0.00), 1.0 - 0.00)

    # Fine grid around coexistence point
    grid = {
        'benefit_from_altruism': [clamp(round(x, 2)) for x in np.arange(0.00, 1.00 + 0.01, 0.01)],
        'cost_of_altruism': [clamp(round(x, 2)) for x in np.arange(0.00, 0.35 + 0.01, 0.01)],
        'disease': [0.26],
        'harshness': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'altruistic_probability': [0.39],
        'selfish_probability': [0.39]
    }
    param_names = list(grid.keys())  # <-- Now includes altruistic_probability and selfish_probability
    param_combos = list(itertools.product(*[grid[k] for k in param_names]))  # <-- Now includes all combinations
    print(f"Grid search: {len(param_combos)} combinations\n")
    steps = [1000]
    found = 0
    n_reps = 5  # Number of replicates per parameter set (increase for finer probability resolution)
    csv_path = 'altruism/grid_search_results_extended.csv'
    completed = set()
    # Only read completed parameter sets from grid_search_results.csv
    if os.path.exists(csv_path):
        try:
            df_conv = pd.read_csv(csv_path)
            for row in df_conv.itertuples(index=False):
                completed.add((
                    round(row.benefit_from_altruism, 6),
                    round(row.cost_of_altruism, 6),
                    round(row.disease, 6),
                    round(row.harshness, 6),
                    round(row.altruistic_probability, 6),
                    round(row.selfish_probability, 6)
                ))
        except Exception as e:
            print(f"Warning: Could not read CSV: {e}")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    total = len(param_combos)
    batch_size = 1000  # Number of results to write at once
    results_buffer = []
    fieldnames = [
        'benefit_from_altruism',
        'cost_of_altruism', 'disease',
        'harshness',
        'altruistic_probability',
        'selfish_probability',
        'coexist_prob',
        'altruist_avg',
        'selfish_avg',
        'black_avg'
    ]
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = []
        for combo in param_combos:
            param_dict = dict(zip(param_names, combo))
            param_tuple = (
                round(param_dict['benefit_from_altruism'], 6),
                round(param_dict['cost_of_altruism'], 6),
                round(param_dict['disease'], 6),
                round(param_dict['harshness'], 6),
                round(param_dict['altruistic_probability'], 6),
                round(param_dict['selfish_probability'], 6)
            )
            if param_tuple in completed:
                continue
            futures.append(executor.submit(simulate_param_set, combo, param_names, n_reps, steps))
        completed_count = len(completed)
        for future in as_completed(futures):
            param_tuple, row, coexist_prob, coexist_count = future.result()
            completed_count += 1
            results_buffer.append({
                k: (
                    "{:.6g}".format(float(v)).strip()
                    if isinstance(v, float) or isinstance(v, int)
                    else str(v).strip()
                )
                for k, v in row.items()
            })
            if coexist_prob > 0:
                found += 1
                print(f"\nParams: {row}")
                print(f"Coexistence probability: {coexist_prob:.2f} ({coexist_count}/{n_reps})")
            if len(results_buffer) >= batch_size:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerows(results_buffer)
                results_buffer = []
            print(f"Progress: {completed_count} / {total} parameter sets completed ({completed_count/total:.1%})")
        # Write any remaining results
        if results_buffer:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(results_buffer)
    print(f"\nFound {found} parameter sets with coexistence probability > 0.")
    print("Results appended to grid_search_results_extended.csv in batches.")


if __name__ == "__main__":
    main()
