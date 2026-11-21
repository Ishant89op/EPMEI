#!/usr/bin/env python3
"""
Seed Ensemble Evaluator for ASTE (Aspect Sentiment Triplet Extraction)
Aggregates predictions across seeds (41-45) using majority voting.
Computes Precision, Recall, and F1 for each vote threshold (>=1 to >=5)
and prints a comparison against the original paper and baseline.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

PAPER_F1 = {
    '14lap': 69.64,
    '14res': 80.98,
    '15res': 73.01,
    '16res': 77.41,
}

BASELINE_F1 = {
    '14lap': 65.25,
    '14res': 80.52,
    '15res': 71.44,
    '16res': 78.88,
}


def load_gold_triplets(test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    gold_triplets = {}
    total_gold = 0
    for item in gold_data:
        sid = str(item['token']['id'])
        triplets = set()
        for r in item['token']['relations']:
            # (a_start, a_end, o_start, o_end, sentiment)
            triplets.add((r[0], r[1], r[2], r[3], r[4]))
            total_gold += 1
        gold_triplets[sid] = triplets
    return gold_triplets, total_gold


def load_predictions(pred_file):
    if not os.path.exists(pred_file):
        return None
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentence_preds = defaultdict(set)
    for sid, val in data.items():
        sid_str = str(sid)
        for block in val:
            for trip in block[1]:
                # trip: [[a_start, a_end], [o_start, o_end], sentiment]
                t = (trip[0][0], trip[0][1], trip[1][0], trip[1][1], trip[2])
                sentence_preds[sid_str].add(t)
    return sentence_preds


def evaluate_preds(pred_map, gold_map, total_gold):
    correct = 0
    total_pred = 0
    for sid, gold_set in gold_map.items():
        preds = pred_map.get(sid, set())
        total_pred += len(preds)
        correct += len(preds & gold_set)
    p = (correct / total_pred * 100.0) if total_pred > 0 else 0.0
    r = (correct / total_gold * 100.0) if total_gold > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1, correct, total_pred


def ensemble_dataset(dataset_name, sc_dir, data_dir, seeds=[41, 42, 43, 44, 45]):
    test_file = os.path.join(data_dir, dataset_name, 'test.json')
    if not os.path.exists(test_file):
        print(f"[{dataset_name}] Test file not found: {test_file}")
        return None

    gold_map, total_gold = load_gold_triplets(test_file)

    # Load predictions and results.json from each seed
    seed_preds = []
    single_f1s = []
    for seed in seeds:
        seed_dir = os.path.join(sc_dir, dataset_name, f"seed{seed}")
        if not os.path.isdir(seed_dir):
            seed_dir = os.path.join(sc_dir, dataset_name, f"result{seed}")
        pred_file = os.path.join(seed_dir, 'pred_results.json')
        res_file = os.path.join(seed_dir, 'results.json')

        # Try to read single seed F1
        if os.path.exists(res_file):
            try:
                with open(res_file) as f:
                    r = json.load(f)
                f1_val = r.get('f1_', r.get('f1_with_ner_', 0.0))
                if isinstance(f1_val, (int, float)):
                    single_f1s.append(f1_val * 100.0 if f1_val <= 1.0 else f1_val)
            except Exception:
                pass

        p_map = load_predictions(pred_file)
        if p_map is not None:
            seed_preds.append(p_map)

    if not seed_preds:
        print(f"[{dataset_name}] No pred_results.json found for any seed!")
        return None

    # Count votes per sentence for each triplet
    vote_map = defaultdict(lambda: defaultdict(int))
    for p_map in seed_preds:
        for sid, trips in p_map.items():
            for t in trips:
                vote_map[sid][t] += 1

    results = {
        'num_seeds': len(seed_preds),
        'single_avg_f1': (sum(single_f1s) / len(single_f1s)) if single_f1s else 0.0,
        'votes': {}
    }

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} ({len(seed_preds)} seeds loaded, {total_gold} gold triplets)")
    if single_f1s:
        print(f"Single Model Seed F1s: {['%.2f' % s for s in single_f1s]} | Avg: {results['single_avg_f1']:.2f}%")
        print(f"Paper Target: {PAPER_F1.get(dataset_name, 0.0):.2f}% | Delta (Single vs Paper): {results['single_avg_f1'] - PAPER_F1.get(dataset_name, 0.0):+.2f}%")
    print(f"{'-'*70}")
    print(f"{'Threshold':<16} | {'Precision':<10} | {'Recall':<10} | {'F1 (%)':<10} | {'Δ Paper':<10} | {'Δ Base':<10}")
    print(f"{'-'*70}")

    for k in range(1, len(seed_preds) + 1):
        filtered_pred = defaultdict(set)
        for sid, trip_counts in vote_map.items():
            for t, count in trip_counts.items():
                if count >= k:
                    filtered_pred[sid].add(t)

        p, r, f1, corr, tot = evaluate_preds(filtered_pred, gold_map, total_gold)
        delta_p = f1 - PAPER_F1.get(dataset_name, 0.0)
        delta_b = f1 - BASELINE_F1.get(dataset_name, 0.0)
        results['votes'][k] = {'p': p, 'r': r, 'f1': f1, 'delta_paper': delta_p, 'delta_base': delta_b}
        print(f"Votes >= {k:<10} | {p:<10.2f} | {r:<10.2f} | {f1:<10.2f} | {delta_p:<+10.2f} | {delta_b:<+10.2f}")

    print(f"{'='*70}")
    return results


def main():
    parser = argparse.ArgumentParser(description="ASTE Multi-Seed Ensemble")
    parser.add_argument('--output_dir', type=str, default='result/sc_v5',
                        help='Base output directory containing dataset/seed subdirectories')
    parser.add_argument('--data_dir', type=str, default='data/ASTE-Data-V2-EMNLP2020_pro',
                        help='Data directory containing dataset test.json files')
    parser.add_argument('--datasets', nargs='+', default=['14lap', '14res', '15res', '16res'],
                        help='Datasets to evaluate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43, 44, 45],
                        help='Seeds to include in ensemble')
    parser.add_argument('--save_json', type=str, default='ensemble_results.json',
                        help='Path to save aggregated ensemble JSON')
    args = parser.parse_args()

    all_summary = {}
    for ds in args.datasets:
        res = ensemble_dataset(ds, args.output_dir, args.data_dir, args.seeds)
        if res:
            all_summary[ds] = res

    if all_summary and args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(all_summary, f, indent=2)
        print(f"\nSaved ensemble metrics to {args.save_json}")


if __name__ == '__main__':
    main()
