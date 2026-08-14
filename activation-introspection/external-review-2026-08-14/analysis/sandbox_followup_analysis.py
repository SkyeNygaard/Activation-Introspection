#!/usr/bin/env python3
"""Offline follow-up analyses for Activation-Introspection.

Uses only checked-in raw JSONL and retained activation tensors.  No repo analyzers,
model weights, transformers, or PEFT are required.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path('/mnt/data/activation_introspection/repo_snapshot')
RES = ROOT / 'results'


def loadjl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def both_correct(groups):
    ok = 0
    for rows in groups.values():
        assert len(rows) == 2
        ok += all(r['correct'] if 'correct' in r else r['model_correct'] for r in rows)
    return ok / len(groups), ok, len(groups)


def heldout_semantic():
    rows = loadjl(RES / 'heldout_semantic_v1_raw.jsonl')
    arms = sorted({r['arm'] for r in rows})
    out = {}
    # Common twin key includes arm implicitly after filtering.
    for arm in arms:
        rs = [r for r in rows if r['arm'] == arm]
        groups = defaultdict(list)
        for r in rs:
            groups[(r['pair'], r['carrier_sha'], r['cell_base'])].append(r)
        assert all({x['query_sign'] for x in g} == {-1, 1} for g in groups.values())
        successes = 0
        flips = 0
        flip_correct = 0
        flip_wrong = 0
        by_pair = defaultdict(lambda: {'success': 0, 'n': 0, 'flips': 0})
        for key, g in groups.items():
            assert len(g) == 2
            succ = all(x['model_correct'] for x in g)
            successes += succ
            pred_flip = len({x['model_predicted'] for x in g}) == 2
            flips += pred_flip
            if pred_flip:
                if succ:
                    flip_correct += 1
                else:
                    flip_wrong += 1
            cp = g[0]['category_pair']
            by_pair[cp]['success'] += int(succ)
            by_pair[cp]['flips'] += int(pred_flip)
            by_pair[cp]['n'] += 1
        out[arm] = {
            'twin_successes': successes,
            'twin_n': len(groups),
            'twin_accuracy': successes / len(groups),
            'prediction_flips': flips,
            'flip_correct_direction': flip_correct,
            'flip_wrong_direction': flip_wrong,
            'by_category_pair': {
                k: {
                    **v,
                    'twin_accuracy': v['success'] / v['n'],
                }
                for k, v in sorted(by_pair.items())
            },
        }
    return out


def report_training_zero_demo():
    by_seed = {}
    for seed in range(4):
        rows = loadjl(RES / f'report_training_v3_seed{seed}_raw.jsonl')
        seed_out = {}
        for arm in ['base', 'trained', 'trained_seen_bank']:
            for cond in ['target', 'random', 'shuffled']:
                rs = [r for r in rows if r['arm'] == arm and r['condition'] == cond]
                if not rs:
                    continue
                groups = defaultdict(list)
                for r in rs:
                    groups[(r['concept'], r['carrier_sha256'])].append(r)
                # Every eval concept x carrier is a +/- twin.
                assert all(len(g) == 2 and {x['sign'] for x in g} == {-1, 1} for g in groups.values())
                twin = sum(all(x['correct'] for x in g) for g in groups.values()) / len(groups)
                seed_out[f'{arm}/{cond}'] = {
                    'row_accuracy': mean(r['correct'] for r in rs),
                    'twin_accuracy': twin,
                    'n_twins': len(groups),
                }
        by_seed[str(seed)] = seed_out
    aggregate = {}
    keys = sorted(set().union(*(x.keys() for x in by_seed.values())))
    for k in keys:
        vals = [by_seed[str(s)][k]['twin_accuracy'] for s in range(4) if k in by_seed[str(s)]]
        aggregate[k] = {'mean_twin_accuracy': mean(vals), 'per_seed': vals}
    return {'by_seed': by_seed, 'aggregate': aggregate}


def remap_with_demos():
    by_seed = {}
    for seed in range(3):
        rows = loadjl(RES / f'remap_training_v2_seed{seed}_raw.jsonl')
        seed_out = {}
        for arm in ['base', 'fixed', 'remap']:
            for cond in ['target', 'random']:
                rs = [
                    r for r in rows
                    if r['arm'] == arm and r['condition'] == cond and r['strength'] == 0.5
                ]
                groups = defaultdict(list)
                for r in rs:
                    groups[(r['concept'], r['carrier_sha256'], r['order_key'], r['positive_label'])].append(r)
                assert all(len(g) == 2 and {x['query_sign'] for x in g} == {-1, 1} for g in groups.values())
                twin = sum(all(x['correct'] for x in g) for g in groups.values()) / len(groups)
                seed_out[f'{arm}/{cond}'] = {
                    'row_accuracy': mean(r['correct'] for r in rs),
                    'twin_accuracy': twin,
                    'n_twins': len(groups),
                }
        by_seed[str(seed)] = seed_out
    aggregate = {}
    keys = sorted(set().union(*(x.keys() for x in by_seed.values())))
    for k in keys:
        vals = [by_seed[str(s)][k]['twin_accuracy'] for s in range(3)]
        aggregate[k] = {'mean_twin_accuracy': mean(vals), 'per_seed': vals}
    return {'by_seed': by_seed, 'aggregate': aggregate}


def carrier_centered_centroids(acts, index, read_layer, arm, depths, concepts):
    pos = {(i['arm'], i['inject_layer'], i['concept'], i['carrier_id']): j for j, i in enumerate(index)}
    carriers = sorted({i['carrier_id'] for i in index})
    centroids = {}
    for dep in depths:
        for concept in concepts:
            centered = []
            for carrier in carriers:
                x = acts[pos[(arm, dep, concept, carrier)], read_layer].float()
                bank = torch.stack([
                    acts[pos[(arm, dep, c, carrier)], read_layer].float() for c in concepts
                ])
                centered.append(x - bank.mean(0))
            centroids[(dep, concept)] = torch.stack(centered).mean(0)
    return centroids


def alignment_summary(centroids, depths, concepts):
    same, off = [], []
    right = 0
    total = 0
    eye = torch.eye(len(concepts), dtype=torch.bool)
    for ai in range(len(depths)):
        for bi in range(ai + 1, len(depths)):
            a, b = depths[ai], depths[bi]
            A = torch.stack([centroids[(a, c)] for c in concepts])
            B = torch.stack([centroids[(b, c)] for c in concepts])
            A = A / (A.norm(dim=1, keepdim=True) + 1e-9)
            B = B / (B.norm(dim=1, keepdim=True) + 1e-9)
            S = A @ B.T
            same.extend(torch.diag(S).tolist())
            off.extend(S[~eye].tolist())
            ids = torch.arange(len(concepts))
            right += int((S.argmax(1) == ids).sum()) + int((S.argmax(0) == ids).sum())
            total += 2 * len(concepts)
    return {
        'mean_same_concept_cosine': float(np.mean(same)),
        'mean_off_concept_cosine': float(np.mean(off)),
        'bidirectional_top1_concept_match': right / total,
        'pairwise_depth_comparisons': math.comb(len(depths), 2),
    }


def retained_cross_depth_geometry():
    files = {
        'qwen05b': RES / 'retained_test_qwen05b_v2_raw.acts.pt',
        'qwen15b': RES / 'retained_test_qwen15b_raw.acts.pt',
        'qwen3b': RES / 'retained_test_qwen3b_raw.acts.pt',
    }
    out = {}
    for model, path in files.items():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        acts, index = blob['acts'], blob['index']
        depths = sorted({i['inject_layer'] for i in index})
        concepts = sorted({i['concept'] for i in index})
        final = acts.shape[1] - 1
        model_out = {'shape': list(acts.shape), 'inject_layers': depths, 'final_read_layer': final, 'arms': {}}
        for arm in ['target', 'random', 'shuffled']:
            cents = carrier_centered_centroids(acts, index, final, arm, depths, concepts)
            model_out['arms'][arm] = alignment_summary(cents, depths, concepts)
        # Also report the target-v-control geometry at each newly included injection depth.
        trajectory = []
        for read_layer in range(acts.shape[1]):
            eligible = [d for d in depths if d <= read_layer]
            if len(eligible) < 2:
                continue
            if read_layer not in depths and read_layer != final:
                continue
            row = {'read_layer': read_layer, 'eligible_inject_layers': eligible}
            for arm in ['target', 'random', 'shuffled']:
                cents = carrier_centered_centroids(acts, index, read_layer, arm, eligible, concepts)
                row[arm] = alignment_summary(cents, eligible, concepts)
            trajectory.append(row)
        model_out['trajectory'] = trajectory
        out[model] = model_out
    return out


def main():
    out = {
        'analysis_scope': 'post-hoc exploratory; existing artifacts only; no model forward pass',
        'heldout_semantic': heldout_semantic(),
        'zero_demo_training': report_training_zero_demo(),
        'trained_with_icl_demos': remap_with_demos(),
        'retained_cross_depth_geometry': retained_cross_depth_geometry(),
    }
    dest = Path('/mnt/data/activation_introspection/sandbox_followup_results.json')
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(dest)
    # Human-readable headline checks
    h = out['heldout_semantic']
    print('heldout semantic twin:', h['heldout_semantic']['twin_accuracy'], 'scrambled:', h['heldout_scrambled']['twin_accuracy'])
    print('semantic flips:', h['heldout_semantic']['prediction_flips'], 'correct-way:', h['heldout_semantic']['flip_correct_direction'], 'wrong-way:', h['heldout_semantic']['flip_wrong_direction'])
    z = out['zero_demo_training']['aggregate']
    print('zero-demo trained target/random/shuffled:', z['trained/target']['mean_twin_accuracy'], z['trained/random']['mean_twin_accuracy'], z['trained/shuffled']['mean_twin_accuracy'])
    d = out['trained_with_icl_demos']['aggregate']
    print('ICL trained fixed target/random:', d['fixed/target']['mean_twin_accuracy'], d['fixed/random']['mean_twin_accuracy'])
    print('ICL trained remap target/random:', d['remap/target']['mean_twin_accuracy'], d['remap/random']['mean_twin_accuracy'])
    for m, v in out['retained_cross_depth_geometry'].items():
        print(m, {a: round(v['arms'][a]['bidirectional_top1_concept_match'], 3) for a in v['arms']})

if __name__ == '__main__':
    main()
