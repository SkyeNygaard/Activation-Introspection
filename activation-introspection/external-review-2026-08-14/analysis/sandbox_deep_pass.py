#!/usr/bin/env python3
"""Deep offline audit/secondary analyses for Activation-Introspection.

Runs only on checked-in raw JSONL and saved activation tensors.  No repository
analyzers, transformers, PEFT, model weights, or network access are used.

All analyses not explicitly stated as preregistered in the source repo are
post-hoc/exploratory and should be labeled that way.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path('/mnt/data/activation_introspection/work/repo')
RES = ROOT / 'results'
OUT = Path('/mnt/data/activation_introspection')


def loadjl(name: str):
    with (RES / name).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def exact_binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    return float(stats.binomtest(k, n, 0.5, alternative='two-sided').pvalue)


def twin_groups(rows, key_fields, sign_field):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(r)
    for g in groups.values():
        assert len(g) == 2, (len(g), g[:1])
        assert {int(x[sign_field]) for x in g} == {-1, 1}
    return groups


def heldout_semantic_paired():
    rows = loadjl('heldout_semantic_v1_raw.jsonl')
    arms = ['same_exemplar', 'heldout_semantic', 'heldout_scrambled', 'heldout_random', 'query_only']
    out = {'arms': {}}
    arm_cells = {}
    for arm in arms:
        rs = [r for r in rows if r['arm'] == arm]
        groups = twin_groups(rs, ['category_pair', 'draw', 'carrier_sha', 'cell_base'], 'query_sign')
        cells = {}
        by_pair = defaultdict(lambda: {'n': 0, 'success': 0, 'flips': 0, 'wrong_flips': 0})
        for k, g in groups.items():
            preds = {x['model_predicted'] for x in g}
            flip = len(preds) == 2
            success = all(bool(x['model_correct']) for x in g)
            wrong_flip = flip and not success
            cells[k] = {'success': success, 'flip': flip, 'wrong_flip': wrong_flip}
            cp = k[0]
            by_pair[cp]['n'] += 1
            by_pair[cp]['success'] += int(success)
            by_pair[cp]['flips'] += int(flip)
            by_pair[cp]['wrong_flips'] += int(wrong_flip)
        out['arms'][arm] = {
            'n_twins': len(groups),
            'successes': sum(v['success'] for v in cells.values()),
            'twin_accuracy': mean(v['success'] for v in cells.values()),
            'flips': sum(v['flip'] for v in cells.values()),
            'wrong_direction_flips': sum(v['wrong_flip'] for v in cells.values()),
            'by_category_pair': {
                cp: {**d, 'accuracy': d['success'] / d['n']} for cp, d in sorted(by_pair.items())
            },
        }
        arm_cells[arm] = cells

    def paired(a, b, field):
        assert arm_cells[a].keys() == arm_cells[b].keys()
        a_only = b_only = both = neither = 0
        for k in arm_cells[a]:
            x, y = bool(arm_cells[a][k][field]), bool(arm_cells[b][k][field])
            if x and y: both += 1
            elif x: a_only += 1
            elif y: b_only += 1
            else: neither += 1
        n_disc = a_only + b_only
        return {
            'a_only': a_only, 'b_only': b_only, 'both': both, 'neither': neither,
            'discordant_n': n_disc,
            'exact_mcnemar_binomial_p': exact_binom_two_sided(a_only, n_disc),
        }

    out['semantic_vs_scrambled'] = {
        'success': paired('heldout_semantic', 'heldout_scrambled', 'success'),
        'prediction_flip': paired('heldout_semantic', 'heldout_scrambled', 'flip'),
        'wrong_direction_flip': paired('heldout_semantic', 'heldout_scrambled', 'wrong_flip'),
    }
    return out


def report_zero_demo_orientation():
    out = {'seeds': {}, 'aggregate': {}}
    slopes_by_seed = {}
    for seed in range(4):
        rows = loadjl(f'report_training_v3_seed{seed}_raw.jsonl')
        seed_out = {}
        seed_slopes = {}
        for arm in ['base', 'trained']:
            for cond in ['target', 'random', 'shuffled']:
                rs = [r for r in rows if r['arm'] == arm and r['condition'] == cond]
                groups = twin_groups(rs, ['concept', 'carrier_sha256'], 'sign')
                orig = inv = flip = neither = 0
                slopes = defaultdict(list)
                for (_, _), g in groups.items():
                    gp = next(x for x in g if x['sign'] == 1)
                    gm = next(x for x in g if x['sign'] == -1)
                    # signed_margin is correctness-oriented under the arbitrary +->Q/- ->K convention.
                    # Recover raw Q-K margin by multiplying by sign.
                    rawp = float(gp['signed_margin'])
                    rawm = -float(gm['signed_margin'])
                    slope = (rawp - rawm) / 2.0
                    slopes[gp['concept']].append(slope)
                    o = all(bool(x['correct']) for x in g)
                    i = all(not bool(x['correct']) for x in g)
                    f = gp['predicted_label'] != gm['predicted_label']
                    orig += int(o); inv += int(i); flip += int(f); neither += int(not f)
                concept_slopes = {c: mean(v) for c, v in slopes.items()}
                seed_out[f'{arm}/{cond}'] = {
                    'n_twins': len(groups),
                    'original_orientation_twin_accuracy': orig / len(groups),
                    'inverted_orientation_twin_accuracy': inv / len(groups),
                    'orientation_invariant_flip_rate': flip / len(groups),
                    'no_flip_rate': neither / len(groups),
                    'mean_abs_response_slope': mean(abs(v) for v in concept_slopes.values()),
                    'concept_response_slopes': concept_slopes,
                }
                if arm == 'trained':
                    seed_slopes[cond] = concept_slopes
        # clean/no-edit label preference
        for arm in ['base', 'trained']:
            rs = [r for r in rows if r['arm'] == arm and r['condition'] == 'clean']
            seed_out[f'{arm}/clean'] = {
                'n': len(rs),
                'predicted_labels': dict(Counter(r['predicted_label'] for r in rs)),
                'mean_abs_qk_margin': mean(abs(float(r['signed_margin'])) for r in rs),
                'mean_label_mass': mean(float(r['label_mass']) for r in rs),
            }
        out['seeds'][str(seed)] = seed_out
        slopes_by_seed[str(seed)] = seed_slopes

    # aggregate rates and orientation symmetry over trained controls
    for cond in ['target', 'random', 'shuffled']:
        vals = [out['seeds'][str(s)][f'trained/{cond}'] for s in range(4)]
        out['aggregate'][f'trained/{cond}'] = {
            'mean_original_twin_accuracy': mean(v['original_orientation_twin_accuracy'] for v in vals),
            'mean_inverted_twin_accuracy': mean(v['inverted_orientation_twin_accuracy'] for v in vals),
            'mean_flip_rate': mean(v['orientation_invariant_flip_rate'] for v in vals),
            'mean_abs_response_slope': mean(v['mean_abs_response_slope'] for v in vals),
        }
    for cond in ['random', 'shuffled']:
        orig = inv = 0
        for s in range(4):
            v = out['seeds'][str(s)][f'trained/{cond}']
            n = v['n_twins']
            orig += round(v['original_orientation_twin_accuracy'] * n)
            inv += round(v['inverted_orientation_twin_accuracy'] * n)
        out['aggregate'][f'trained/{cond}']['orientation_symmetry_counts'] = {'original': orig, 'inverted': inv}
        out['aggregate'][f'trained/{cond}']['orientation_symmetry_exact_p'] = exact_binom_two_sided(orig, orig + inv)

    # common training-induced response geometry across arbitrary axes
    concepts = sorted(slopes_by_seed['0']['random'])
    labels = [('random', c) for c in concepts] + [('shuffled', c) for c in concepts]
    trained = np.array([[slopes_by_seed[str(s)][cond][c] for cond, c in labels] for s in range(4)], dtype=float)
    # base slopes from seed0 (base model identical across seeds; direction banks identical)
    base = []
    for cond, c in labels:
        base.append(out['seeds']['0'][f'base/{cond}']['concept_response_slopes'][c])
    base = np.asarray(base)
    induced = trained - base[None, :]
    cors = []
    pairwise = {}
    for i, j in combinations(range(4), 2):
        r = float(stats.pearsonr(induced[i], induced[j]).statistic)
        pairwise[f'{i}-{j}'] = r; cors.append(r)
    sv = np.linalg.svd(induced, full_matrices=False, compute_uv=False)
    rank1_energy = float(sv[0] ** 2 / np.sum(sv ** 2))

    rng = np.random.default_rng(20260813)
    observed = mean(cors)
    exceed = 0
    nperm = 100_000
    for _ in range(nperm):
        perm = np.stack([row[rng.permutation(row.size)] for row in induced])
        cs = []
        for i, j in combinations(range(4), 2):
            cs.append(float(np.corrcoef(perm[i], perm[j])[0, 1]))
        if mean(cs) >= observed - 1e-15:
            exceed += 1
    out['common_detector_geometry'] = {
        'axis_labels': [f'{a}:{c}' for a, c in labels],
        'pairwise_training_induced_pearson': pairwise,
        'mean_pairwise_pearson': observed,
        'rank1_energy_fraction': rank1_energy,
        'permutation_trials': nperm,
        'permutation_p': (exceed + 1) / (nperm + 1),
    }
    # Deflationary check: does common trained profile just inherit base response?
    for cond in ['target', 'random', 'shuffled']:
        cpts = sorted(slopes_by_seed['0'][cond])
        b = np.array([out['seeds']['0'][f'base/{cond}']['concept_response_slopes'][c] for c in cpts])
        tm = np.array([mean(slopes_by_seed[str(s)][cond][c] for s in range(4)) for c in cpts])
        pr = stats.pearsonr(b, tm); sr = stats.spearmanr(b, tm)
        out['common_detector_geometry'][f'base_vs_trained_mean_{cond}'] = {
            'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
            'spearman_rho': float(sr.statistic), 'spearman_p': float(sr.pvalue),
        }
    return out


def remap_demo_decoder():
    out = {'seeds': {}, 'confidence': {}, 'pooled_direction_accuracy': {}}
    all_rows = {}
    for seed in range(3):
        rows = loadjl(f'remap_training_v2_seed{seed}_raw.jsonl')
        all_rows[seed] = rows
        so = {}
        for arm in ['base', 'fixed', 'remap']:
            for cond in ['target', 'random']:
                rs = [r for r in rows if r['arm'] == arm and r['condition'] == cond and float(r['strength']) == 0.5]
                groups = twin_groups(rs, ['concept', 'carrier_sha256', 'order_key', 'positive_label'], 'query_sign')
                # mapping flip pairs: same order/query sign with both positive_label conventions correct
                mf = defaultdict(list)
                for r in rs:
                    mf[(r['concept'], r['carrier_sha256'], r['order_key'], r['query_sign'])].append(r)
                assert all(len(g) == 2 and {x['positive_label'] for x in g} == {'Q', 'K'} for g in mf.values())
                so[f'{arm}/{cond}'] = {
                    'row_accuracy': mean(bool(r['correct']) for r in rs),
                    'twin_accuracy': mean(all(bool(x['correct']) for x in g) for g in groups.values()),
                    'mapping_flip_accuracy': mean(all(bool(x['correct']) for x in g) for g in mf.values()),
                }
        out['seeds'][str(seed)] = so

        # Confidence discriminates correct random-codebook decoding.
        for arm in ['base', 'fixed', 'remap']:
            rs = [r for r in rows if r['arm'] == arm and r['condition'] == 'random' and float(r['strength']) == 0.5]
            y = np.array([int(bool(r['correct'])) for r in rs])
            score = np.array([abs(float(r['signed_margin'])) for r in rs])
            auc = float(roc_auc_score(y, score)) if len(set(y)) == 2 else float('nan')
            order = np.argsort(-score)
            top = {}
            for frac in [0.2, 0.5]:
                n = max(1, math.ceil(len(rs) * frac))
                top[str(frac)] = float(y[order[:n]].mean())
            out['confidence'][f'seed{seed}/{arm}/random'] = {
                'n_rows': len(rs), 'accuracy': float(y.mean()), 'abs_margin_correctness_auroc': auc,
                'top_confidence_accuracy': top,
            }

    # pooled accuracy by random direction/concept under demonstrated codebook
    concepts = sorted({r['concept'] for r in all_rows[0] if r['condition'] == 'random'})
    for arm in ['fixed', 'remap']:
        out['pooled_direction_accuracy'][arm] = {}
        for c in concepts:
            rs = [r for s in range(3) for r in all_rows[s]
                  if r['arm'] == arm and r['condition'] == 'random' and float(r['strength']) == 0.5 and r['concept'] == c]
            out['pooled_direction_accuracy'][arm][c] = mean(bool(r['correct']) for r in rs)
    return out


def cross_paradigm_binding(zero_demo, demo):
    # Same fixed control seed/bank: compare zero-demo trained random response magnitude
    # with demonstrated arbitrary-axis decoding difficulty. This is a diagnostic join,
    # not a causal factorial because adapters differ.
    concepts = sorted(zero_demo['seeds']['0']['trained/random']['concept_response_slopes'])
    z_abs = []
    z_flip = []
    demo_fixed = []
    demo_remap = []
    # Aggregate zero-demo by concept: mean absolute slope and empirical seed flip fraction.
    for c in concepts:
        slopes = [zero_demo['seeds'][str(s)]['trained/random']['concept_response_slopes'][c] for s in range(4)]
        z_abs.append(mean(abs(x) for x in slopes))
        # Approximate concept-level flip by checking whether slope overcomes intercept would require carrier-level;
        # use orientation-stable response magnitude only for formal correlation.
        demo_fixed.append(demo['pooled_direction_accuracy']['fixed'][c])
        demo_remap.append(demo['pooled_direction_accuracy']['remap'][c])
    out = {'concepts': concepts, 'zero_demo_mean_abs_slope': z_abs,
           'demo_fixed_accuracy': demo_fixed, 'demo_remap_accuracy': demo_remap}
    for name, y in [('fixed', demo_fixed), ('remap', demo_remap)]:
        pr = stats.pearsonr(z_abs, y); sr = stats.spearmanr(z_abs, y)
        out[f'abs_slope_vs_{name}'] = {
            'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
            'spearman_rho': float(sr.statistic), 'spearman_p': float(sr.pvalue),
        }
    # The banana axis is a useful falsifier: zero-demo trained slopes all negative,
    # yet both mapping conventions are decoded well after demonstrations.
    c = 'banana'
    out['banana_case'] = {
        'zero_demo_slopes_by_seed': [zero_demo['seeds'][str(s)]['trained/random']['concept_response_slopes'][c] for s in range(4)],
        'demo_fixed_accuracy': demo['pooled_direction_accuracy']['fixed'][c],
        'demo_remap_accuracy': demo['pooled_direction_accuracy']['remap'][c],
    }
    return out


def protocol_null_audit():
    hits = {}
    for name in ['report_training_protocol_v1.json', 'report_training_protocol_v2.json', 'report_training_protocol_v3.json']:
        p = RES / name
        text = p.read_text()
        # Preserve exact phrase snippets around 0.500 pair identity.
        idx = text.find('0.500')
        hits[name] = text[max(0, idx - 220): idx + 260] if idx >= 0 else None
    source = (ROOT / 'scripts' / 'run_report_training.py').read_text()
    return {'protocol_snippets': hits, 'current_runner_contains_0.500_pair_identity': 'scores exactly 0.500 on pairs' in source}


def clustering_replication_check():
    orig = json.loads((RES / 'clustering_prediction_v1_summary.json').read_text())
    rep = json.loads((RES / 'clustering_replication_v1_summary.json').read_text())
    # Keep small, source-independent facts from summaries.
    return {'original_summary': orig, 'replication_summary': rep}


def control_gram_fingerprint(path: Path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    acts, index = blob['acts'].float(), blob['index']
    pos = {(i['arm'], i['inject_layer'], i['carrier_id'], i['concept']): j for j, i in enumerate(index)}
    depths = sorted({i['inject_layer'] for i in index})
    carriers = sorted({i['carrier_id'] for i in index})
    concepts = sorted({i['concept'] for i in index})
    rows = []
    for d in depths:
        # activation at injection layer, average Gram over carriers
        gt = []; gs = []; random_norm = []; target_norm = []
        for car in carriers:
            X = {}
            for arm in ['target', 'random', 'shuffled']:
                mat = torch.stack([acts[pos[(arm, d, car, c)], d] for c in concepts])
                mat = mat - mat.mean(0, keepdim=True)
                X[arm] = mat
            Gt = (X['target'] @ X['target'].T).numpy()
            Gs = (X['shuffled'] @ X['shuffled'].T).numpy()
            tri = np.triu_indices(len(concepts), 1)
            gt.append(Gt[tri]); gs.append(Gs[tri])
            random_norm.append(float(X['random'].norm(dim=1).mean()))
            target_norm.append(float(X['target'].norm(dim=1).mean()))
        a = np.concatenate(gt); b = np.concatenate(gs)
        rows.append({
            'inject_layer': d,
            'target_vs_shuffled_gram_pearson': float(stats.pearsonr(a, b).statistic),
            'random_centered_norm': mean(random_norm),
            'target_centered_norm': mean(target_norm),
            'random_to_target_centered_norm_ratio': mean(random_norm) / mean(target_norm),
        })
    return rows


def retained_control_audit():
    files = {
        'qwen05b_repaired': RES / 'retained_test_qwen05b_v2_raw.acts.pt',
        'qwen15b_old': RES / 'retained_test_qwen15b_raw.acts.pt',
        'qwen3b_old': RES / 'retained_test_qwen3b_raw.acts.pt',
    }
    return {k: control_gram_fingerprint(v) for k, v in files.items()}


def retained_scale_repair_crossbank():
    old_names = {'qwen15b': 'retained_test_qwen15b_raw.jsonl', 'qwen3b': 'retained_test_qwen3b_raw.jsonl'}
    dev_names = {'qwen15b': 'retained_dev_qwen15b_raw.jsonl', 'qwen3b': 'retained_dev_qwen3b_raw.jsonl'}
    comparison = []
    for model in old_names:
        old = loadjl(old_names[model]); dev = loadjl(dev_names[model])
        old_depths = sorted({r['inject_layer'] for r in old})
        dev_depths = sorted({r['inject_layer'] for r in dev})
        # dev depths are subset of test depths by design
        for d in dev_depths:
            for arm in ['target', 'random', 'shuffled']:
                ors = [r for r in old if r['inject_layer'] == d and r['arm'] == arm and float(r['strength']) == 1.0]
                drs = [r for r in dev if r['inject_layer'] == d and r['arm'] == arm and float(r['strength']) == 1.0]
                comparison.append({
                    'model': model, 'layer': d, 'arm': arm,
                    'old_test_accuracy': mean(bool(r['correct']) for r in ors),
                    'repaired_dev_accuracy': mean(bool(r['correct']) for r in drs),
                    'old_n': len(ors), 'dev_n': len(drs),
                })
    pairs = [x for x in comparison if x['arm'] == 'target']
    x = np.array([p['old_test_accuracy'] for p in pairs]); y = np.array([p['repaired_dev_accuracy'] for p in pairs])
    pr = stats.pearsonr(x, y); sr = stats.spearmanr(x, y)
    return {'cells': comparison, 'target_crossbank_shape': {
        'n_cells': len(pairs), 'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
        'spearman_rho': float(sr.statistic), 'spearman_p': float(sr.pvalue)}}


def centered_vectors(acts, index, read_layer, arms=None):
    arms = arms or sorted({i['arm'] for i in index})
    depths = sorted({i['inject_layer'] for i in index})
    carriers = sorted({i['carrier_id'] for i in index})
    concepts = sorted({i['concept'] for i in index})
    pos = {(i['arm'], i['inject_layer'], i['carrier_id'], i['concept']): j for j, i in enumerate(index)}
    vec = {}
    for arm in arms:
        for d in depths:
            for car in carriers:
                mat = torch.stack([acts[pos[(arm, d, car, c)], read_layer].float() for c in concepts])
                cen = mat.mean(0)
                for ci, c in enumerate(concepts):
                    vec[(arm, d, car, c)] = mat[ci] - cen
    return vec, depths, carriers, concepts


def cosine_predict(x, centroids, labels):
    X = x / (x.norm() + 1e-12)
    C = torch.stack([centroids[l] for l in labels])
    C = C / (C.norm(dim=1, keepdim=True) + 1e-12)
    return labels[int((C @ X).argmax())]


def cross_depth_carrier_concept_accuracy(path: Path, strength_filter=None):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    acts, index = blob['acts'], blob['index']
    if strength_filter is not None:
        keep = [j for j, i in enumerate(index) if float(i.get('strength', 1.0)) == float(strength_filter)]
        acts = acts[keep]
        index = [index[j] for j in keep]
    read_layer = acts.shape[1] - 1
    vec, depths, carriers, concepts = centered_vectors(acts, index, read_layer, ['target', 'random', 'shuffled'])
    out = {}
    for arm in ['target', 'random', 'shuffled']:
        correct = total = 0
        # Leave one injection depth and one carrier out jointly; concept seen elsewhere.
        for td in depths:
            for tcarr in carriers:
                train = [(d, car) for d in depths for car in carriers if d != td and car != tcarr]
                if not train:
                    continue
                cents = {c: torch.stack([vec[(arm, d, car, c)] for d, car in train]).mean(0) for c in concepts}
                for c in concepts:
                    pred = cosine_predict(vec[(arm, td, tcarr, c)], cents, concepts)
                    correct += int(pred == c); total += 1
        out[arm] = {'accuracy': correct / total, 'correct': correct, 'n': total, 'chance': 1 / len(concepts)}
    return {'shape': list(acts.shape), 'depths': depths, 'carriers': carriers, 'concepts': concepts,
            'final_read_layer': read_layer, 'arms': out}


def depth_provenance_leave_concept_carrier(path: Path, strength_filter=None):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    acts, index = blob['acts'], blob['index']
    if strength_filter is not None:
        keep = [j for j, i in enumerate(index) if float(i.get('strength', 1.0)) == float(strength_filter)]
        acts = acts[keep]; index = [index[j] for j in keep]
    read_layer = acts.shape[1] - 1
    depths = sorted({i['inject_layer'] for i in index}); carriers = sorted({i['carrier_id'] for i in index}); concepts = sorted({i['concept'] for i in index})
    pos = {(i['arm'], i['inject_layer'], i['carrier_id'], i['concept']): j for j, i in enumerate(index)}
    out = {}
    for arm in ['target', 'random', 'shuffled']:
        correct = total = 0
        for hc in concepts:
            for hcar in carriers:
                train_cs = [c for c in concepts if c != hc]
                train_car = [car for car in carriers if car != hcar]
                cents = {}
                for d in depths:
                    xs = [acts[pos[(arm, d, car, c)], read_layer].float() for c in train_cs for car in train_car]
                    cents[d] = torch.stack(xs).mean(0)
                for d in depths:
                    x = acts[pos[(arm, d, hcar, hc)], read_layer].float()
                    pred = cosine_predict(x, cents, depths)
                    correct += int(pred == d); total += 1
        out[arm] = {'accuracy': correct / total, 'correct': correct, 'n': total, 'chance': 1 / len(depths)}
    return {'depths': depths, 'concepts': concepts, 'carriers': carriers, 'arms': out}


def variance_main_effects(path: Path, strength_filter=None):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    acts, index = blob['acts'], blob['index']
    if strength_filter is not None:
        keep = [j for j, i in enumerate(index) if float(i.get('strength', 1.0)) == float(strength_filter)]
        acts = acts[keep]; index = [index[j] for j in keep]
    Xall = acts[:, -1].float()
    out = {}
    for arm in ['target', 'random', 'shuffled']:
        ids = [j for j, i in enumerate(index) if i['arm'] == arm]
        X = Xall[ids]
        meta = [index[j] for j in ids]
        mu = X.mean(0); total_ss = float(((X - mu) ** 2).sum())
        arm_out = {}
        for field in ['concept', 'inject_layer', 'carrier_id']:
            vals = sorted({m[field] for m in meta}, key=str)
            ss = 0.0
            for v in vals:
                js = [j for j, m in enumerate(meta) if m[field] == v]
                gm = X[js].mean(0)
                ss += len(js) * float(((gm - mu) ** 2).sum())
            arm_out[field] = ss / total_ss
        out[arm] = arm_out
    return out


def retained_representation_geometry():
    files = {
        'qwen05b_test_repaired': (RES / 'retained_test_qwen05b_v2_raw.acts.pt', None),
        'qwen15b_dev_repaired': (RES / 'retained_dev_qwen15b_raw.acts.pt', 1.0),
        'qwen3b_dev_repaired': (RES / 'retained_dev_qwen3b_raw.acts.pt', 1.0),
    }
    out = {}
    for name, (p, strength) in files.items():
        out[name] = {
            'cross_depth_cross_carrier_concept': cross_depth_carrier_concept_accuracy(p, strength),
            'depth_provenance_holdout_concept_carrier': depth_provenance_leave_concept_carrier(p, strength),
            'variance_main_effect_fraction': variance_main_effects(p, strength),
        }
    return out


def retained_codebook_inversion():
    out = {}
    for model, name in [('qwen05b', 'retained_test_qwen05b_v2_raw.jsonl'), ('qwen15b', 'retained_test_qwen15b_raw.jsonl'), ('qwen3b', 'retained_test_qwen3b_raw.jsonl')]:
        rows = loadjl(name)
        cells = []
        for arm in ['target', 'random', 'shuffled']:
            for d in sorted({r['inject_layer'] for r in rows}):
                rs = [r for r in rows if r['arm'] == arm and r['inject_layer'] == d and float(r['strength']) == 1.0]
                if not rs: continue
                # Each codebook maps each concept to a unique label. Invert predicted label.
                by_cb = defaultdict(list)
                for r in rs: by_cb[(r['carrier_id'], r['codebook_id'])].append(r)
                true_pred = []
                for (_, _), g in by_cb.items():
                    inv = {r['correct_label']: r['concept'] for r in g}
                    for r in g:
                        true_pred.append((r['concept'], inv.get(r['pred_label'], '<other>')))
                concepts = sorted({t for t, _ in true_pred})
                labels = concepts + ['<other>']
                ti = {c:i for i,c in enumerate(concepts)}; pi={c:i for i,c in enumerate(labels)}
                table = np.zeros((len(concepts), len(labels)), dtype=int)
                for t,p in true_pred: table[ti[t], pi[p]] += 1
                # Mutual information in bits.
                N=table.sum(); pr=table.sum(1)/N; pc=table.sum(0)/N
                mi=0.0
                for i in range(table.shape[0]):
                    for j in range(table.shape[1]):
                        if table[i,j]:
                            q=table[i,j]/N; mi += q*math.log2(q/(pr[i]*pc[j]))
                acc=sum(t==p for t,p in true_pred)/len(true_pred)
                cells.append({'arm':arm,'layer':d,'inverted_concept_accuracy':acc,'mutual_information_bits':mi,'n':len(true_pred)})
        out[model]=cells
    return out


def cot_contamination():
    rows = [r for r in loadjl('heldout_cot_v1_raw.jsonl') if r['readout'] != 'forced' and r.get('generation')]
    qk = re.compile(r'\b(query|queries|key|keys)\b', re.I)
    pattern = re.compile(r'\b(alternat\w*|pattern\w*|sequence|repeat\w*|cycle\w*)\b', re.I)
    return {
        'n_generated_rows': len(rows),
        'qk_semantic_word_rows': sum(bool(qk.search(r['generation'])) for r in rows),
        'qk_semantic_word_rate': mean(bool(qk.search(r['generation'])) for r in rows),
        'visible_pattern_language_rows': sum(bool(pattern.search(r['generation'])) for r in rows),
        'visible_pattern_language_rate': mean(bool(pattern.search(r['generation'])) for r in rows),
        'examples': [r['generation'][:500] for r in rows[:3]],
    }


def main():
    print('Running deep offline analyses...')
    held = heldout_semantic_paired(); print('heldout semantic paired done')
    zero = report_zero_demo_orientation(); print('zero-demo orientation/common detector done')
    demo = remap_demo_decoder(); print('demo decoder/confidence done')
    cross = cross_paradigm_binding(zero, demo); print('cross-paradigm binding done')
    prot = protocol_null_audit(); print('protocol null audit done')
    ctrl = retained_control_audit(); print('retained control fingerprint done')
    scale = retained_scale_repair_crossbank(); print('scale repair join done')
    geom = retained_representation_geometry(); print('retained representation geometry done')
    inv = retained_codebook_inversion(); print('codebook inversion done')
    cot = cot_contamination(); print('CoT contamination done')
    cluster = clustering_replication_check(); print('clustering handoff check data loaded')
    result = {
        'scope': {
            'type': 'post-hoc offline secondary analysis',
            'data': 'checked-in raw JSONL and saved activation tensors only',
            'no_model_forward_pass': True,
            'no_repo_analyzers_used_for_core_new_metrics': True,
            'warning': 'Exploratory findings require prospective confirmation before headline claims.',
        },
        'heldout_semantic_paired': held,
        'zero_demo_orientation_and_common_detector': zero,
        'demonstrated_axis_decoder': demo,
        'cross_paradigm_binding': cross,
        'protocol_null_audit': prot,
        'retained_control_fingerprint': ctrl,
        'retained_scale_repair_crossbank': scale,
        'retained_representation_geometry': geom,
        'retained_codebook_inversion': inv,
        'cot_generation_contamination': cot,
        'clustering_replication_summaries': cluster,
    }
    dest = OUT / 'sandbox_deep_pass_results.json'
    dest.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print('WROTE', dest)
    # concise machine log
    print('semantic vs scrambled paired success:', held['semantic_vs_scrambled']['success'])
    print('zero-demo trained flips:', {c: zero['aggregate'][f'trained/{c}']['mean_flip_rate'] for c in ['target','random','shuffled']})
    print('common detector:', zero['common_detector_geometry'])
    print('cross paradigm:', {k:v for k,v in cross.items() if 'vs_' in k or k=='banana_case'})
    print('scale target crossbank:', scale['target_crossbank_shape'])
    print('concept geometry:', {m:{a:v['cross_depth_cross_carrier_concept']['arms'][a]['accuracy'] for a in ['target','random','shuffled']} for m,v in geom.items()})
    print('depth provenance:', {m:{a:v['depth_provenance_holdout_concept_carrier']['arms'][a]['accuracy'] for a in ['target','random','shuffled']} for m,v in geom.items()})
    print('cot:', {k:v for k,v in cot.items() if k!='examples'})

if __name__ == '__main__':
    main()
