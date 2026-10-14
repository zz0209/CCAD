import argparse
import json
import math
import os
import statistics
from pathlib import Path

DESCRIPTION = ('Summarise P2 swap results per checkpoint (success rate with Wilson 95% intervals per method, on all '
               'correctly answered two-hop prompts, on held-out and training relation pairs, and on prompts whose '
               'swap target one-hop control is also correct; latent-bridge lens readout ranks) and apply the frozen '
               'kill criteria.')
METHODS = ['jlens', 'logit', 'tuned', 'dim', 'dim_outside_j', 'random']
FINAL_OLMO = ['olmo1b_main', 'olmo211247b']
MIN_N_K2 = 20


def write_json(path, obj):
    tmp = f'{path}.tmp.{os.getpid()}'
    Path(tmp).write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def rates(rows):
    out = {}
    for m in METHODS:
        vals = [r['swap'][m]['success'] for r in rows if r['swap'][m] is not None]
        out[m] = dict(k=sum(vals), n=len(vals), rate=sum(vals) / len(vals) if vals else None,
                      ci95=wilson(sum(vals), len(vals)))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='label=SWAP.json')
    parser.add_argument('--tuned_meta', nargs='*', default=[], help='label=TUNED.pt.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tuned_meta = {s.split('=')[0]: json.loads(Path(s.split('=')[1]).read_text()) for s in args.tuned_meta}
    per = {}
    for spec in args.runs:
        label, path = spec.split('=')
        d = json.loads(Path(path).read_text())
        rows = d['rows']
        ok = [r for r in rows if r['correct']]
        pairs = sorted({r['pair'] for r in rows})
        s = dict(revision=d['revision'], band=d['band'], n_items=len(rows), n_correct=len(ok),
                 n_one_hop_correct=sum(r['one_hop_correct'] for r in rows),
                 correct_by_pair={p: sum(r['correct'] for r in rows if r['pair'] == p) for p in pairs},
                 swap_all=rates(ok), swap_held_out=rates([r for r in ok if r['held_out']]),
                 swap_training_pairs=rates([r for r in ok if not r['held_out']]),
                 swap_target_known=rates([r for r in ok if r['target_one_hop_correct']]),
                 swap_by_pair={p: rates([r for r in ok if r['pair'] == p]) for p in pairs},
                 bridge_top5_by_lens={k: sum(r['bridge_rank_min'][k] < 5 for r in rows) / len(rows)
                                      for k in ('jlens', 'logit', 'tuned')},
                 bridge_top5_by_lens_correct={k: (sum(r['bridge_rank_min'][k] < 5 for r in ok) / len(ok)
                                                  if ok else None) for k in ('jlens', 'logit', 'tuned')},
                 jspace_energy_frac_median=statistics.median(
                     v for e in d['concept']['jspace_energy_frac'].values() for v in e.values()))
        both = [r for r in ok if r['swap']['jlens'] is not None and r['swap']['tuned'] is not None]
        s['jlens_vs_tuned_discordant'] = dict(
            jlens_only=sum(r['swap']['jlens']['success'] and not r['swap']['tuned']['success'] for r in both),
            tuned_only=sum(r['swap']['tuned']['success'] and not r['swap']['jlens']['success'] for r in both))
        if label in tuned_meta:
            s['tuned_heldout_kl'] = tuned_meta[label]['heldout_kl']
        per[label] = s
    k1_parts = {}
    for label in FINAL_OLMO:
        if label in per:
            sw = per[label]['swap_all']
            j, lg = sw['jlens']['rate'], sw['logit']['rate']
            # No correctly answered prompt means no evidence of a working J-swap; it counts as failing both parts.
            k1_parts[label] = dict(jlens=j, logit=lg, n=sw['jlens']['n'],
                                   below_30=True if j is None else j < 0.30,
                                   below_2x_logit=True if j is None else j < 2 * lg)
    k1 = len(k1_parts) == len(FINAL_OLMO) and all(v['below_30'] and v['below_2x_logit'] for v in k1_parts.values())
    gaps = {lb: dict(gap=(s['swap_all']['jlens']['rate'] - s['swap_all']['tuned']['rate'])
                     if s['swap_all']['jlens']['n'] else None, n=s['swap_all']['jlens']['n'])
            for lb, s in per.items()}
    k2_beats = [lb for lb, g in gaps.items() if g['gap'] is not None and g['n'] >= MIN_N_K2 and g['gap'] >= 0.10]
    k2_beats_any_n = [lb for lb, g in gaps.items() if g['gap'] is not None and g['gap'] >= 0.10]
    k2 = len(k2_beats) == 0
    # Kill 1 needs every final OLMo checkpoint to be weak, so one present checkpoint that is not weak rules it out;
    # Kill 2 needs no checkpoint to beat the tuned lens, so one that does rules it out. Missing checkpoints cannot
    # change a ruled-out criterion.
    k1_ruled_out = any(not (v['below_30'] and v['below_2x_logit']) for v in k1_parts.values())
    k2_ruled_out = len(k2_beats) > 0
    complete = all(lb in per for lb in FINAL_OLMO)
    if k1_ruled_out and k2_ruled_out:
        verdict = 'PASS'
    elif complete:
        verdict = 'KILL' if (k1 or k2) else 'PASS'
    else:
        verdict = 'UNDETERMINED'
    decision = dict(kill_1_final_olmo_weak=k1, kill_1_parts=k1_parts, kill_1_ruled_out=k1_ruled_out,
                    kill_2_never_beats_tuned=k2, kill_2_ruled_out=k2_ruled_out, jlens_minus_tuned=gaps,
                    k2_min_eligible=MIN_N_K2, checkpoints_beating_tuned=k2_beats,
                    checkpoints_beating_tuned_any_n=k2_beats_any_n, final_olmo_checkpoints_present=complete,
                    checkpoints_evaluated=sorted(per), verdict=verdict)
    write_json(args.out, dict(per_checkpoint=per, decision=decision))
    for lb, s in per.items():
        sw = s['swap_all']
        print(f"{lb:20s} correct {s['n_correct']:3d}/{s['n_items']}  " + '  '.join(
            f"{m} {sw[m]['k']}/{sw[m]['n']}" for m in METHODS))
    print(json.dumps(decision, indent=1, default=float))


if __name__ == '__main__':
    main()
