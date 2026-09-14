"""Summarize genuine hybrid answers from digit-component source interventions."""
import argparse, collections, hashlib, json
from pathlib import Path
from datetime import datetime, timezone


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    assert json.loads((a.run/"status.json").read_text())["status"]=="PASS"
    raw=a.run/"metrics.raw.jsonl";rows=[json.loads(s) for s in raw.read_text().splitlines()]
    config=json.loads((a.run/"config.resolved.json").read_text())
    panel=json.loads((a.run/"panel.json").read_text());cells=[];groups=collections.defaultdict(list)
    for r in rows:
        if r["kind"]=="source_patch":groups[r["method"],r["seed"],r["mode"],r["operation"],r["task"]].append(r)
    for key,rr in groups.items():
        for subset in ["all","both_base_answers_correct"]:
            selected=[r for r in rr if subset=="all" or (r["base_correct"] and r["donor_correct"])]
            if not selected:continue
            counts=collections.Counter()
            for r in selected:
                pair=panel["pairs"][r["row_id"]];answer=r["answer"]
                category=next((label for label in ["base","unit","tens","donor"] if answer==pair[label+"_answer"]),"other")
                counts[category]+=1
            cells.append(dict(zip(["method","seed","members","operation","task"],key),subset=subset,n=len(selected),
                              exact_hybrid=sum(r["exact_hybrid"] for r in selected)/len(selected),
                              target_digit_success=sum(r["target_digit_success"] for r in selected)/len(selected),
                              preserve_digit_success=sum(r["preserve_digit_success"] for r in selected)/len(selected),
                              mean_edit_norm=sum(r["edit_norm"] for r in selected)/len(selected),
                              outcomes={s:counts[s]/len(selected) for s in ["base","unit","tens","donor","other"]}))
    base=collections.defaultdict(list)
    for r in rows:
        if r["kind"]=="base":base[r["task"],r["split"]].append(r)
    quality=[dict(task=k[0],split=k[1],n=len(rr),exact_accuracy=sum(r["correct"] for r in rr)/len(rr)) for k,rr in base.items()]
    summary=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run.resolve()),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),rows=len(rows),base_quality=quality,cells=cells,
                 scope=config.get("scope","Same operand pairs are reused across SAE seeds and prompt forms; these are not independent per-record trials."),
                 denominator="Full denominator is primary; both-correct subset is displayed separately. Four distinct hybrid outcomes are computed from exact free generation.")
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(base_quality=quality,rows=len(rows),cells=len(cells))))
    aggregate=collections.defaultdict(list)
    for c in cells:
        if c["subset"]=="all":aggregate[c["method"],c["members"],c["operation"],c["task"]].append(c)
    for key,cc in aggregate.items():
        print(json.dumps(dict(method=key[0],members=key[1],operation=key[2],task=key[3],seeds=len(cc),
                              hybrid=sum(c["exact_hybrid"] for c in cc)/len(cc),
                              target=sum(c["target_digit_success"] for c in cc)/len(cc),
                              preserve=sum(c["preserve_digit_success"] for c in cc)/len(cc))))


if __name__=="__main__":main()
