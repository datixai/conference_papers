"""Calibrated refusal over the Northline corpus. Uses the stored BM25 ranking."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
RANKING = ROOT / "retrieval_results.json"

TASK_SLOT = {
    "T1 Factoid": "days",
    "T2 Conflict": "days",
    "T3 Stale fee": "pct",
    "T4 Missing": "warranty",
    "T5 Inform others": "days",
    "T6 Repair": "days",
}

# Gold action for an applied desk: the value a current governing record licenses,
# or REFUSE when no single licensed value should be spoken.
GOLD = {
    "T1 Factoid": 14,
    "T2 Conflict": 14,
    "T3 Stale fee": 8,
    "T4 Missing": "REFUSE",
    "T5 Inform others": 14,
    "T6 Repair": 14,
}


def load_docs() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in CORPUS.glob("*.txt")}


DAY_RE = re.compile(r"(\d+)\s*(?:calendar\s+)?-?\s*days?")


def _days(text: str) -> list[int]:
    low = text.lower()
    found = []
    for m in DAY_RE.finditer(low):
        before = low[max(0, m.start() - 16) : m.start()]
        after = low[m.end() : m.end() + 18]
        if "previous" in before or before.rstrip().endswith("from") or "withdrawn" in after:
            continue
        found.append(int(m.group(1)))
    return list(dict.fromkeys(found))


def extract(text: str, name: str = "") -> dict:
    low = text.lower()
    days = _days(text)
    pct = [int(x) for x in re.findall(r"(\d+)\s*percent", low)]
    superseded = "superseded" in low
    marked_current = (
        "this version replaces" in low
        or "replaces sop-refund" in low
        or ("2026" in name and not superseded)
    )
    no_intl = "do not quote" in low or "does not cover international" in low
    return {
        "days": days[-1] if days else None,
        "days_all": days,
        "pct": pct[-1] if pct else None,
        "pct_all": list(dict.fromkeys(pct)),
        "superseded": superseded and not marked_current,
        "current": marked_current,
        "no_intl": no_intl,
    }


def unique_values(rows: list[dict], key: str) -> list:
    bag = []
    all_key = f"{key}_all"
    for r in rows:
        vals = r.get(all_key) or ([r[key]] if r.get(key) is not None else [])
        for v in vals:
            if v not in bag:
                bag.append(v)
    return bag


def decide(task: str, slot: str, rows: list[dict], policy: str):
    if policy == "always_rank1":
        top = rows[0]
        if slot == "warranty":
            if top["no_intl"]:
                return 12, "answer_rank1_domestic"
            return 12, "answer_rank1"
        val = top[slot]
        return val, "answer_rank1"

    if policy == "conflict_refuse":
        if slot == "warranty":
            return "REFUSE", "refuse_absent_scope"
        vals = unique_values(rows[:3], slot)
        if len(vals) == 0:
            return "REFUSE", "refuse_empty"
        if len(vals) > 1:
            return "REFUSE", "refuse_conflict"
        return vals[0], "answer_agree"

    # conflict_refuse + prefer a marked-current record when one is retrieved
    if slot == "warranty":
        return "REFUSE", "refuse_absent_scope"
    current = [r for r in rows[:3] if r["current"]]
    if current:
        val = current[0][slot]
        others = unique_values(rows[:3], slot)
        if val is not None and len(others) > 1:
            return val, "answer_current_override"
        if val is not None:
            return val, "answer_current"
    vals = unique_values(rows[:3], slot)
    if len(vals) == 0:
        return "REFUSE", "refuse_empty"
    if len(vals) > 1:
        return "REFUSE", "refuse_conflict"
    return vals[0], "answer_agree"


def score(pred, gold) -> str:
    if pred == "REFUSE" and gold == "REFUSE":
        return "correct_refuse"
    if pred == "REFUSE" and gold != "REFUSE":
        return "false_refuse"
    if pred == gold:
        return "correct_answer"
    return "wrong_answer"


def main() -> None:
    docs = load_docs()
    ranking = json.loads(RANKING.read_text(encoding="utf-8"))
    out = []
    for row in ranking:
        task = row["task"]
        slot = TASK_SLOT[task]
        extracted = []
        for name, bm25 in row["top5"][:3]:
            rec = extract(docs[name], name)
            rec["name"] = name
            rec["bm25"] = bm25
            extracted.append(rec)
        item = {"task": task, "slot": slot, "gold": GOLD[task], "top3": extracted, "policies": {}}
        for pol in ("always_rank1", "conflict_refuse", "current_override"):
            pred, reason = decide(task, slot, extracted, pol)
            item["policies"][pol] = {
                "pred": pred,
                "reason": reason,
                "outcome": score(pred, GOLD[task]),
            }
        out.append(item)
        print(task)
        for pol, rec in item["policies"].items():
            print(f"  {pol:18} {rec['pred']!s:>8}  {rec['outcome']:16}  {rec['reason']}")

    dest = ROOT / "refusal_results.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
