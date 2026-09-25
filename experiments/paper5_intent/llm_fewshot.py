"""Few-shot LLM intent baseline (Phi-3.5-mini, int4, CPU) on a stratified test subset.

The model sees the 23 known intents with one training example each and may answer
'unknown'. An answer that is not a known label (or 'unknown') counts as an abstain.
Usage: python llm_fewshot.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import onnxruntime_genai as og

HERE = Path(__file__).parent
MODEL = HERE.parent / "_models" / "phi35"
PER_INTENT = 4

te = pd.read_csv(HERE / "test_minilm_n20.csv", index_col=0)
tr = pd.read_csv(HERE / "train_n20.csv", index_col=0)
labels = sorted(tr.intent.unique())
shots = tr.groupby("intent").head(1).set_index("intent").instruction.to_dict()
sub = te.groupby("intent").sample(PER_INTENT, random_state=3)

lines = "\n".join(f"- {l}: e.g. \"{' '.join(shots[l].split()[:8])}\"" for l in labels)
SYSTEM = ("Classify the customer message into exactly one intent label from this list. "
          "If none fits, answer unknown.\n" + lines +
          "\n- unknown: the request is not one of the labels above\n"
          "Answer with the label only.")


def main():
    out_path = HERE / "llm_fewshot_raw.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    model = og.Model(str(MODEL))
    tok = og.Tokenizer(model)
    for idx, row in sub.iterrows():
        k = str(idx)
        if k in done:
            continue
        prompt = f"<|system|>\n{SYSTEM}<|end|>\n<|user|>\n{row.instruction}<|end|>\n<|assistant|>\n"
        ids = tok.encode(prompt)
        p = og.GeneratorParams(model)
        p.set_search_options(max_length=len(ids) + 12, do_sample=False)
        g = og.Generator(model, p)
        t0 = time.time()          # wall time includes prompt processing
        g.append_tokens(ids)
        new = []
        while not g.is_done():
            g.generate_next_token()
            new.append(g.get_next_tokens()[0])
        raw = tok.decode(new).strip()
        done[k] = {"raw": raw, "seconds": round(time.time() - t0, 2)}
        out_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(k, row.intent, "->", raw, done[k]["seconds"], flush=True)
        del g
    score(done)


def score(done):
    rows = []
    for idx, row in sub.iterrows():
        raw = done[str(idx)]["raw"].split("\n")[0].strip().strip(".\"'` ").lower()
        pred = raw if raw in labels or raw == "unknown" else "invalid"
        rows.append({"intent": row.intent, "noisy": row.noisy, "llm": pred,
                     "minilm": row.pred_intent, "minilm_auto": row.auto_open,
                     "seconds": done[str(idx)]["seconds"]})
    r = pd.DataFrame(rows)
    held = ~r.intent.isin(labels)
    known = ~held
    res = {
        "n": len(r), "n_known": int(known.sum()), "n_unknown": int(held.sum()),
        "llm_acc_known": float((r.llm[known] == r.intent[known]).mean()),
        "minilm_acc_known": float((r.minilm[known] == r.intent[known]).mean()),
        "llm_acc_known_noisy": float((r.llm[known & r.noisy] == r.intent[known & r.noisy]).mean()),
        "minilm_acc_known_noisy": float((r.minilm[known & r.noisy] == r.intent[known & r.noisy]).mean()),
        "llm_invalid_rate": float((r.llm == "invalid").mean()),
        "llm_unknown_on_heldout": float((r.llm[held] == "unknown").mean()),
        "llm_unknown_on_known": float((r.llm[known] == "unknown").mean()),
        "llm_wrong_label_on_heldout": float(((r.llm[held] != "unknown") & (r.llm[held] != "invalid")).mean()),
        "minilm_auto_on_heldout": float(r.minilm_auto[held].mean()),
        "llm_mean_seconds": float(r.seconds.mean()),
    }
    (HERE / "llm_fewshot_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
