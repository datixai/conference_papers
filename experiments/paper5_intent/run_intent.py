"""Intent + aspect detection with an abstain band for handover to automation.

Data: Bitext customer-support dataset (public, 26,872 messages, 27 intents,
11 categories). Four intents are held out of training and appear only at test
time as unknown requests that must go to a human.
Models: TF-IDF + logistic regression; frozen MiniLM sentence encoder + logistic
regression. The few-shot LLM baseline is run separately (llm_fewshot.py).
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tokenizers import Tokenizer

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "bitext.csv"
MINILM = HERE.parent / "_models" / "minilm"
SEED = 7

ASPECT = {  # business aspect used for routing, derived from the Bitext category
    "ACCOUNT": "access", "PAYMENT": "billing", "INVOICE": "billing", "REFUND": "billing",
    "CANCEL": "billing", "DELIVERY": "delivery", "SHIPPING": "delivery", "ORDER": "order",
    "CONTACT": "other", "FEEDBACK": "other", "SUBSCRIPTION": "other",
}
HELD_OUT = ["track_refund", "change_shipping_address", "check_cancellation_fee", "switch_account"]
NOISE_FLAGS = set("ZQEW")   # typos, colloquial, abbreviations, offensive


def load():
    d = pd.read_csv(DATA)
    d["aspect"] = d["category"].map(ASPECT)
    d["noisy"] = d["flags"].fillna("").apply(lambda f: bool(set(f) & NOISE_FLAGS))
    d = d.drop_duplicates("instruction").reset_index(drop=True)
    return d


def split(d, n_train):
    rng = np.random.default_rng(SEED)
    tr, va, te = [], [], []
    for intent, g in d.groupby("intent"):
        idx = rng.permutation(g.index.values)
        if intent in HELD_OUT:
            te += list(idx[:150])
            continue
        te += list(idx[:150])
        va += list(idx[150:180])
        tr += list(idx[180:180 + n_train])
    return d.loc[tr], d.loc[va], d.loc[te]


class MiniLM:
    def __init__(self):
        self.tok = Tokenizer.from_file(str(MINILM / "tokenizer.json"))
        self.tok.enable_truncation(128)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(str(MINILM / "model.onnx"), providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]

    def encode(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok.encode_batch(list(texts[i:i + bs]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / m.sum(1)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.vstack(out)


def features(kind, tr, others, enc=None):
    if kind == "tfidf":
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(tr.instruction)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit(tr.instruction)
        f = lambda s: hstack([w.transform(s.instruction), c.transform(s.instruction)]).tocsr()
        return f(tr), [f(o) for o in others]
    cache = {}
    def f(s):
        key = tuple(s.index)
        if key not in cache:
            cache[key] = enc.encode(s.instruction.tolist())
        return cache[key]
    return f(tr), [f(o) for o in others]


def handover(p_int, p_asp, int_classes, asp_classes, parent, t):
    """Auto-route only if intent confidence >= t, aspect is not 'other', and the
    aspect head agrees with the aspect implied by the predicted intent."""
    ii = p_int.argmax(1)
    ai = p_asp.argmax(1)
    conf = p_int.max(1)
    pred_int = int_classes[ii]
    pred_asp = asp_classes[ai]
    consistent = np.array([parent[x] == y for x, y in zip(pred_int, pred_asp)])
    auto = (conf >= t) & (pred_asp != "other") & consistent
    return auto, pred_int, pred_asp, conf, consistent


def route_stats(auto, pred_int, te):
    true = te.intent.values
    unknown = np.isin(true, HELD_OUT)
    n = len(te)
    false_auto = auto & (pred_int != true)
    return {
        "n": int(n),
        "coverage": float(auto.mean()),
        "false_auto_rate_of_all": float(false_auto.sum() / n),
        "false_auto_rate_of_auto": float(false_auto.sum() / max(auto.sum(), 1)),
        "false_auto_known": int((false_auto & ~unknown).sum()),
        "false_auto_unknown": int((false_auto & unknown).sum()),
        "unknown_auto_rate": float(auto[unknown].mean()),
    }


def main():
    d = load()
    enc = MiniLM()
    parent = d.drop_duplicates("intent").set_index("intent").aspect.to_dict()
    results = {"held_out": HELD_OUT, "runs": {}, "curves": {}}
    for n_train in (20, 100):
        tr, va, te = split(d, n_train)
        known_te = te[~te.intent.isin(HELD_OUT)]
        for kind in ("tfidf", "minilm"):
            t0 = time.time()
            Xtr, (Xva, Xte) = features(kind, tr, [va, te], enc)
            C = 10.0 if kind == "tfidf" else 20.0
            ci = LogisticRegression(C=C, max_iter=3000).fit(Xtr, tr.intent)
            ca = LogisticRegression(C=C, max_iter=3000).fit(Xtr, tr.aspect)
            fit_s = time.time() - t0
            t1 = time.time()
            pi_te, pa_te = ci.predict_proba(Xte), ca.predict_proba(Xte)
            pi_va, pa_va = ci.predict_proba(Xva), ca.predict_proba(Xva)
            infer_ms = (time.time() - t1) / len(te) * 1000
            known = ~te.intent.isin(HELD_OUT).values
            pred_int = ci.classes_[pi_te.argmax(1)]
            pred_asp = ca.classes_[pa_te.argmax(1)]
            # threshold chosen on the validation split (known intents only):
            # the lowest t whose auto-routed validation error is <= 2 %
            best_t = 0.99
            for t in np.round(np.arange(0.05, 1.0, 0.01), 2):
                a, pv, _, _, _ = handover(pi_va, pa_va, ci.classes_, ca.classes_, parent, t)
                if a.sum() and (pv[a] != va.intent.values[a]).mean() <= 0.02:
                    best_t = float(t)
                    break
            # open-set tuning: hold out 4 known intents at a time as pseudo-unknowns,
            # retrain on the rest, and pick the lowest t whose auto-routed error on
            # the validation split (pseudo-unknowns included) is <= 2 %.
            rng = np.random.default_rng(SEED + n_train)
            known_int = sorted(tr.intent.unique())
            folds = [list(x) for x in np.array_split(rng.permutation(known_int), 6)]
            fold_scores = []
            for hold in folds:
                trf = tr[~tr.intent.isin(hold)]
                Xf, (Xvf,) = features(kind, trf, [va], enc)
                cif = LogisticRegression(C=C, max_iter=3000).fit(Xf, trf.intent)
                caf = LogisticRegression(C=C, max_iter=3000).fit(Xf, trf.aspect)
                fold_scores.append((cif.predict_proba(Xvf), caf.predict_proba(Xvf), cif.classes_, caf.classes_))
            open_t = 0.99
            for t in np.round(np.arange(0.05, 1.0, 0.01), 2):
                err = tot = 0
                for pif, paf, icl, acl in fold_scores:
                    a, pv, _, _, _ = handover(pif, paf, icl, acl, parent, t)
                    err += (pv[a] != va.intent.values[a]).sum(); tot += a.sum()
                if tot and err / tot <= 0.02:
                    open_t = float(t)
                    break
            key = f"{kind}_n{n_train}"
            run = {
                "train_size": int(len(tr)), "val_size": int(len(va)), "test_size": int(len(te)),
                "intent_acc_known": float(accuracy_score(te.intent.values[known], pred_int[known])),
                "intent_macro_f1_known": float(f1_score(te.intent.values[known], pred_int[known], average="macro")),
                "aspect_macro_f1_known": float(f1_score(te.aspect.values[known], pred_asp[known], average="macro")),
                "intent_acc_clean": float(accuracy_score(te.intent.values[known & ~te.noisy.values], pred_int[known & ~te.noisy.values])),
                "intent_acc_noisy": float(accuracy_score(te.intent.values[known & te.noisy.values], pred_int[known & te.noisy.values])),
                "n_clean": int((known & ~te.noisy.values).sum()), "n_noisy": int((known & te.noisy.values).sum()),
                "threshold": best_t, "fit_seconds": round(fit_s, 1), "infer_ms_per_msg": round(infer_ms, 3),
            }
            auto0, pi0, _, _, cons = handover(pi_te, pa_te, ci.classes_, ca.classes_, parent, 0.0)
            run["no_abstain"] = route_stats(auto0 | True, pred_int, te)   # route everything
            autot, pit, _, _, _ = handover(pi_te, pa_te, ci.classes_, ca.classes_, parent, best_t)
            run["abstain"] = route_stats(autot, pit, te)
            run["open_threshold"] = open_t
            ao, po, _, _, _ = handover(pi_te, pa_te, ci.classes_, ca.classes_, parent, open_t)
            run["abstain_open"] = route_stats(ao, po, te)
            run["abstain_open_noisy_only"] = route_stats(ao[te.noisy.values], po[te.noisy.values], te[te.noisy.values])
            run["abstain_noisy_only"] = route_stats(autot[te.noisy.values], pit[te.noisy.values], te[te.noisy.values])
            run["aspect_consistency_rate"] = float(cons.mean())
            results["runs"][key] = run
            curve = []
            for t in np.round(np.arange(0.0, 1.0, 0.02), 2):
                a, p, _, _, _ = handover(pi_te, pa_te, ci.classes_, ca.classes_, parent, t)
                s = route_stats(a, p, te)
                curve.append([float(t), s["coverage"], s["false_auto_rate_of_all"], s["false_auto_rate_of_auto"]])
            results["curves"][key] = curve
            print(key, json.dumps(run, indent=None)[:900], flush=True)
            if n_train == 20 and kind == "minilm":
                te = te.assign(auto_open=ao)
                # save the test split and encoder predictions for the LLM comparison
                te_out = te.assign(pred_intent=pred_int, conf=pi_te.max(1))
                te_out[["instruction", "intent", "aspect", "noisy", "pred_intent", "conf", "auto_open"]].to_csv(HERE / "test_minilm_n20.csv")
                tr[["instruction", "intent"]].to_csv(HERE / "train_n20.csv")
    (HERE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
