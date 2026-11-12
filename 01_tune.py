# -*- coding: utf-8 -*-
"""Крок 1 (ЛР3). Емпіричний добір параметрів на відкладеній вибірці.
Для кожного активного/пасивного користувача, який має «позитивну» взаємодію (strength ≥ 3: улюблене / запис / висока оцінка)
і ≥ 2 курси в історії, ховаємо одну таку взаємодію. Кандидати = 100 неоцінених (та сама стратифікована вибірка) + прихований курс.
Метрика HR@10 — частка користувачів, у яких прихований курс потрапив до 10 найвищих оцінок серед 101 кандидата
(для випадкового списку ≈ 9,9 %). Некрита: користувачі, для яких прогноз прихованого курсу неможливий, вважаються «промахами»."""
import json
from cf_lib import *

u, c, R, cl, uid_idx, cid_idx, sd = load_matrix()
U, C = R.shape
grp = u["activity_group"].to_numpy()
rng = np.random.default_rng(SEED)
POS = 3.0
hidden = {}
for i in np.where(np.isin(grp, EVAL_GROUPS))[0]:
    row = R.getrow(i)
    pos = row.indices[row.data >= POS]
    if len(pos) and row.nnz >= 2:
        hidden[i] = int(rng.choice(pos))
rows = np.array(sorted(hidden))
Rl = R.tolil()
for i, j in hidden.items():
    Rl[i, j] = 0
Rtr = Rl.tocsr(); Rtr.eliminate_zeros()
hist = (R > 0).toarray()
cands0 = sample_candidates(Rtr, hist, cl, rows)
cands = {i: np.sort(np.append(cands0[i], hidden[i])) for i in rows}
n_g = {g: int((grp[rows] == g).sum()) for g in EVAL_GROUPS}
print("відкладена вибірка:", n_g, flush=True)

out = []
def evaluate(res, method, params):
    """res: {user: (pred, score, sup)}"""
    for g in EVAL_GROUPS:
        hits, cov = [], []
        for i in rows[grp[rows] == g]:
            pred, score, sup = res[i]
            h = int(np.where(cands[i] == hidden[i])[0][0])
            p = np.where(np.isfinite(score), score, -1e9)
            order = np.lexsort((np.arange(len(p)), -sup, -p))
            hits.append(h in order[:10]); cov.append(np.isfinite(pred[h]))
        out.append(dict(method=method, params=params, group=g, HR10=float(np.mean(hits)), hidden_covered=float(np.mean(cov))))
    a, p_ = out[-2], out[-1]
    print(f"{method:34s} {params:26s} акт {a['HR10']:.3f} ({a['hidden_covered']:.2f}) | пас {p_['HR10']:.3f} ({p_['hidden_covered']:.2f})", flush=True)

def ref(fn, name):
    res = {i: (fn(i, cands[i]),) * 2 + (np.zeros(len(cands[i])),) for i in rows}
    evaluate(res, "орієнтир", name)

# --- орієнтири
rr = np.random.default_rng(1)
ref(lambda i, cd: rr.random(len(cd)), "випадковий список")
pop = np.asarray((Rtr > 0).sum(0)).ravel().astype(float)
ref(lambda i, cd: pop[cd], "за популярністю")
K = int(cl.max()) + 1
oh = sp.csr_matrix((np.ones(C), (np.arange(C), cl)), shape=(C, K))
prof = (sp.csr_matrix(Rtr) @ oh).toarray()
ref(lambda i, cd: prof[i][cl[cd]] / max(prof[i].sum(), 1), "частка кластера в історії")

# --- способи обчислення (РЕК. 1)
S_co, nco = pearson_corated(Rtr)
evaluate(user_predict_resnick(Rtr, corated_filter(S_co, nco, 2, 1), rows, cands), "Реснік, Пірсон за ко-рейтингом", "центрований, мін. 2")
SF = pearson_full(Rtr)
for lam in [0, 0.5, 1, 2, 4]:
    evaluate(user_predict(Rtr, SF, rows, cands, lam=lam), "РЕК.1 повний Пірсон", f"λ={lam}")
evaluate(user_predict(Rtr, corated_filter(S_co, nco, 2, 1), rows, cands), "Пірсон за ко-рейтингом, прогноз λ=2", "мін. 2")

# --- РЕК. 2: IUF
iuf = iuf_weights(Rtr)
SI_ = pearson_full(Rtr, w=iuf)
for beta in [0, 0.5, 1, 1.5]:
    evaluate(user_predict(Rtr, SI_, rows, cands, pop_w=iuf, beta=beta), "РЕК.2 IUF", f"β={beta}")

# --- РЕК. 3: ко-рейтингові позиції
for gm in [2, 3, 5, 10, 20, 30]:
    for gd in [1, 10]:
        evaluate(user_predict(Rtr, corated_filter(S_co, nco, gm, gd), rows, cands), "РЕК.3 ко-рейтинг", f"γ_min={gm}, γ_disc={gd}")

# --- РЕК. 4: кращі сусіди
S4 = corated_filter(SI_, nco, 1, 5)
for k in [10, 20, 30, 50, 100, 300]:
    evaluate(user_predict(Rtr, S4, rows, cands, k=k, pop_w=iuf, beta=1), "РЕК.4 top-k (IUF, γ_d=5, β=1)", f"k={k}")

# --- РЕК. 5: item-based
for kind in ["cos", "pearson", "adjcos"]:
    SIt = item_similarity(Rtr, kind)
    for k in [20, 50]:
        for lam in [0.3, 2]:
            evaluate(item_predict(Rtr, SIt, rows, cands, k=k, lam=lam), f"РЕК.5 об'єкти: {kind}", f"k={k}, λ={lam}")

df = pd.DataFrame(out)
df.to_csv(RES / "tuning_holdout.csv", index=False, encoding=ENC)
json.dump(dict(n_eval=n_g, pos_threshold=POS), open(RES / "tuning_meta.json", "w"), ensure_ascii=False)

# ------------------------------------------------------------------ вибрані конфігурації та добір порога θ
CFG = dict(lam=LAMBDA, rec2_beta=1.0, rec3=dict(active=dict(gamma_min=10, gamma_disc=10), passive=dict(gamma_min=2, gamma_disc=10)),
           rec4_k=50, rec4_gamma_disc=5, rec4_beta=1.0, rec5_kind="cos", rec5_k=50)
final = {
    "REC1": user_predict(Rtr, SF, rows, cands),
    "REC2": user_predict(Rtr, SI_, rows, cands, pop_w=iuf, beta=1.0),
    "REC4": user_predict(Rtr, S4, rows, cands, k=50, pop_w=iuf, beta=1.0),
    "REC5": item_predict(Rtr, item_similarity(Rtr, "cos"), rows, cands, k=50),
}
r3 = {}
for g, par in CFG["rec3"].items():
    gi = [i for i in rows if grp[i] == dict(active="активні", passive="пасивні")[g]]
    r3.update(user_predict(Rtr, corated_filter(S_co, nco, par["gamma_min"], par["gamma_disc"]), gi, cands))
final["REC3"] = r3
th_rows = []
cnt_tr = np.diff(Rtr.indptr)
mean_tr = np.divide(np.asarray(Rtr.sum(1)).ravel(), cnt_tr, out=np.zeros(U), where=cnt_tr > 0)
for nm in ["REC1", "REC2", "REC3", "REC4", "REC5"]:
    for g in EVAL_GROUPS:
        for kap in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            L, hit, tot = [], 0, 0
            for i in rows[grp[rows] == g]:
                pred, score, sup = final[nm][i]
                top = select_top(pred, score, sup, kap * mean_tr[i])       # поріг відносно середньої сили взаємодії користувача
                L.append(len(top)); h = int(np.where(cands[i] == hidden[i])[0][0])
                hit += int(h in top); tot += 1
            ml = float(np.mean(L))
            prec = hit / max(sum(L), 1); rec = hit / tot
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            th_rows.append(dict(rec=nm, group=g, kappa=kap, mean_len=ml, share_empty=float(np.mean(np.array(L) == 0)), precision=prec, recall=rec, F1=f1))
tdf = pd.DataFrame(th_rows)
tdf.to_csv(RES / "tuning_theta.csv", index=False, encoding=ENC)
print(tdf.pivot_table(index="kappa", columns=["rec", "group"], values="F1").round(3).to_string())
print(tdf.pivot_table(index="kappa", columns=["rec", "group"], values="mean_len").round(1).to_string())
CFG["kappa"] = None
json.dump(CFG, open(RES / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
