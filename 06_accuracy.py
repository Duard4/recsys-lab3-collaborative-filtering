# -*- coding: utf-8 -*-
"""Крок 6 (ЛР3). Класична оцінка точності прогнозу сили взаємодії (MAE, RMSE, покриття) на відкладеній вибірці.
Для кожного активного/пасивного користувача з ≥ 2 курсами в історії випадково ховаємо 20 % його взаємодій (не менше однієї);
схожості перераховуємо на решті даних; прогнозуємо сили прихованих взаємодій кожним методом.
Базові лінії: середня сила користувача (за навчальною частиною) та середня сила курсу.
Покриття – частка прихованих пар, для яких метод дав прогноз. Для порівняння точності «на однакових парах»
наводимо також MAE базової лінії (середня користувача) на тих самих парах."""
import json
from cf_lib import *

u, c, R, cl, uid_idx, cid_idx, sd = load_matrix()
U, C = R.shape
grp = u["activity_group"].to_numpy()
rng = np.random.default_rng(SEED + 1)
hidden = {}
for i in np.where(np.isin(grp, EVAL_GROUPS))[0]:
    row = R.getrow(i)
    if row.nnz < 2:
        continue
    n_h = max(1, int(round(0.2 * row.nnz)))
    hidden[i] = rng.choice(row.indices, size=n_h, replace=False)
rows = np.array(sorted(hidden))
Rl = R.tolil()
for i, js in hidden.items():
    for j in js:
        Rl[i, j] = 0
Rtr = Rl.tocsr(); Rtr.eliminate_zeros()
cands = {i: np.sort(hidden[i]) for i in rows}
truth = {i: np.asarray(R[i, cands[i]].todense()).ravel() for i in rows}
n_eval = {g: int((grp[rows] == g).sum()) for g in EVAL_GROUPS}
n_pairs = {g: int(sum(len(cands[i]) for i in rows[grp[rows] == g])) for g in EVAL_GROUPS}
print("користувачів:", n_eval, "прихованих пар:", n_pairs, flush=True)

cnt = np.diff(Rtr.indptr)
mean_u = np.divide(np.asarray(Rtr.sum(1)).ravel(), cnt, out=np.zeros(U), where=cnt > 0)
n_i = np.asarray((Rtr > 0).sum(0)).ravel()
mean_i = np.divide(np.asarray(Rtr.sum(0)).ravel(), n_i, out=np.full(C, np.nan), where=n_i > 0)

preds = {}
preds["База: середня користувача"] = {i: np.full(len(cands[i]), mean_u[i]) for i in rows}
preds["База: середня курсу"] = {i: mean_i[cands[i]] for i in rows}

S_co, nco = pearson_corated(Rtr)
preds["Реснік, Пірсон за ко-рейтингом (класика)"] = {i: v[0] for i, v in user_predict_resnick(Rtr, corated_filter(S_co, nco, 2, 1), rows, cands).items()}
SF = pearson_full(Rtr)
preds["РЕК. 1 – Пірсон"] = {i: v[0] for i, v in user_predict(Rtr, SF, rows, cands).items()}
iuf = iuf_weights(Rtr)
SIU = pearson_full(Rtr, w=iuf)
preds["РЕК. 2 – Пірсон + IUF"] = {i: v[0] for i, v in user_predict(Rtr, SIU, rows, cands, pop_w=iuf, beta=1.0).items()}
CFG = json.load(open(RES / "config.json", encoding="utf-8"))
r3 = {}
for g, par in CFG["rec3"].items():
    gi = [i for i in rows if grp[i] == dict(active="активні", passive="пасивні")[g]]
    r3.update(user_predict(Rtr, corated_filter(S_co, nco, par["gamma_min"], par["gamma_disc"]), gi, cands))
preds["РЕК. 3 – ко-рейтинг"] = {i: v[0] for i, v in r3.items()}
S4 = corated_filter(SIU, nco, 1, CFG["rec4_gamma_disc"])
preds["РЕК. 4 – кращі сусіди"] = {i: v[0] for i, v in user_predict(Rtr, S4, rows, cands, k=CFG["rec4_k"], pop_w=iuf, beta=1.0).items()}
preds["РЕК. 5 – item-based"] = {i: v[0] for i, v in item_predict(Rtr, item_similarity(Rtr, CFG["rec5_kind"]), rows, cands, k=CFG["rec5_k"]).items()}

out = []
for nm, pr in preds.items():
    row = dict(method=nm)
    for g in EVAL_GROUPS:
        gi = rows[grp[rows] == g]
        y = np.concatenate([truth[i] for i in gi])
        p = np.concatenate([pr[i] for i in gi])
        b = np.concatenate([np.full(len(cands[i]), mean_u[i]) for i in gi])
        ok = np.isfinite(p)
        e = p[ok] - y[ok]
        key = GROUP_NAMES[g]
        row.update({f"MAE_{key}": float(np.abs(e).mean()), f"RMSE_{key}": float(np.sqrt((e ** 2).mean())),
                    f"coverage_{key}": float(ok.mean()), f"MAE_base_same_{key}": float(np.abs(b[ok] - y[ok]).mean())})
    out.append(row)
df = pd.DataFrame(out)
df.to_csv(RES / "accuracy_holdout.csv", index=False, encoding=ENC)
json.dump(dict(n_eval=n_eval, n_pairs=n_pairs, hide_share=0.2), open(RES / "accuracy_meta.json", "w", encoding="utf-8"), ensure_ascii=False)
print(df.round(3).to_string())
json.dump(dict(meta=dict(n_eval=n_eval, n_pairs=n_pairs, hide_share=0.2), rows=out), open(RES / "accuracy_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
