# -*- coding: utf-8 -*-
"""Крок 2 (ЛР3). Підвибірки з 100 неоцінених курсів і РЕКОМЕНДАЦІЇ 1–5 для активних та пасивних користувачів."""
import json
from cf_lib import *

KAPPA = 1.0     # поріг прогнозу: r̂ ≥ κ·r̄_u (прогноз не нижчий за власну середню силу взаємодії користувача; див. 01_tune.py)
cfg = json.load(open(RES / "config.json", encoding="utf-8"))
u, c, R, cl, uid_idx, cid_idx, sd = load_matrix()
U, C = R.shape
cnt_ = np.diff(R.indptr)
mean_u = np.divide(np.asarray(R.sum(1)).ravel(), cnt_, out=np.zeros(U), where=cnt_ > 0)
grp = u["activity_group"].to_numpy()
hist = (R > 0).toarray()
GMAP = {"активні": "active", "пасивні": "passive"}
rows = np.where(np.isin(grp, EVAL_GROUPS))[0]
print("користувачів:", {g: int((grp == g).sum()) for g in EVAL_GROUPS}, "| «звичайні» використовуються лише як однорангові")

cands = sample_candidates(R, hist, cl, rows)
assert all(not hist[i, cands[i]].any() and len(cands[i]) == N_CAND for i in rows)      # перевірка: жодного оціненого курсу
# --- подібності
iuf = iuf_weights(R)
SF = pearson_full(R)
SI_ = pearson_full(R, w=iuf)
S_co, nco = pearson_corated(R)
S4 = corated_filter(SI_, nco, 1, cfg["rec4_gamma_disc"])
SI_item = item_similarity(R, cfg["rec5_kind"])
res = {"REC1": user_predict(R, SF, rows, cands, lam=cfg["lam"]),
       "REC2": user_predict(R, SI_, rows, cands, lam=cfg["lam"], pop_w=iuf, beta=cfg["rec2_beta"]),
       "REC4": user_predict(R, S4, rows, cands, lam=cfg["lam"], k=cfg["rec4_k"], pop_w=iuf, beta=cfg["rec4_beta"]),
       "REC5": item_predict(R, SI_item, rows, cands, k=cfg["rec5_k"], lam=cfg["lam"])}
r3 = {}
for g in EVAL_GROUPS:
    gi = [i for i in rows if grp[i] == g]
    p = cfg["rec3"][GMAP[g]]
    r3.update(user_predict(R, corated_filter(S_co, nco, p["gamma_min"], p["gamma_disc"]), gi, cands, lam=cfg["lam"]))
res["REC3"] = r3

cid, uid = c["course_id"].to_numpy(), u["user_id"].to_numpy()
title, cat = c["title"].to_numpy(), c["category"].to_numpy()
n_users = np.asarray((R > 0).sum(0)).ravel()
rec_rows, cand_rows, diag = [], [], []
for i in rows:
    cand_rows.append((uid[i], grp[i], ";".join(cid[cands[i]])))
    for nm in ["REC1", "REC2", "REC3", "REC4", "REC5"]:
        pred, score, sup = res[nm][i]
        theta = KAPPA * mean_u[i]
        top = select_top(pred, score, sup, theta)
        diag.append(dict(user_id=uid[i], group=grp[i], rec=nm, n_pred=int(np.isfinite(pred).sum()), n_pass=int((np.isfinite(pred) & (pred >= theta)).sum()), n_list=len(top)))
        for r, j in enumerate(top, 1):
            g_ = cands[i][j]
            rec_rows.append((uid[i], grp[i], nm, r, cid[g_], title[g_], int(cl[g_]), cat[g_], round(float(pred[j]), 3), round(float(score[j]), 3), int(sup[j]), int(n_users[g_])))
cols = ["user_id", "activity_group", "rec", "rank", "course_id", "title", "cluster", "category", "predicted_strength", "score", "support", "course_n_users"]
rec = pd.DataFrame(rec_rows, columns=cols)
rec.to_csv(RES / "recommendations_lab3.csv", index=False, encoding=ENC)
pd.DataFrame(cand_rows, columns=["user_id", "activity_group", "course_ids"]).to_csv(RES / "candidates_100.csv", index=False, encoding=ENC)
dg = pd.DataFrame(diag); dg.to_csv(RES / "list_diagnostics.csv", index=False, encoding=ENC)
# перевірка: рекомендації не містять оцінених курсів
cidx = rec["course_id"].map(cid_idx).to_numpy(); uidx = rec["user_id"].map(uid_idx).to_numpy()
viol = int(hist[uidx, cidx].sum())
print("порушень виключення історії:", viol); assert viol == 0
print(dg.groupby(["group", "rec"])[["n_pred", "n_pass", "n_list"]].mean().round(1))
print("користувачів без жодної рекомендації:\n", dg[dg.n_list == 0].groupby(["group", "rec"]).size())
json.dump(dict(kappa=KAPPA, violations=viol, rows=len(rec)), open(RES / "lab3_meta.json", "w"))
