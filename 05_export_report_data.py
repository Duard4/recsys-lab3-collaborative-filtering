# -*- coding: utf-8 -*-
"""Крок 5 (ЛР3). Збирання даних для звіту (results/report_data.json)."""
import json
from cf_lib import *

u, c, R, cl, uid_idx, cid_idx, sd = load_matrix()
U, C = R.shape
K = int(cl.max()) + 1
grp = u["activity_group"].to_numpy()
pop = np.asarray((R > 0).sum(0)).ravel()
rec = pd.read_csv(RES / "recommendations_lab3.csv", encoding=ENC)
diag = pd.read_csv(RES / "list_diagnostics.csv", encoding=ENC)
main = pd.read_csv(RES / "compare_main.csv", encoding=ENC)
ov = pd.read_csv(RES / "compare_overlap_lab2.csv", encoding=ENC)
chance = pd.read_csv(RES / "compare_chance.csv", encoding=ENC)
pw = pd.read_csv(RES / "compare_pairwise_rec.csv", encoding=ENC)
sh = pd.read_csv(RES / "compare_cluster_share.csv", encoding=ENC)
tune = pd.read_csv(RES / "tuning_holdout.csv", encoding=ENC)
tth = pd.read_csv(RES / "tuning_theta.csv", encoding=ENC)
cfg = json.load(open(RES / "config.json", encoding="utf-8"))
meta = json.load(open(RES / "lab3_meta.json"))
tmeta = json.load(open(RES / "tuning_meta.json"))
cand = pd.read_csv(RES / "candidates_100.csv", encoding=ENC)

M = (R > 0).astype(np.float64)
O = (M @ M.T).toarray(); np.fill_diagonal(O, 0)
nb = {}
for g in EVAL_GROUPS:
    idx = np.where(grp == g)[0]
    nb[g] = {f"g{x}": float(np.mean((O[idx] >= x).sum(1) >= 1)) for x in [1, 2, 3, 5, 10]}
    nb[g].update({f"g{x}_ge10": float(np.mean((O[idx] >= x).sum(1) >= 10)) for x in [1, 2, 3, 5, 10, 20]})
    nb[g]["median_neighbors_g1"] = float(np.median((O[idx] >= 1).sum(1)))
    nb[g]["median_neighbors_g2"] = float(np.median((O[idx] >= 2).sum(1)))
    nb[g]["mean_courses"] = float(np.diff(R.indptr)[idx].mean())
    nb[g]["min_courses"] = int(np.diff(R.indptr)[idx].min()); nb[g]["max_courses"] = int(np.diff(R.indptr)[idx].max())
    nb[g]["single_item_users"] = int((np.diff(R.indptr)[idx] == 1).sum())
elig = pop > 0
elig_sizes = [int((elig & (cl == k)).sum()) for k in range(K)]
# кандидатна підвибірка
cand_cluster = {}
for g in EVAL_GROUPS:
    cnt = np.zeros(K)
    cid_i = {x: i for i, x in enumerate(c["course_id"])}
    sub = cand[cand.activity_group == g]
    for s in sub.course_ids:
        cnt += np.bincount(cl[[cid_i[x] for x in s.split(";")]], minlength=K)
    cand_cluster[g] = (cnt / len(sub)).tolist()
pops = {g: float(np.mean([pop[[cid_i[x] for x in s.split(";")]].mean() for s in cand[cand.activity_group == g].course_ids])) for g in EVAL_GROUPS}

def trow(method, params, g):
    r = tune[(tune.method == method) & (tune.params == params) & (tune.group == g)].iloc[0]
    return dict(hr=float(r.HR10), cov=float(r.hidden_covered))
def p3(g):
    p = cfg["rec3"]["active" if g == "активні" else "passive"]
    return f"γ_min={p['gamma_min']}, γ_disc={p['gamma_disc']}"
holdout = {}
for g in EVAL_GROUPS:
    holdout[g] = {
        "random": trow("орієнтир", "випадковий список", g), "pop": trow("орієнтир", "за популярністю", g), "clu": trow("орієнтир", "частка кластера в історії", g),
        "resnick": trow("Реснік, Пірсон за ко-рейтингом", "центрований, мін. 2", g),
        "rec1_l0": trow("РЕК.1 повний Пірсон", "λ=0", g), "rec1": trow("РЕК.1 повний Пірсон", "λ=2", g),
        "corated_raw": trow("Пірсон за ко-рейтингом, прогноз λ=2", "мін. 2", g),
        "rec2": trow("РЕК.2 IUF", "β=1", g), "rec2_b0": trow("РЕК.2 IUF", "β=0", g),
        "rec3": trow("РЕК.3 ко-рейтинг", p3(g), g),
        "rec4": trow("РЕК.4 top-k (IUF, γ_d=5, β=1)", "k=50", g),
        "rec5": trow("РЕК.5 об'єкти: cos", "k=50, λ=2", g),
        "rec5_pearson": trow("РЕК.5 об'єкти: pearson", "k=50, λ=2", g), "rec5_adjcos": trow("РЕК.5 об'єкти: adjcos", "k=50, λ=2", g),
    }
# приклади
ex = {}
ev_cnt = sd.groupby("user_id")["n_events"].sum()
for g in EVAL_GROUPS:
    ids = u.loc[grp == g, "user_id"]
    ncs = u.set_index("user_id").loc[ids, "n_courses"]
    med = int(np.median(ncs))
    d = diag[diag.group == g].pivot_table(index="user_id", columns="rec", values="n_list")
    ok = d[(d >= 5).all(axis=1)].index
    pick = [x for x in ids if ncs[x] in (med, med + 1, med - 1) and x in ok]
    uid = pick[0] if pick else list(ok)[0]
    row = u[u.user_id == uid].iloc[0]
    hs = sd[sd.user_id == uid].sort_values("strength", ascending=False)
    hs = hs.merge(c[["course_id", "title"]], on="course_id").head(3)
    hcl = np.bincount(cl[[cid_idx[x] for x in sd[sd.user_id == uid].course_id]], minlength=K)
    ex[g] = dict(user_id=uid, faculty=row.faculty, n_courses=int(row.n_courses), n_events=int(ev_cnt[uid]), interests=row.interests,
                 hist_clusters=[[int(k), int(hcl[k])] for k in np.argsort(-hcl)[:3] if hcl[k] > 0],
                 top_hist=[[r.course_id, r.title, round(float(r.strength), 2)] for r in hs.itertuples()],
                 recs={nm: rec[(rec.user_id == uid) & (rec.rec == nm)].sort_values("rank").head(5)[["course_id", "title", "cluster", "predicted_strength"]].to_dict("records") for nm in ["REC1", "REC2", "REC3", "REC4", "REC5"]})
dg = diag.groupby(["group", "rec"]).agg(n_pred=("n_pred", "mean"), n_pass=("n_pass", "mean"), n_list=("n_list", "mean"), empty=("n_list", lambda s: int((s == 0).sum())), full=("n_list", lambda s: int((s >= 10).sum())), n=("n_list", "size")).reset_index()
out = dict(
    meta=dict(n_users=U, n_courses=C, n_pairs=int(R.nnz), density=float(R.nnz / (U * C)), groups={g: int((u.activity_group == g).sum()) for g in u.activity_group.unique()},
              eligible=int(elig.sum()), new_courses=int((~elig).sum()), mean_strength=float(R.data.mean()), median_strength=float(np.median(R.data)),
              cluster_sizes=np.bincount(cl, minlength=K).tolist(), eligible_sizes=elig_sizes, **{k: v for k, v in cfg.items() if k != "kappa"}, kappa=1.0, violations=meta["violations"], rec_rows=meta["rows"]),
    nb=nb, cand_cluster=cand_cluster, cand_pop=pops, cat_pop_mean=float(pop[elig].mean()), holdout=holdout, n_eval=tmeta["n_eval"],
    main=main.to_dict("records"), overlap=ov.to_dict("records"), chance=chance.to_dict("records"), pairwise=pw.to_dict("records"), cluster_share=sh.to_dict("records"),
    diag=dg.to_dict("records"), examples=ex, theta_tbl=tth.to_dict("records"))
json.dump(out, open(RES / "report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("ok", {g: ex[g]["user_id"] for g in ex}); print(json.dumps(nb, ensure_ascii=False)[:600])
