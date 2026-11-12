# -*- coding: utf-8 -*-
"""Крок 3 (ЛР3). Порівняння рекомендацій між собою та з ЛР2 за чотирма показниками завдання + допоміжні."""
import json
from cf_lib import *

u, c, R, cl, uid_idx, cid_idx, sd = load_matrix()
U, C = R.shape
K = int(cl.max()) + 1
grp = u["activity_group"].to_numpy()
rec = pd.read_csv(RES / "recommendations_lab3.csv", encoding=ENC)
l2 = pd.read_csv(LAB2 / "recommendations_top10.csv", encoding=ENC)
l2 = l2[l2["pool"].isin(["P1", "P2", "P3", "P4_100"])]
cand = pd.read_csv(RES / "candidates_100.csv", encoding=ENC)
S, _, _ = content_similarity(c)
truth_c = pd.read_csv(ROOT / "data" / "ground_truth" / "courses_truth.csv", encoding=ENC).set_index("course_id")
truth_u = pd.read_csv(ROOT / "data" / "ground_truth" / "users_truth.csv", encoding=ENC).set_index("user_id")
area_c = truth_c.loc[c["course_id"], "area"].to_numpy()
pop_grp = c["popularity_group"].to_numpy()
n_users = np.asarray((R > 0).sum(0)).ravel()
# кластерний профіль історії (кількість подій у кластерах, як у ЛР2)
_ev = sd.assign(ui=sd["user_id"].map(uid_idx), cl=cl[sd["course_id"].map(cid_idx).to_numpy()])
Eclu = np.zeros((U, K)); np.add.at(Eclu, (_ev["ui"].to_numpy(), _ev["cl"].to_numpy()), _ev["n_events"].to_numpy(float))

LISTS = ["REC1", "REC2", "REC3", "REC4", "REC5", "ЛР2 №1", "ЛР2 №2", "ЛР2 №3", "ЛР2 №4 (M=100)"]
L2MAP = {"ЛР2 №1": "P1", "ЛР2 №2": "P2", "ЛР2 №3": "P3", "ЛР2 №4 (M=100)": "P4_100"}

def lists_of(name, group):
    df = rec[(rec.rec == name) & (rec.activity_group == group)] if name.startswith("REC") else \
        l2[(l2.pool == L2MAP[name]) & (l2.activity_group == group)]
    return {uid: g["course_id"].map(cid_idx).to_numpy() for uid, g in df.groupby("user_id")}

users_g = {g: u.loc[grp == g, "user_id"].tolist() for g in EVAL_GROUPS}
# ---- Lab 2 pools для «частки у пулі»
pools = {nm: pd.read_csv(LAB2 / f"pool_{nm}.csv", encoding=ENC).set_index("user_id")["course_ids"] for nm in ["P1", "P2", "P3", "P4_100"]}

rows_main, share_rows, rep_extra = [], [], []
LST = {}
for g in EVAL_GROUPS:
    for nm in LISTS:
        L = lists_of(nm, g)
        LST[(nm, g)] = L
        ids = users_g[g]
        n = len(ids)
        lens = np.array([len(L.get(x, [])) for x in ids])
        # (а) частка рекомендацій з кожного кластера
        allc = np.concatenate([L[x] for x in ids if x in L]) if L else np.array([], int)
        sh = np.bincount(cl[allc], minlength=K) / max(len(allc), 1)
        share_rows.append(dict(list=nm, group=g, **{f"c{k}": sh[k] for k in range(K)}))
        # (б) повторювані рекомендації між користувачами групи
        B = sp.csr_matrix((np.ones(len(allc)), (np.repeat(np.arange(len(L)), [len(v) for v in L.values()]), allc)), shape=(len(L), C))
        Gm = (B @ B.T).toarray()
        m = len(L)
        pair_mean = (Gm.sum() - np.trace(Gm)) / (m * (m - 1))
        colsum = np.asarray(B.sum(0)).ravel()
        cov = colsum > 0
        per_course = colsum[cov].mean()
        share_repeated = float(((B.multiply(colsum[None, :] >= 2)).sum()) / max(B.sum(), 1))   # частка рекомендацій, що повторюються ≥ у 2 користувачів
        # (в) різноманітність: ILD та кількість кластерів у списку
        ild = [intra_list_diversity(S, L[x]) for x in ids if x in L and len(L[x]) >= 2]
        ncl = [len(set(cl[L[x]])) for x in ids if x in L and len(L[x]) >= 1]
        # допоміжні: популярність, охоплення, істинна область
        prec = np.mean([np.mean(area_c[L[x]] == truth_u.loc[x, "dominant_area"]) for x in ids if x in L and len(L[x]) > 0])
        dom = np.mean([np.mean(cl[L[x]] == Eclu[uid_idx[x]].argmax()) for x in ids if x in L and len(L[x]) > 0])
        inhist = np.mean([np.mean(Eclu[uid_idx[x]][cl[L[x]]] > 0) for x in ids if x in L and len(L[x]) > 0])
        pop_mean = np.mean(np.concatenate([n_users[L[x]] for x in ids if x in L]))
        popshare = float(np.mean(np.concatenate([pop_grp[L[x]] == "популярні" for x in ids if x in L])))
        rows_main.append(dict(list=nm, group=g, n_users=n, with_list=int((lens > 0).sum()), mean_len=float(lens.mean()),
                              pair_overlap=float(pair_mean), users_per_course=float(per_course), share_repeated=share_repeated,
                              ILD=float(np.mean(ild)), n_clusters=float(np.mean(ncl)), precision_area=float(prec), share_dom_cluster=float(dom), share_hist_clusters=float(inhist),
                              mean_course_users=float(pop_mean), share_popular=popshare, catalog_coverage=float(cov.sum() / C)))
main = pd.DataFrame(rows_main)
# (г) збіг з ЛР2 (за списками ТОП-10) та потрапляння у пули ЛР2
ov_rows = []
for g in EVAL_GROUPS:
    for nm in ["REC1", "REC2", "REC3", "REC4", "REC5"]:
        for l2n in L2MAP:
            a, b = LST[(nm, g)], LST[(l2n, g)]
            vals, inpool = [], []
            for x in users_g[g]:
                if x in a and len(a[x]) > 0:
                    vals.append(len(set(a[x]) & set(b.get(x, []))) / len(a[x]))
            pool_ser = pools[L2MAP[l2n]]
            for x in users_g[g]:
                if x in a and len(a[x]) > 0:
                    pl = set(pool_ser.get(x, "").split(";")) if isinstance(pool_ser.get(x, ""), str) else set()
                    inpool.append(np.mean([c["course_id"].iloc[j] in pl for j in a[x]]))
            ov_rows.append(dict(rec=nm, group=g, lab2=l2n, share_in_top10=float(np.mean(vals)), share_in_pool=float(np.mean(inpool)),
                                users_with_any=float(np.mean(np.array(vals) > 0))))
ov = pd.DataFrame(ov_rows)
# «випадковий рівень» збігу: очікувана частка кандидатної вибірки, що збігається з ТОП-10 ЛР2, якби вибір був незалежним
cd = {r.user_id: set(r.course_ids.split(";")) for r in cand.itertuples()}
chance = []
for g in EVAL_GROUPS:
    for l2n in L2MAP:
        v = [len(cd[x] & {c["course_id"].iloc[j] for j in LST[(l2n, g)].get(x, [])}) / 100 for x in users_g[g]]
        chance.append(dict(group=g, lab2=l2n, share_of_candidates_in_lab2_top10=float(np.mean(v))))
chance = pd.DataFrame(chance)
# ---- попарне порівняння РЕК. 1–5 між собою (середній Жаккар списків користувача)
pw_rows = []
names5 = ["REC1", "REC2", "REC3", "REC4", "REC5"]
for g in EVAL_GROUPS:
    for a_ in names5:
        for b_ in names5:
            vals = []
            for x in users_g[g]:
                A, B_ = set(LST[(a_, g)].get(x, [])), set(LST[(b_, g)].get(x, []))
                if A and B_:
                    vals.append(len(A & B_) / len(A | B_))
            pw_rows.append(dict(group=g, a=a_, b=b_, jaccard=float(np.mean(vals)) if vals else np.nan))
pw = pd.DataFrame(pw_rows)
shares = pd.DataFrame(share_rows)
main.to_csv(RES / "compare_main.csv", index=False, encoding=ENC)
ov.to_csv(RES / "compare_overlap_lab2.csv", index=False, encoding=ENC)
chance.to_csv(RES / "compare_chance.csv", index=False, encoding=ENC)
pw.to_csv(RES / "compare_pairwise_rec.csv", index=False, encoding=ENC)
shares.to_csv(RES / "compare_cluster_share.csv", index=False, encoding=ENC)
pd.set_option("display.width", 250)
print(main.round(3).to_string())
print(ov.round(3).to_string())
print(chance.round(3).to_string())
