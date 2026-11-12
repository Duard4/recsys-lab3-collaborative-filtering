# -*- coding: utf-8 -*-
"""Бібліотека ЛР2: контентна схожість курсів, пули стратегій А/Б/В, ТОП-10 рекомендацій."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from recsys_lib import load_clean, sets_to_binary, jaccard_distance_binary, PROC, RES, FIG, TRUTH, ENC, PALETTE, ROOT

L1 = ROOT / "data" / "lab1_results"
POOL_M = [20, 50, 100, 200, 500]          # порогові розміри пулу №4 (стратегія В)
K_NEIGH = 10                               # скільки найближчих сусідів курсу розглядається для пулу №1
TAU = 0.40                                 # мінімальна контентна схожість сусіда (медіана схожості 10-го сусіда ≈ 0,46)
N_SEED = 5                                 # кількість «насіннєвих» курсів для нових користувачів
GROUPS = ["активні", "звичайні", "пасивні", "нові (0)"]


def content_similarity(c):
    """Гібридна контентна схожість курсів: 0.5·TF-IDF(опис) + 0.3·Жаккар(ключові слова) + 0.2·атрибути."""
    n = len(c)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
    T = tf.fit_transform(c["text_clean"])
    S_t = (T @ T.T).toarray()
    Bk, _ = sets_to_binary(c["keywords"])
    S_k = 1 - jaccard_distance_binary(Bk)
    S_a = np.zeros((n, n))
    for col in ["category", "language", "format", "assessment"]:
        v = c[col].fillna("?").to_numpy(dtype=object)
        S_a += (v[:, None] == v[None, :])
    d = c["difficulty"].fillna(c["difficulty"].median()).to_numpy(float)
    S_a += 1 - np.abs(d[:, None] - d[None, :]) / 4
    S_a /= 5
    S = (0.5 * S_t + 0.3 * S_k + 0.2 * S_a).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S, tf, T


def neighbours(S, k=K_NEIGH, tau=TAU):
    """Булева розріджена матриця: курс i → його k найближчих курсів із схожістю ≥ tau."""
    n = len(S)
    S2 = S.copy()
    np.fill_diagonal(S2, -1)
    idx = np.argpartition(-S2, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    ok = S2[rows, cols] >= tau
    return sp.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (rows[ok], cols[ok])), shape=(n, n))


def seeds_for_new_users(u, c, T, tf, strength_df, cats, cat_idx, uid_idx, new_ids, n_seed=N_SEED):
    """Холодний старт: вага категорій з заявлених інтересів (P(категорія|тег), навчена на користувачах з історією)
    + збіг формату + косинус тексту тегів із описом курсу → N_SEED найкращих курсів як «псевдоісторія»."""
    K = len(cats)
    sd = strength_df.copy()
    sd["cat"] = sd["course_id"].map(c.set_index("course_id")["category"]).map(cat_idx)
    A = sp.csr_matrix((sd["strength"], (sd["user_id"].map(uid_idx), sd["cat"])), shape=(len(u), K)).toarray()
    tot = A.sum(1)
    P_user = np.divide(A, tot[:, None], out=np.zeros_like(A), where=tot[:, None] > 0)
    tags = u["interests"].apply(lambda s: [t for t in s.split(";") if t])
    vocab = sorted({t for l in tags for t in l})
    ptag = {}
    for t in vocab:
        m = np.array([(t in l) and tot[i] > 0 for i, l in enumerate(tags)])
        ptag[t] = P_user[m].mean(0) if m.any() else np.ones(K) / K
    cat_of_course = c["category"].map(cat_idx).to_numpy()
    fmt = c["format"].fillna("?").to_numpy(dtype=object)
    out = {}
    for uid in new_ids:
        i = uid_idx[uid]
        tl = tags.iloc[i]
        pi = np.mean([ptag[t] for t in tl], axis=0) if tl else np.ones(K) / K
        q = tf.transform([" ".join(tl).lower()])
        cosq = (T @ q.T).toarray().ravel() if tl else np.zeros(len(c))
        pf = u.loc[i, "preferred_format"]
        score = pi[cat_of_course] + 0.3 * (fmt == pf) + 0.5 * cosq
        top = np.argsort(-score, kind="stable")[:n_seed]
        out[uid] = (top, score[top], pi)
    return out


def quota_split(shares, M, avail):
    """Розподіл M місць пулу №4 між кластерами пропорційно shares (метод найбільших залишків) з урахуванням наявності."""
    K = len(shares)
    raw = shares * M
    q = np.floor(raw).astype(int)
    rem = M - q.sum()
    for k in np.argsort(-(raw - q))[:rem]:
        q[k] += 1
    q = np.minimum(q, avail)
    left = M - q.sum()
    order = np.argsort(-shares, kind="stable")
    while left > 0 and (q < avail).any():
        for k in order:
            if left == 0:
                break
            if q[k] < avail[k] and shares[k] > 0:
                q[k] += 1
                left -= 1
        if not any((q[k] < avail[k]) and shares[k] > 0 for k in order):
            break
    return q


def build_pools(W, E, S, NB, cl, hist_mask, pool_m=POOL_M, top_n=10):
    """Будує пули та ТОП-N для всіх користувачів.
    W (U×C) — ваги історії (сила взаємодії або ваги насіння), E (U×C) — «кількість взаємодій» для активності в кластерах,
    cl — вектор кластерів курсів, hist_mask (U×C) — булева, курси, що виключаються (реальні взаємодії).
    Повертає pools[name] = список масивів індексів курсів, recs[name] = список масивів ТОП-N (індекси) та score."""
    U, C = W.shape
    K = int(cl.max()) + 1
    Wc = sp.csr_matrix(W)
    wsum = np.asarray(Wc.sum(1)).ravel()
    score = np.asarray((Wc @ S)) / np.maximum(wsum, 1e-9)[:, None]       # профільна контентна оцінка (U×C)
    Hb = (Wc > 0).astype(np.float32)
    P1 = (Hb @ NB).toarray() > 0                                          # курси-сусіди курсів історії
    onehot = sp.csr_matrix((np.ones(C), (np.arange(C), cl)), shape=(C, K))
    Eclu = np.asarray(sp.csr_matrix(E) @ onehot.todense())                # U×K активність (кількість взаємодій)
    Hclu = np.asarray(Hb @ onehot.todense()) > 0                          # кластери з історії
    names = ["P1", "P2", "P3"] + [f"P4_{m}" for m in pool_m]
    pools = {nm: [] for nm in names}
    recs = {nm: [] for nm in names}
    cl_members = [np.where(cl == k)[0] for k in range(K)]
    for i in range(U):
        excl = hist_mask[i]
        ok = ~excl
        has = wsum[i] > 0
        sc = score[i]
        if not has:
            for nm in names:
                pools[nm].append(np.array([], int)); recs[nm].append((np.array([], int), np.array([])))
            continue
        pl = {}
        pl["P1"] = np.where(P1[i] & ok)[0]
        pl["P2"] = np.where(Hclu[i][cl] & ok)[0]
        # P3: кластер з найбільшою кількістю взаємодій (за рівності — з більшою сумою ваг)
        wclu = np.asarray(Wc[i] @ onehot.todense()).ravel()
        kmax = int(np.lexsort((-wclu, -Eclu[i]))[0])
        pl["P3"] = cl_members[kmax][ok[cl_members[kmax]]]
        # P4: пропорційно активності
        sh = Eclu[i] / Eclu[i].sum()
        cand = [cl_members[k][ok[cl_members[k]]] for k in range(K)]
        cand = [cd[np.argsort(-sc[cd], kind="stable")] for cd in cand]
        avail = np.array([len(cd) for cd in cand])
        for m in pool_m:
            q = quota_split(sh, m, avail)
            pl[f"P4_{m}"] = np.concatenate([cand[k][:q[k]] for k in range(K)]) if q.sum() else np.array([], int)
        for nm in names:
            idx = pl[nm]
            if nm.startswith("P4"):
                # стратегія В: 10 місць ТОП-10 також розподіляються між кластерами пропорційно активності
                inpool = [idx[cl[idx] == k] for k in range(K)]
                inpool = [cd[np.argsort(-sc[cd], kind="stable")] for cd in inpool]
                slots = quota_split(sh, top_n, np.array([len(cd) for cd in inpool]))
                sel = np.concatenate([inpool[k][:slots[k]] for k in range(K)]) if slots.sum() else np.array([], int)
                order = np.argsort(-sc[sel], kind="stable")
                top = sel[order]
            else:
                order = np.argsort(-sc[idx], kind="stable")[:top_n]
                top = idx[order]
            pools[nm].append(np.sort(idx))
            recs[nm].append((top, sc[top]))
    return pools, recs, dict(score=score, Eclu=Eclu, Hclu=Hclu)


def load_context(cluster_col="cluster_C1"):
    u, c, it, X, R, ut, ct = load_clean()
    cl_df = pd.read_csv(L1 / "courses_cluster_labels.csv", encoding=ENC).set_index("course_id")
    cl = cl_df.loc[c["course_id"], cluster_col].to_numpy()
    uid_idx = {x: i for i, x in enumerate(u["user_id"])}
    cid_idx = {x: i for i, x in enumerate(c["course_id"])}
    sd = pd.read_csv(PROC / "user_course_strength.csv", encoding=ENC)
    return u, c, it, ut, ct, cl, uid_idx, cid_idx, sd
