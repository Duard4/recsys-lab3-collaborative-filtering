# -*- coding: utf-8 -*-
"""Бібліотека ЛР3: колаборативна фільтрація (user-based та item-based) на матриці взаємодій."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys_lib import PROC, RES, FIG, ROOT, ENC, PALETTE
from cb_lib import load_context, content_similarity

LAB2 = ROOT / "lab2_results"
GROUP_NAMES = {"активні": "active", "пасивні": "passive"}
EVAL_GROUPS = ["активні", "пасивні"]
N_CAND = 100            # розмір підвибірки неоцінених об'єктів
SEED = 20260930
MIN_OVERLAP = 2         # мінімум спільно оцінених об'єктів, за якого кореляція Пірсона взагалі визначена
EPS = 1e-9


# ----------------------------------------------------------------------------------------------
# Матриця взаємодій
# ----------------------------------------------------------------------------------------------
def load_matrix():
    """R (U×C, розріджена): сила взаємодії strength(u,c); M — маска; cl — кластер C1; u, c — таблиці."""
    u, c, it, ut, ct, cl, uid_idx, cid_idx, sd = load_context("cluster_C1")
    U, C = len(u), len(c)
    ui = sd["user_id"].map(uid_idx).to_numpy()
    ci = sd["course_id"].map(cid_idx).to_numpy()
    R = sp.csr_matrix((sd["strength"].to_numpy(float), (ui, ci)), shape=(U, C))
    return u, c, R, cl, uid_idx, cid_idx, sd


def centered(R):
    """Матриця відхилень D = r_uc − r̄_u лише на спостережених позиціях; r̄_u — середнє по всьому профілю користувача."""
    R = sp.csr_matrix(R)
    cnt = np.diff(R.indptr)
    mean = np.divide(np.asarray(R.sum(1)).ravel(), cnt, out=np.zeros(R.shape[0]), where=cnt > 0)
    D = R.copy()
    D.data = D.data - np.repeat(mean, cnt)
    return D, mean, cnt


# ----------------------------------------------------------------------------------------------
# Підвибірка з 100 неоцінених об'єктів (стратифікація за кластерами ЛР1 + популярність усередині страти)
# ----------------------------------------------------------------------------------------------
def quotas(sizes, n):
    """Метод найбільших залишків: n місць пропорційно sizes."""
    raw = sizes / sizes.sum() * n
    q = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - q))[: n - q.sum()]:
        q[k] += 1
    return q


def sample_candidates(R, hist, cl, rows, n=N_CAND, seed=SEED, popularity=True):
    """Для кожного користувача зі списку rows — n неоцінених курсів.
    Квоти страт (кластерів C1) пропорційні розміру кластера серед «кандидатних» курсів (≥1 взаємодії в системі);
    усередині страти курси обираються без повторень з імовірністю, пропорційною кількості користувачів курсу."""
    pop = np.asarray((R > 0).sum(0)).ravel().astype(float)
    elig = pop > 0
    K = int(cl.max()) + 1
    sizes = np.array([(elig & (cl == k)).sum() for k in range(K)])
    out = {}
    for i in rows:
        rng = np.random.default_rng([seed, int(i)])
        free = elig & ~hist[i]
        avail = np.array([(free & (cl == k)).sum() for k in range(K)])
        q = np.minimum(quotas(sizes.astype(float), n), avail)
        left = n - q.sum()
        while left > 0:                                  # якщо в страті не вистачає курсів — добираємо з найбільших страт
            room = avail - q
            k = int(np.argmax(np.where(room > 0, sizes, -1)))
            if room[k] <= 0:
                break
            q[k] += 1
            left -= 1
        pick = []
        for k in range(K):
            if q[k] == 0:
                continue
            idx = np.where(free & (cl == k))[0]
            p = pop[idx] if popularity else np.ones(len(idx))
            pick.append(rng.choice(idx, size=q[k], replace=False, p=p / p.sum()))
        out[i] = np.sort(np.concatenate(pick))
    return out


# ----------------------------------------------------------------------------------------------
# USER-BASED: кореляція Пірсона між користувачами
# ----------------------------------------------------------------------------------------------
LAMBDA = 2.0            # поправка (демпфування) прогнозу на кількість «свідчень»: r̂ = Σ s·r / (Σ s + λ)


def iuf_weights(R):
    """Зворотна частота користувачів: w_i = ln(N / n_i) (Breese та ін.); N — кількість користувачів з історією."""
    n_i = np.asarray((R > 0).sum(0)).ravel().astype(float)
    N = float((np.diff(sp.csr_matrix(R).indptr) > 0).sum())
    return np.log(N / np.maximum(n_i, 1.0))


def pearson_full(R, w=None):
    """Пірсон між ПОВНИМИ векторами взаємодій користувачів (відсутня взаємодія = 0, середнє — по всьому каталогу).
    w — вага об'єкта (None → 1; для РЕК. 2 — IUF; тоді зважено і середнє, і суми)."""
    X = sp.csr_matrix(R).toarray()
    W = np.ones(X.shape[1]) if w is None else w
    m = (X * W).sum(1, keepdims=True) / W.sum()
    Z = X - m
    Zw = Z * W
    num = Zw @ Z.T
    v = (Zw * Z).sum(1)
    S = num / np.sqrt(np.outer(v, v) + 1e-12)
    np.fill_diagonal(S, 0.0)
    return np.clip(S, -1, 1)


def pearson_corated(R):
    """Класичний Пірсон за ко-рейтинговими позиціями: сума лише по об'єктах, оцінених обома користувачами;
    відхилення — від середнього по спостережених оцінках користувача. Повертає (sim, n_co)."""
    R = sp.csr_matrix(R)
    cnt = np.diff(R.indptr)
    mean = np.divide(np.asarray(R.sum(1)).ravel(), cnt, out=np.zeros(R.shape[0]), where=cnt > 0)
    D = R.copy()
    D.data = D.data - np.repeat(mean, cnt)
    M = (R > 0).astype(np.float64)
    num = (D @ D.T).toarray()
    A = (D.multiply(D).tocsr() @ M.T).toarray()
    den = np.sqrt(np.maximum(A, 0) * np.maximum(A.T, 0))
    S = np.divide(num, den, out=np.zeros_like(num), where=den > EPS)
    n_co = (M @ M.T).toarray()
    np.fill_diagonal(S, 0.0)
    np.fill_diagonal(n_co, 0.0)
    return np.clip(S, -1, 1), n_co


def corated_filter(S, n_co, gamma_min=2, gamma_disc=1):
    """Ко-рейтингові позиції: пари з n_co < gamma_min обнуляються; решта множиться на min(n_co, γ_d)/γ_d."""
    out = S * np.minimum(n_co, gamma_disc) / float(gamma_disc)
    out[n_co < gamma_min] = 0.0
    return out


def user_predict(R, S, rows, cands, k=None, lam=LAMBDA, pop_w=None, beta=0.0):
    """Прогноз r̂_uc = Σ_v s_uv r_vc / Σ_v s_uv по однорангових v (s>0), що мали взаємодії з c; для впорядкування — той самий
    вираз зі знаменником (Σ s + λ), що штрафує об'єкти з малою кількістю свідчень.
    Однорангові — користувачі, що взаємодіяли з ≥1 об'єктом підвибірки. k — залишити k найкращих сусідів за s.
    Повертає {user: (pred, score, support)}: pred — прогноз (поріг застосовується до нього); score — величина впорядкування
    (для РЕК. 2: pred·(w_c/w̄)^β, де w_c — IUF кандидата); support — кількість сусідів, що оцінили об'єкт."""
    Rc = sp.csc_matrix(R)
    Mc = sp.csc_matrix((R > 0).astype(np.float64))
    wbar = pop_w[pop_w > 0].mean() if pop_w is not None else 1.0
    res = {}
    for i in rows:
        cand = cands[i]
        s = np.maximum(S[i], 0.0)
        s = s * (Mc[:, cand].sum(1).A1 > 0)
        if k is not None and (s > 0).sum() > k:
            s[s < np.sort(s)[-k]] = 0.0
        num = Rc[:, cand].T @ s
        den = Mc[:, cand].T @ s
        sup = Mc[:, cand].T @ (s > 0).astype(float)
        pred = np.full(len(cand), np.nan)
        damped = np.full(len(cand), np.nan)
        ok = den > EPS
        pred[ok] = num[ok] / den[ok]                       # прогноз оцінки r̂ (зважене середнє сусідів) — до нього застосовується поріг
        damped[ok] = num[ok] / (den[ok] + lam)             # поправка на кількість свідчень — для впорядкування
        score = damped * (pop_w[cand] / wbar) ** beta if (pop_w is not None and beta) else damped
        res[i] = (pred, score, sup)
    return res


# ----------------------------------------------------------------------------------------------
# ITEM-BASED
# ----------------------------------------------------------------------------------------------
def item_similarity(R, kind="cos"):
    """Подібність об'єктів за векторами користувачів (рядки — користувачі, без взаємодії = 0).
    cos — косинус за strength; pearson — Пірсон (центрування по об'єкту); adjcos — скоригований косинус лише за
    користувачами, що оцінили обидва об'єкти (відхилення від середнього користувача), зі значущістю min(n_co,5)/5."""
    X = sp.csr_matrix(R).toarray()
    if kind == "cos":
        nrm = np.linalg.norm(X, axis=0) + 1e-9
        S = (X.T @ X) / np.outer(nrm, nrm)
    elif kind == "pearson":
        Xc = X - X.mean(0, keepdims=True)
        nrm = np.linalg.norm(Xc, axis=0) + 1e-9
        S = (Xc.T @ Xc) / np.outer(nrm, nrm)
    else:
        Rr = sp.csr_matrix(R)
        cnt = np.diff(Rr.indptr)
        mean = np.divide(np.asarray(Rr.sum(1)).ravel(), cnt, out=np.zeros(Rr.shape[0]), where=cnt > 0)
        D = Rr.copy(); D.data = D.data - np.repeat(mean, cnt)
        M = (Rr > 0).astype(np.float64)
        num = (D.T @ D).toarray()
        A = (D.multiply(D).tocsr().T @ M).toarray()
        S = np.divide(num, np.sqrt(np.maximum(A, 0) * np.maximum(A.T, 0)), out=np.zeros_like(num), where=A > EPS)
        n_co = (M.T @ M).toarray()
        S = S * np.minimum(n_co, 5) / 5.0
        S[n_co < 2] = 0.0
    np.fill_diagonal(S, 0.0)
    return np.clip(S, -1, 1)


def item_predict(R, SI, rows, cands, k=50, lam=LAMBDA):
    """r̂_ui = Σ_j s_ij r_uj / (Σ_j s_ij + λ) по k найбільш схожих на i об'єктах j з профілю u (s>0).
    Вектор подібності об'єкта i — рядок SI[i, I_u]."""
    Rr = sp.csr_matrix(R)
    res = {}
    for i in rows:
        lo, hi = Rr.indptr[i], Rr.indptr[i + 1]
        items, vals = Rr.indices[lo:hi], Rr.data[lo:hi]
        Sm = np.maximum(SI[np.ix_(cands[i], items)], 0.0)
        if Sm.shape[1] > k:
            thr = np.sort(Sm, axis=1)[:, -k][:, None]
            Sm = np.where(Sm >= thr, Sm, 0.0)
        den = Sm.sum(1)
        pred = np.full(len(cands[i]), np.nan)
        damped = np.full(len(cands[i]), np.nan)
        ok = den > EPS
        pred[ok] = (Sm @ vals)[ok] / den[ok]
        damped[ok] = (Sm @ vals)[ok] / (den[ok] + lam)
        res[i] = (pred, damped, (Sm > 0).sum(1).astype(float))
    return res


# ----------------------------------------------------------------------------------------------
# Формування списків і метрики
# ----------------------------------------------------------------------------------------------
def select_top(pred, score, sup, theta, n=10):
    """Відбір за порогом r̂ ≥ θ; впорядкування за score ↓, далі за кількістю опор ↓. Повертає позиції в підвибірці."""
    ok = np.where(np.isfinite(pred) & (pred >= theta))[0]
    if len(ok) == 0:
        return ok
    order = np.lexsort((ok, -sup[ok], -score[ok]))
    return ok[order][:n]


def intra_list_diversity(S, items):
    """Середня різноманітність списку: 1 − середня контентна схожість по всіх парах."""
    if len(items) < 2:
        return np.nan
    sub = S[np.ix_(items, items)]
    iu = np.triu_indices(len(items), 1)
    return float(1 - sub[iu].mean())


def user_predict_resnick(R, S, rows, cands, k=None):
    """Класична формула Ресніка (для порівняння способів обчислення): r̂ = r̄_u + Σ s (r_vc − r̄_v) / Σ |s|."""
    Rr = sp.csr_matrix(R)
    cnt = np.diff(Rr.indptr)
    mean = np.divide(np.asarray(Rr.sum(1)).ravel(), cnt, out=np.zeros(Rr.shape[0]), where=cnt > 0)
    D = Rr.copy(); D.data = D.data - np.repeat(mean, cnt)
    Dc, Mc = sp.csc_matrix(D), sp.csc_matrix((Rr > 0).astype(np.float64))
    res = {}
    for i in rows:
        cand = cands[i]
        s = S[i] * (Mc[:, cand].sum(1).A1 > 0)
        if k is not None and (s != 0).sum() > k:
            s[np.abs(s) < np.sort(np.abs(s))[-k]] = 0.0
        num = Dc[:, cand].T @ s
        den = Mc[:, cand].T @ np.abs(s)
        sup = Mc[:, cand].T @ (np.abs(s) > 0).astype(float)
        pred = np.full(len(cand), np.nan)
        ok = den > EPS
        pred[ok] = mean[i] + num[ok] / den[ok]
        res[i] = (pred, pred.copy(), sup)
    return res
