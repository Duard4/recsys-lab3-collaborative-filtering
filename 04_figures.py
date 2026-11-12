# -*- coding: utf-8 -*-
"""Крок 4 (ЛР3). Рисунки для звіту."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cf_lib import *

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 150, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
P_ = PALETTE
GA, GP = "активні", "пасивні"
GCOL = {GA: "#4C78A8", GP: "#F58518"}
tune = pd.read_csv(RES / "tuning_holdout.csv", encoding=ENC)
main = pd.read_csv(RES / "compare_main.csv", encoding=ENC)
ov = pd.read_csv(RES / "compare_overlap_lab2.csv", encoding=ENC)
sh = pd.read_csv(RES / "compare_cluster_share.csv", encoding=ENC)
pw = pd.read_csv(RES / "compare_pairwise_rec.csv", encoding=ENC)
u, c, R, cl, uid_idx, cid_idx, sd = load_matrix()
K = int(cl.max()) + 1
grp = u["activity_group"].to_numpy()
SHORT = ["Природн.", "Економіка", "Психол.", "Мови", "Мистецтво", "Гуманіт.", "Право", "Інженерія", "Змішаний", "Інформ."]

# ---------- 1. кандидатна вибірка
hist = (R > 0).toarray()
rows = np.where(np.isin(grp, EVAL_GROUPS))[0]
cand = pd.read_csv(RES / "candidates_100.csv", encoding=ENC)
cid_i = {x: i for i, x in enumerate(c["course_id"])}
pop = np.asarray((R > 0).sum(0)).ravel()
elig = pop > 0
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
size_all = np.bincount(cl, minlength=K) / len(cl)
size_el = np.array([(elig & (cl == k)).sum() for k in range(K)]); size_el = size_el / size_el.sum()
for g, off in [(GA, -0.2), (GP, 0.2)]:
    sub = cand[cand.activity_group == g]
    cnt = np.zeros(K)
    for s in sub.course_ids:
        cnt += np.bincount(cl[[cid_i[x] for x in s.split(";")]], minlength=K)
    ax[0].bar(np.arange(K) + off, cnt / cnt.sum() * 100, 0.4, color=GCOL[g], label=f"підвибірка, {g}")
ax[0].plot(np.arange(K), size_el * 100, "k_", ms=14, mew=2, label="частка кластера серед курсів із взаємодіями")
ax[0].set_xticks(range(K)); ax[0].set_xlabel("кластер C1"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 21); ax[0].legend(loc="upper left", fontsize=7); ax[0].set_title("а) частка кластерів у підвибірці")
allpop = np.concatenate([pop[[cid_i[x] for x in s.split(";")]] for s in cand.course_ids])
bins = np.logspace(0, np.log10(pop.max() + 1), 26)
ax[1].hist(pop[elig], bins=bins, weights=np.ones(elig.sum()) / elig.sum() * 100, color="#B0B0B0", alpha=.9, label="усі курси з взаємодіями")
ax[1].hist(allpop, bins=bins, weights=np.ones(len(allpop)) / len(allpop) * 100, color="#54A24B", alpha=.65, label="курси підвибірок")
ax[1].set_xscale("log"); ax[1].set_xlabel("кількість користувачів курсу"); ax[1].set_ylabel("% курсів"); ax[1].legend(fontsize=7); ax[1].set_title("б) популярність: каталог і підвибірки")
# в) ко-рейтингові позиції: частка користувачів, що мають ≥ n сусідів із ≥ γ спільними курсами
M = (R > 0).astype(np.float64)
O = (M @ M.T).toarray(); np.fill_diagonal(O, 0)
gam = [1, 2, 3, 5, 10, 20]
for g in [GA, GP]:
    idx = np.where(grp == g)[0]
    ax[2].plot(gam, [np.mean((O[idx] >= x).sum(1) >= 10) * 100 for x in gam], "o-", color=GCOL[g], label=g)
ax[2].set_xlabel("мінімум спільно оцінених курсів γ"); ax[2].set_ylabel("% користувачів"); ax[2].set_xscale("log"); ax[2].set_xticks(gam); ax[2].set_xticklabels(gam)
ax[2].legend(); ax[2].set_title("в) користувачі з ≥ 10 сусідами (≥ γ спільних курсів)")
plt.tight_layout(); plt.savefig(FIG / "l3_candidates.png", bbox_inches="tight"); plt.close()

# ---------- 2. налаштування параметрів
fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
def curve(a, method, key, vals, fmt, title, xl, log=False):
    for g in [GA, GP]:
        y = []
        for v in vals:
            r = tune[(tune.method == method) & (tune.params == fmt.format(v)) & (tune.group == g)]
            y.append(r.HR10.iloc[0] * 100)
        a.plot(range(len(vals)), y, "o-", color=GCOL[g], label=g)
    a.set_xticks(range(len(vals))); a.set_xticklabels(vals); a.set_title(title); a.set_xlabel(xl); a.set_ylabel("HR@10, %"); a.axhline(10, color="#999", lw=.8, ls=":")
curve(ax[0], "РЕК.1 повний Пірсон", "lam", [0, 0.5, 1, 2, 4], "λ={}", "а) РЕК.1: поправка λ", "λ")
curve(ax[1], "РЕК.2 IUF", "beta", [0, 0.5, 1, 1.5], "β={}", "б) РЕК.2: вага IUF у списку", "β")
for g in [GA, GP]:
    y = [tune[(tune.method == "РЕК.3 ко-рейтинг") & (tune.params == f"γ_min={m}, γ_disc=10") & (tune.group == g)].HR10.iloc[0] * 100 for m in [2, 3, 5, 10, 20, 30]]
    ax[2].plot(range(6), y, "o-", color=GCOL[g], label=g)
ax[2].set_xticks(range(6)); ax[2].set_xticklabels([2, 3, 5, 10, 20, 30]); ax[2].set_title("в) РЕК.3: мінімум ко-рейтингових позицій"); ax[2].set_xlabel("γ_min"); ax[2].axhline(10, color="#999", lw=.8, ls=":")
curve(ax[3], "РЕК.4 top-k (IUF, γ_d=5, β=1)", "k", [10, 20, 30, 50, 100, 300], "k={}", "г) РЕК.4: кількість найкращих сусідів", "k")
ax[0].legend()
plt.tight_layout(); plt.savefig(FIG / "l3_tuning.png", bbox_inches="tight"); plt.close()

# ---------- 3. якість на відкладеній вибірці (фінальні конфігурації)
sel = [("випадковий", "орієнтир", "випадковий список"), ("за популярністю", "орієнтир", "за популярністю"), ("частка кластера\nв історії", "орієнтир", "частка кластера в історії"),
       ("Реснік\n(ко-рейтинг)", "Реснік, Пірсон за ко-рейтингом", "центрований, мін. 2"), ("РЕК.1", "РЕК.1 повний Пірсон", "λ=2"), ("РЕК.2", "РЕК.2 IUF", "β=1"),
       ("РЕК.3", "РЕК.3 ко-рейтинг", None), ("РЕК.4", "РЕК.4 top-k (IUF, γ_d=5, β=1)", "k=50"), ("РЕК.5", "РЕК.5 об'єкти: cos", "k=50, λ=2")]
cfg = json.load(open(RES / "config.json", encoding="utf-8"))
def val(method, params, g):
    if params is None:
        p = cfg["rec3"]["active" if g == GA else "passive"]
        params = f"γ_min={p['gamma_min']}, γ_disc={p['gamma_disc']}"
    r = tune[(tune.method == method) & (tune.params == params) & (tune.group == g)].iloc[0]
    return r.HR10 * 100, r.hidden_covered * 100
fig, ax = plt.subplots(figsize=(11, 3.6))
x = np.arange(len(sel))
for g, off in [(GA, -0.2), (GP, 0.2)]:
    v = [val(m, p, g)[0] for _, m, p in sel]
    b = ax.bar(x + off, v, 0.4, color=GCOL[g], label=g)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + 0.8, f"{vi:.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sel], fontsize=8); ax.set_ylabel("HR@10, %"); ax.legend()
ax.set_title("Відкладена вибірка: частка користувачів, у яких прихований курс потрапив до 10 найкращих серед 101 кандидата")
plt.tight_layout(); plt.savefig(FIG / "l3_holdout.png", bbox_inches="tight"); plt.close()

# ---------- 4. частки кластерів
order = ["REC1", "REC2", "REC3", "REC4", "REC5", "ЛР2 №1", "ЛР2 №2", "ЛР2 №3", "ЛР2 №4 (M=100)"]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for a, g in zip(axs, [GA, GP]):
    m = sh[sh.group == g].set_index("list").loc[order][[f"c{k}" for k in range(K)]].to_numpy() * 100
    im = a.imshow(m, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(40, m.max()))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if m[i, j] > 25 else "black")
    a.set_xticks(range(K)); a.set_xticklabels(SHORT, rotation=35, ha="right", fontsize=8); a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=8)
    a.axhline(4.5, color="k", lw=1.2)
    a.set_title(f"{g.capitalize()}: частка рекомендацій кожного кластера, %")
plt.tight_layout(); plt.savefig(FIG / "l3_cluster_share.png", bbox_inches="tight"); plt.close()

# ---------- 5. чотири показники
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
w = 0.38
xs = np.arange(len(order))
def bars(a, col, title, fmt="{:.2f}", ylim=None):
    for g, off in [(GA, -w / 2), (GP, w / 2)]:
        v = main[main.group == g].set_index("list").loc[order][col].to_numpy()
        a.bar(xs + off, v, w, color=GCOL[g], label=g)
        for xi, vi in zip(xs + off, v):
            a.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=6.5)
    a.axvline(4.5, color="k", lw=.8, ls="--"); a.set_xticks(xs); a.set_xticklabels([o.replace(" (M=100)", "") for o in order], rotation=30, ha="right", fontsize=8); a.set_title(title)
    if ylim: a.set_ylim(*ylim)
bars(axs[0, 0], "pair_overlap", "а) середня кількість спільних рекомендацій у парі користувачів групи")
bars(axs[0, 1], "ILD", "б) середня різноманітність списку (ILD = 1 − схожість пар)", ylim=(0, 1.05))
bars(axs[1, 1], "share_popular", "г) частка «популярних» курсів у рекомендаціях", ylim=(0, 1.05))
a = axs[1, 0]
l2n = ["ЛР2 №1", "ЛР2 №2", "ЛР2 №3", "ЛР2 №4 (M=100)"]
for g, off in [(GA, -w / 2), (GP, w / 2)]:
    v = [ov[(ov.rec == r) & (ov.group == g)].share_in_top10.mean() * 100 for r in order[:5]]
    a.bar(np.arange(5) + off, v, w, color=GCOL[g], label=g)
    for xi, vi in zip(np.arange(5) + off, v):
        a.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=7)
a.set_xticks(range(5)); a.set_xticklabels(order[:5]); a.set_title("в) частка рекомендацій, що збігаються з ТОП-10 ЛР2 (середнє за 4 пулами), %")
axs[0, 0].legend()
plt.tight_layout(); plt.savefig(FIG / "l3_measures.png", bbox_inches="tight"); plt.close()

# ---------- 6. попарне порівняння РЕК.
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
n5 = ["REC1", "REC2", "REC3", "REC4", "REC5"]
for a, g in zip(axs, [GA, GP]):
    m = pw[pw.group == g].pivot(index="a", columns="b", values="jaccard").loc[n5, n5].to_numpy()
    im = a.imshow(m, cmap="Greens", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if m[i, j] > .55 else "black")
    a.set_xticks(range(5)); a.set_xticklabels(n5, fontsize=8); a.set_yticks(range(5)); a.set_yticklabels(n5, fontsize=8)
    a.set_title(f"{g.capitalize()}: середній Жаккар списків")
plt.tight_layout(); plt.savefig(FIG / "l3_pairwise.png", bbox_inches="tight"); plt.close()
print("ok")
