"""Unconstrained gravity model with income distance:

    T_ij ~ P_i P_j / d_ij^g

    ln T_ij ~ ln P_i + ln P_j - g ln d_ij^(g)
    ln T_ij ~ ln P_i + ln P_j - g ln d_ij^(g) - b ln d_ij^(Income)

Estimated via:
  (A) OLS on logs, observed pairs only
  (B) PPML on levels, full matrix incl. zeros (Santos Silva & Tenreyro, 2006)

Writes figures and summary table to results/gravity/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA = Path(__file__).resolve().parents[1] / "data"
GEO  = DATA / "geometries"
OUT  = Path(__file__).resolve().parents[1] / "results" / "gravity"
OUT.mkdir(parents=True, exist_ok=True)
K = 10

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "font.size": 12, "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666", "xtick.color": "#333", "ytick.color": "#333",
})

# ------------------------------------------------------------------ data
print("loading…")
pairs = pd.read_parquet(DATA / "recent_2025-03" / "_od_pairs_cache.parquet")
esp = pairs["id_origin"].str[:1].str.isdigit() & pairs["id_destination"].str[:1].str.isdigit()
pairs = pairs[esp]

xw    = pd.read_csv(GEO / "relacion_ine_zonificacionMitma.csv", sep="|", dtype=str)
renta = pd.read_csv(DATA / "renta/renta_seccion_2023.csv", dtype={"seccion_ine": str})
pop   = pd.read_csv(DATA / "demo/pop_seccion_2023.csv",     dtype={"seccion_ine": str})
sec = (xw[["seccion_ine", "distrito_mitma"]]
       .merge(renta, on="seccion_ine", how="left").merge(pop, on="seccion_ine", how="left"))
sec["w"]  = np.where(sec["renta_persona"].notna(), sec["poblacion"].fillna(0.0), 0.0)
sec["rw"] = sec["renta_persona"].fillna(0.0) * sec["w"]
g = sec.groupby("distrito_mitma")
D = pd.DataFrame({"renta": g["rw"].sum() / g["w"].sum().replace(0, np.nan),
                  "pop":   g["poblacion"].sum()})
D = D[(D["renta"] > 0) & (D["pop"] > 0)].dropna()

di = D.sort_values("renta")
D["q"] = np.ceil((di["pop"].cumsum() / di["pop"].sum()) * K).clip(1, K).astype("Int64")

cent = gpd.read_file(GEO / "distritos_2.geojson")[["id", "geometry"]].set_index("id")
cent = cent.to_crs(25830).geometry.centroid.to_crs(4326)
D = D[D.index.isin(cent.index)]
ids = D.index.to_numpy()
n = len(ids)
print(f"  {n:,} districts -> {n*n - n:,} ordered pairs (full matrix)")

lat = np.radians(cent.reindex(ids).y.to_numpy())
lon = np.radians(cent.reindex(ids).x.to_numpy())
P   = D["pop"].to_numpy(float)
I   = D["renta"].to_numpy(float)
q   = D["q"].to_numpy(int)

# dense matrices (n x n, ~56 MB each in float32)
a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
     + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
dg = (2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)
del a
dinc = np.abs(I[:, None] - I[None, :]).astype(np.float32)

pos = pd.Series(np.arange(n), index=ids)
T = np.zeros((n, n), np.float64)
oi = pos.reindex(pairs["id_origin"]).to_numpy()
dj = pos.reindex(pairs["id_destination"]).to_numpy()
ok = ~(np.isnan(oi) | np.isnan(dj))
np.add.at(T, (oi[ok].astype(int), dj[ok].astype(int)), pairs["n_trips"].to_numpy()[ok])

off = ~np.eye(n, dtype=bool)
valid = off & (dg > 0)
lnPi = np.log(P)[:, None] * np.ones((1, n), np.float32)
lnPj = np.ones((n, 1), np.float32) * np.log(P)[None, :]
lndg = np.log(np.where(valid, dg, 1.0), dtype=np.float64)
lndi = np.log1p(dinc)                      # +1 € offset: income ties would give ln 0
del dinc

m = valid.ravel()
y = T.ravel()[m]
X = np.column_stack([np.ones(m.sum()), lnPi.ravel()[m], lnPj.ravel()[m],
                     lndg.ravel()[m], lndi.ravel()[m]])
del lnPi, lnPj, lndg, lndi, T, dg
NAMES = ["const", "lnP_i", "lnP_j", "ln d_geo", "ln d_income"]
print(f"  matrix cells used {m.sum():,}   nonzero {(y>0).sum():,} "
      f"({100*(y>0).mean():.1f}%)   zeros {(y==0).sum():,}")


def ols_logs(cols):
    """(A) OLS on ln T, observed pairs only."""
    s = y > 0
    Xs, ys = X[np.ix_(s, cols)], np.log(y[s])
    b, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    r = ys - Xs @ b
    return b, 1 - r.var() / ys.var(), int(s.sum())


def ppml(cols):
    """(B) Poisson PMLE on levels, zeros included."""
    Xs = X[:, cols]

    def f(b):
        eta = np.clip(Xs @ b, -50, 50)
        mu = np.exp(eta)
        return -(y @ eta - mu.sum()), -(Xs.T @ (y - mu))

    b0 = np.zeros(len(cols)); b0[0] = np.log(y.mean() + 1e-9)
    r = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-14})
    mu = np.exp(np.clip(Xs @ r.x, -50, 50))
    ss = 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r.x, ss, len(y)


C_BASE, C_FULL = [0, 1, 2, 3], [0, 1, 2, 3, 4]
rows = []
for lab, cols in [("geo only", C_BASE), ("geo + income", C_FULL)]:
    for meth, fn in [("OLS on logs (zeros dropped)", ols_logs), ("PPML (zeros kept)", ppml)]:
        b, fit, nobs = fn(cols)
        d = dict(zip([NAMES[c] for c in cols], b))
        rows.append({"model": lab, "method": meth, "n": nobs,
                     "lnP_i": d["lnP_i"], "lnP_j": d["lnP_j"],
                     "gamma": -d["ln d_geo"], "beta": -d.get("ln d_income", np.nan), "fit": fit})
        print(f"  fitted {lab:14s} {meth}")
R = pd.DataFrame(rows)
print("\n", R.round(4).to_string(index=False))
R.to_csv(OUT / "results.csv", index=False)

# ------------------------------------------------------------------ images
def eq(fname, lines, size=22, h=None):
    fig, ax = plt.subplots(figsize=(9, h or 0.95 * len(lines) + 0.4))
    ax.set_axis_off()
    ys = np.linspace(0.82, 0.18, len(lines)) if len(lines) > 1 else [0.5]
    for t, yy in zip(lines, ys):
        ax.text(0.5, yy, t, ha="center", va="center", fontsize=size, color="black")
    fig.savefig(OUT / fname); plt.close(fig)
    print(f"  {fname}")


def table(fname, cols, rws, widths, fs=12, right_from=1):
    w = sum(widths)
    fig, ax = plt.subplots(figsize=(w, 0.4 * (len(rws) + 1) + 0.3))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(-1.1, len(rws) + 0.9)
    xl = np.cumsum([0] + widths[:-1]) / w
    xr = np.cumsum(widths) / w
    for ci, c in enumerate(cols):
        ax.text(xl[ci] if ci < right_from else xr[ci] - .01, len(rws) + 0.35, c,
                ha="left" if ci < right_from else "right", va="center", fontsize=fs, color="#444")
    ax.plot([0, 1], [len(rws) + 0.02] * 2, color="black", lw=0.9)
    for ri, r in enumerate(rws):
        yy = len(rws) - 1 - ri - 0.35
        for ci, c in enumerate(r):
            ax.text(xl[ci] if ci < right_from else xr[ci] - .01, yy, str(c),
                    ha="left" if ci < right_from else "right", va="center", fontsize=fs, color="black")
    ax.plot([0, 1], [-0.8] * 2, color="black", lw=0.9)
    fig.savefig(OUT / fname); plt.close(fig)
    print(f"  {fname}")


eq("01_model.png", [
    r"$T_{ij}\ \sim\ \dfrac{P_i P_j}{d_{ij}^{\gamma}}$",
    r"$\ln T_{ij}\ \sim\ \ln P_i + \ln P_j - \gamma \ln d_{ij}^{(g)}$",
    r"$\ln T_{ij}\ \sim\ \ln P_i + \ln P_j - \gamma \ln d_{ij}^{(g)} - \beta \ln d_{ij}^{(Income)}$",
], size=21, h=3.4)

eq("01b_income_distance.png",
   [r"$d_{ij}^{(Income)}\ =\ |\,I_i - I_j\,|$"], size=22, h=1.1)

dt = D.groupby("q").agg(lo=("renta", "min"), hi=("renta", "max"), nd=("renta", "size"))
table("02_income_groups.png", ["decile", "min €", "max €", "districts"],
      [[f"D{int(k)}", f"{r['lo']:,.0f}", f"{r['hi']:,.0f}", f"{int(r['nd']):,}"]
       for k, r in dt.iterrows()],
      [1.5, 1.4, 1.4, 1.4])

# fit is deliberately not shown: R² on logs and R² on levels are different quantities
table("03_results.png",
      ["", "pairs used", r"$\ln P_i$", r"$\ln P_j$", r"$\gamma$", r"$\beta$"],
      [[("OLS, zeros dropped" if "OLS" in r["method"] else "PPML, zeros kept")
        + ("" if r["model"] == "geo + income" else "   (geo only)"),
        f"{r['n']:,}", f"{r['lnP_i']:.3f}", f"{r['lnP_j']:.3f}",
        f"{r['gamma']:.3f}", "—" if np.isnan(r["beta"]) else f"{r['beta']:+.3f}"]
       for _, r in R.iterrows()],
      [4.3, 1.6, 1.1, 1.1, 1.0, 1.1])

# distance decay, observed vs what the zeros do to it
s = y > 0
bins = np.logspace(np.log10(2), np.log10(1200), 26)
dgeo = np.exp(X[:, 3])
ix = np.digitize(dgeo, bins)
obs = np.array([y[s][ix[s] == k].mean() if (ix[s] == k).sum() else np.nan for k in range(1, len(bins))])
alll = np.array([y[ix == k].mean() if (ix == k).sum() else np.nan for k in range(1, len(bins))])
mid = np.sqrt(bins[:-1] * bins[1:])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(mid, obs, marker="o", ms=4, lw=1.8, color="#2b6cb0", label="observed pairs only")
ax.loglog(mid, alll, marker="o", ms=4, lw=1.8, color="#c05621", label="all pairs (zeros = 0)")
ax.set_xlabel("distance (km)"); ax.set_ylabel("mean trips per pair")
ax.legend(frameon=False, fontsize=10)
fig.savefig(OUT / "04_zeros_effect.png"); plt.close(fig)
print("  04_zeros_effect.png")

print(f"\ndone → {OUT}")
