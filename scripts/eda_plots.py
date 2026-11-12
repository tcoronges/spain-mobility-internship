"""Exploratory plots for the March 2025 district OD data.

Writes figures to results/eda/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[1] / "data"
GEO  = DATA / "geometries"
OUT  = Path(__file__).resolve().parents[1] / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999", "axes.labelcolor": "#333",
    "xtick.color": "#666", "ytick.color": "#666",
    "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})
INK, ACCENT, MUTED = "#1a1a1a", "#2b6cb0", "#b0b0b0"


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- load
print("loading…")
pairs = pd.read_parquet(DATA / "recent_2025-03" / "_od_pairs_cache.parquet")

# MITMA ships 118 non-Spanish zones (French + Portuguese NUTS regions, plus "externo").
# They are whole regions, not districts, have no INE attributes, and as origins they are
# meaningless here. Effect as destinations is negligible (Spearman 0.999 on S) but we drop
# them on both sides so the support is one consistent geography.
_esp = pairs["id_origin"].str[:1].str.isdigit() & pairs["id_destination"].str[:1].str.isdigit()
print(f"  dropping {(~_esp).sum():,} pairs involving non-Spanish zones "
      f"({100*pairs.loc[~_esp,'n_trips'].sum()/pairs['n_trips'].sum():.2f}% of trips)")
pairs = pairs[_esp].copy()

xw    = pd.read_csv(GEO / "relacion_ine_zonificacionMitma.csv", sep="|", dtype=str)
renta = pd.read_csv(DATA / "renta/renta_seccion_2023.csv", dtype={"seccion_ine": str})
pop   = pd.read_csv(DATA / "demo/pop_seccion_2023.csv",     dtype={"seccion_ine": str})
wa    = pd.read_csv(DATA / "demo/workingage_seccion_2023.csv", dtype={"seccion_ine": str})
paro  = pd.read_csv(DATA / "paro/paro_municipios_2025-03.csv", dtype={"municipio_ine": str})

sec = (xw[["seccion_ine", "municipio_ine", "distrito_mitma"]]
       .merge(renta, on="seccion_ine", how="left")
       .merge(pop,   on="seccion_ine", how="left")
       .merge(wa,    on="seccion_ine", how="left"))
sec["w"]  = np.where(sec["renta_persona"].notna(), sec["poblacion"].fillna(0.0), 0.0)
sec["rw"] = sec["renta_persona"].fillna(0.0) * sec["w"]
g = sec.groupby("distrito_mitma")
D = pd.DataFrame({
    "renta":  g["rw"].sum() / g["w"].sum().replace(0, np.nan),
    "pop":    g["poblacion"].sum(),
    "wa_pop": g["working_age_pop"].sum(),
})
muni_wa = sec.groupby("municipio_ine")["working_age_pop"].sum().rename("muni_wa")
sm = sec.merge(paro, on="municipio_ine", how="left").merge(muni_wa, on="municipio_ine", how="left")
sm["sec_paro"] = sm["total_paro"] * sm["working_age_pop"] / sm["muni_wa"].replace(0, np.nan)
D["paro"] = sm.groupby("distrito_mitma")["sec_paro"].sum()
D["paro_rate"] = 100 * D["paro"] / D["wa_pop"].replace(0, np.nan)
D.loc[~np.isfinite(D["paro_rate"]), "paro_rate"] = np.nan

# per-origin summaries
go = pairs.groupby("id_origin")["n_trips"]
O = pd.DataFrame({"T": go.sum(), "N": go.count()})
O = O.join(D[["renta", "pop", "paro_rate"]])
print(f"  {len(pairs):,} OD pairs · {len(O):,} origin districts · {len(D):,} districts w/ attributes")

# ------------------------------------------------- 1. flow size distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
v = pairs["n_trips"].values
ax[0].hist(np.log10(v), bins=80, color=ACCENT, alpha=.85, edgecolor="none")
ax[0].set_xlabel("flow size (log₁₀ trips per OD pair, whole month)")
ax[0].set_ylabel("OD pairs")
ax[0].set_title("Most flows are tiny")
for x, lab in [(np.log10(31), "1 trip/day")]:
    ax[0].axvline(x, color=INK, lw=1, ls="--")
    ax[0].annotate(lab, (x, ax[0].get_ylim()[1]*.92), xytext=(6, 0),
                   textcoords="offset points", fontsize=9, color=INK)
sv = np.sort(v)[::-1]
ax[1].loglog(np.arange(1, len(sv)+1), sv, color=ACCENT, lw=1.4)
ax[1].set_xlabel("rank of OD pair"); ax[1].set_ylabel("trips")
ax[1].set_title("Rank–size: five orders of magnitude")
share = 100*(v < 31).mean()
fig.suptitle(f"OD flow sizes — {share:.0f}% of pairs carry <1 trip/day", y=1.03, fontsize=12, weight="600")
save(fig, "01_flow_sizes.png")

# ------------------------------------------------- 2. per-district totals
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(np.log10(O["T"]), bins=60, color=ACCENT, alpha=.85, edgecolor="none")
ax[0].set_xlabel("log₁₀ total outbound trips"); ax[0].set_ylabel("districts")
ax[0].set_title("Trips generated per district")
ax[1].hist(O["N"], bins=60, color=ACCENT, alpha=.85, edgecolor="none")
ax[1].set_xlabel("N — distinct destinations reached"); ax[1].set_ylabel("districts")
ax[1].set_title("Destinations reached per district")
fig.suptitle("Districts differ enormously in how much data they contribute", y=1.03, fontsize=12, weight="600")
save(fig, "02_district_totals.png")

# ------------------------------------------------- 3. the sampling artifact
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.scatter(O["T"], O["N"], s=6, alpha=.18, color=ACCENT, edgecolors="none")
b = pd.qcut(np.log10(O["T"]), 20, duplicates="drop")
med = O.groupby(b, observed=True).agg(T=("T", "median"), N=("N", "median"))
ax.plot(med["T"], med["N"], color=INK, lw=2, marker="o", ms=4, label="median")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total outbound trips  T"); ax.set_ylabel("destinations reached  N")
r = np.corrcoef(np.log(O["T"]), np.log(O["N"]))[0, 1]
ax.set_title(f"N is largely a function of sampling effort\nlog–log correlation = {r:+.2f}")
ax.legend(frameon=False)
save(fig, "03_sampling_artifact.png")

# ------------------------------------------------- 4. destination profiles
tot = pairs.groupby("id_origin")["n_trips"].transform("sum")
pairs["P"] = pairs["n_trips"] / tot
try:
    nm = pd.read_csv(GEO / "nombres_distritos.csv", sep="|", dtype=str)
    name = dict(zip(nm.iloc[:, 0], nm.iloc[:, 1]))
except Exception:
    name = {}
qs = [0.02, 0.35, 0.75, 0.99]
picks = [O["T"].sort_values().index[int(q*(len(O)-1))] for q in qs]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
cols = ["#c05621", "#2b6cb0", "#2f855a", "#6b46c1"]
for did, c, q in zip(picks, cols, qs):
    s = pairs.loc[pairs["id_origin"] == did, "P"].sort_values(ascending=False).values
    H = -(s*np.log(s)).sum()
    lab = f"{name.get(did, did)[:22]} — T={O.loc[did,'T']:,.0f}, eᴴ={np.exp(H):.0f}"
    ax.loglog(np.arange(1, len(s)+1), s, color=c, lw=1.8, label=lab)
ax.set_xlabel("destination rank"); ax.set_ylabel("share of the district's trips  P")
ax.set_title("What entropy actually sees\nsteep curve = trips funnel into few places; flat = spread out")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
save(fig, "04_destination_profiles.png")

# ------------------------------------------------- 5. context maps
gdf = gpd.read_file(GEO / "distritos_2.geojson")[["id", "geometry"]]
gdf = gdf.merge(O, left_on="id", right_index=True, how="left")
specs = [("renta", "cividis", "net income €/person/yr"),
         ("paro_rate", "magma_r", "registered unemployed per 100 working-age"),
         ("pop", "viridis", "population"),
         ("T", "viridis", "total outbound trips")]
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
for a, (col, cm, lab) in zip(axes.ravel(), specs):
    kw = dict(column=col, cmap=cm, ax=a, legend=True,
              legend_kwds={"label": lab, "shrink": .45},
              missing_kwds={"color": "#e5e5e5"})
    if col in ("pop", "T"):
        kw["norm"] = mpl.colors.LogNorm(vmin=max(gdf[col].min(), 1), vmax=gdf[col].max())
    gdf.plot(**kw); a.set_axis_off(); a.set_title(lab, fontsize=10)
fig.suptitle("Context — everything here is spatially smooth, which is why Moran's I matters",
             y=.93, fontsize=13, weight="600")
save(fig, "05_context_maps.png")

# ------------------------------------------------- 6. collinearity
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(O["pop"], O["T"], s=6, alpha=.18, color=ACCENT, edgecolors="none")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("population"); ax[0].set_ylabel("total outbound trips")
rr = np.corrcoef(np.log(O["pop"].replace(0, np.nan).dropna()),
                 np.log(O.loc[O["pop"].replace(0, np.nan).dropna().index, "T"]))[0, 1]
ax[0].set_title(f"Population vs trips — r={rr:+.2f} in logs")
ax[1].scatter(O["renta"], O["T"], s=6, alpha=.18, color=ACCENT, edgecolors="none")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("net income €/person/yr"); ax[1].set_ylabel("total outbound trips")
ax[1].set_title("Income vs trips")
fig.suptitle("Why population has to be in the regression", y=1.02, fontsize=12, weight="600")
save(fig, "06_collinearity.png")

# ------------------------------------------------- 7. trip distance
cent = gpd.read_file(GEO / "distritos_2.geojson")[["id", "geometry"]].set_index("id")
# centroid must be taken in a projected CRS (ETRS89 / UTM 30N covers Spain), then back to lat/lon
cent = cent.to_crs(25830).geometry.centroid.to_crs(4326)
lon, lat = cent.x, cent.y
R = 6371.0088
o, d = pairs["id_origin"].values, pairs["id_destination"].values
la1, lo1, la2, lo2 = map(np.radians, (lat.reindex(o).values, lon.reindex(o).values,
                                      lat.reindex(d).values, lon.reindex(d).values))
a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
dist = 2*R*np.arcsin(np.sqrt(a))
ok = np.isfinite(dist) & (dist > 0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
w = pairs["n_trips"].values[ok]
ax[0].hist(np.log10(dist[ok]), bins=70, weights=w, color=ACCENT, alpha=.85, edgecolor="none")
ax[0].set_xlabel("log₁₀ distance (km)"); ax[0].set_ylabel("trips")
ax[0].set_title("Trip-weighted distance")
srt = np.argsort(dist[ok])
cum = np.cumsum(w[srt]) / w.sum()
ax[1].semilogx(dist[ok][srt], 100*cum, color=ACCENT, lw=2)
for km in (10, 25, 100):
    pct = 100*cum[np.searchsorted(dist[ok][srt], km)]
    ax[1].axvline(km, color=MUTED, lw=1, ls=":")
    ax[1].annotate(f"{pct:.0f}% under {km} km", (km, pct), xytext=(6, -12),
                   textcoords="offset points", fontsize=9, color=INK)
ax[1].set_xlabel("distance (km)"); ax[1].set_ylabel("cumulative % of trips")
ax[1].set_title("Almost everything is local")
fig.suptitle("How far do trips actually go", y=1.03, fontsize=12, weight="600")
save(fig, "07_trip_distance.png")

print(f"\ndone → {OUT}")
