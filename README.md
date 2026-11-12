# Spain Mobility

Analysis of 3.11B Spanish origin–destination trips across 3,792 districts (MITMA OpenData), combined with census income/demographics (INE), unemployment (SEPE), and Foursquare POIs.

Research internship at Northeastern University (supervised by Prof. Esteban Moro). Brief: [`docs/project.md`](docs/project.md).

## Findings

1. **Income diversity:** Higher-income destinations have more balanced visitor mixes and lower divergence from the national baseline (JSD $\rho \approx -0.37$); lower-income and peripheral areas are the most selective.
2. **Local assortativity:** Mobility is strongly income-assortative under 25 km ($r \approx 0.49$, ~90% of trips) and decays to ~0 beyond 100 km, confirmed across log-OLS and PPML gravity models.
3. **Destination entropy & geographic opportunity:** Raw destination entropy correlates positively with income and negatively with working-age unemployment ($\rho \approx -0.15$), but controlling for the 50 km geographic opportunity set explains the bulk of variance (~38–41%), leaving a modest positive income effect (~1–2%) and absorbing the unemployment association. Within-origin differences across MITMA income bands are largely driven by unattributed (`NA` age/gender) trip records concentrated in the lowest income band.

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | `phase01`–`phase05` (ingestion, temporal patterns, v1/v2 crosswalks, age×income gravity, segregation) and `entropy.ipynb` ($S_{50}$ spatial entropy) |
| `scripts/` | `prep_external_data.py`, `gravity_income.py` (OLS/PPML), `eda_plots.py`, `build_pptx.py` |
| `results/` | HTML slide decks (`phase03_results.html`–`phase05_results.html`) |
| `data/` | Git-ignored (~7.8 GB of derived MITMA parquet, geometries, POI, and INE/SEPE tables) |

## Setup

```bash
conda env create -f environment.yml
conda activate spain-mobility
python scripts/prep_external_data.py
```
