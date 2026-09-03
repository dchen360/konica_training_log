# Konica Minolta Interrater Reliability

This project analyzes Konica Minolta measurements with repeated measurements across raters, participants, and body sites.

## Expected input

The preferred input is a `data/` directory with one CSV file per participant visit.

The script extracts metadata from each filename:

- rater code from `KH`, `EB`, or `RVZ`, case-insensitive
- participant ID from `SID####` or `SID ####`, where the ID is exactly 4 digits

Example:

- `5-19-2026 Subject 3 (SID1329) KH.csv` maps to participant `1329` and rater `KH`

Within each file, the script expects:

- `Group` for body site
- `L*`
- `b*`

It computes ITA with `openox.ita(...)`, then for each `participant` x `rater` x `body_site` triplicate it keeps the row whose ITA is the median. The retained ITA is used as the analysis measurement.

The script can also accept a preprocessed CSV or Excel file if it already contains:

- `participant`
- `rater`
- `body_site`
- `ita` or another analysis column passed through `--measurement-column`

## Analyses

The script runs:

1. A mixed-effects model `measurement ~ C(rater) + C(body_site) + (1 | participant)` to test overall rater bias.
2. A mixed-effects model `measurement ~ C(rater) * C(body_site) + (1 | participant)` to test whether rater differences depend on body site.
3. Pairwise Bland-Altman plots for each rater pair, using a random-intercept model on pairwise differences to account for repeated measures within participant.
4. Pairwise summary tables by body site with the mean and SD of paired rater differences.

## Usage

Create the environment and install dependencies:

```bash
uv venv .venv
uv sync
```

`openox-library` is installed from its GitHub repository through the project dependency definition.

Run the analysis on a folder of raw participant files:

```bash
uv run python analyze_konica.py data
```

Render the HTML reports:

```bash
quarto render ucsf_lab.qmd --to html --output-dir results/report --no-clean
quarto render uganda_lab.qmd --to html --output-dir results/report --no-clean
```

Use `--no-clean` when rendering into the shared `results/report/` directory so that rendering one report does not remove the others.

## Outputs

The script writes analysis outputs into `results/analysis/` by default:

- `preprocessed_measurements.csv`
- `model_main_effects_summary.txt`
- `model_interaction_summary.txt`
- `pairwise_difference_summary.csv`
- `bland_altman_<rater1>_vs_<rater2>.png`

The Quarto reports can be rendered into `results/report/`.
