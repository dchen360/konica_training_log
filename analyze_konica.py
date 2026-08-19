from __future__ import annotations

import argparse
import hashlib
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


RATER_PATTERN = re.compile(r"(?<![A-Za-z])(KH|EB|RVZ)(?![A-Za-z])", re.IGNORECASE)
SID_PATTERN = re.compile(r"SID\s*([0-9]{4})", re.IGNORECASE)
REQUIRED_RAW_COLUMNS = ("Group", "L*", "b*")
BODY_SITE_NORMALIZATION = {
    "Palmer": "Palmar",
}
BODY_SITE_PLOT_ORDER = ["Arm", "Chest", "Dorsal", "Ear", "Forehead", "Palmar"]
BODY_SITE_MARKERS = {
    "Arm": "o",
    "Chest": "s",
    "Dorsal": "^",
    "Ear": "D",
    "Forehead": "P",
    "Palmar": "X",
}
MONK_GROUP_ORDER = list("ABCDEFGHIJ")
MONK_COLORS = {
    "A": "#f7ede4",
    "B": "#f3e7da",
    "C": "#f6ead0",
    "D": "#ead9bb",
    "E": "#d7bd96",
    "F": "#9f7d54",
    "G": "#815d44",
    "H": "#604234",
    "I": "#3a312a",
    "J": "#2a2420",
}
ELLA_PARTICIPANT_PATTERN = re.compile(r"test with\s+(chidera|lily|katie)", re.IGNORECASE)
ELLA_PARTICIPANT_ORDER = ["Chidera", "Lily", "Katie"]
ELLA_PARTICIPANT_COLORS = {
    "Chidera": "#2c7fb8",
    "Lily": "#d95f0e",
    "Katie": "#238b45",
}
ELLA_PARTICIPANT_MARKERS = {
    "Chidera": "o",
    "Lily": "s",
    "Katie": "^",
}
MAIN_MODEL_FORMULA = "measurement ~ C(rater, Treatment(reference='KH')) + C(body_site)"
INTERACTION_MODEL_FORMULA = "measurement ~ C(rater, Treatment(reference='KH')) * C(body_site)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Konica Minolta participant, Monk Scale, and Ella datasets."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a directory containing raw CSV/Excel files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory for tables, markdown fragments, and plots.",
    )
    return parser.parse_args()


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_raw_files(files: list[Path]) -> list[Path]:
    unique_files: list[Path] = []
    seen_hashes: set[str] = set()

    for file in files:
        file_hash = file_sha1(file)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        unique_files.append(file)

    return unique_files


def list_raw_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if not path.is_dir():
        raise ValueError("This report expects a directory of raw CSV/Excel files.")

    files = sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if not files:
        raise ValueError(f"No CSV or Excel files found in directory: {path}")
    return deduplicate_raw_files(files)


def extract_rater_from_filename(path: Path) -> str:
    match = RATER_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not extract rater code KH, EB, or RVZ from filename: {path.name}")
    return match.group(1).upper()


def extract_participant_from_filename(path: Path) -> str:
    match = SID_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not extract a 4-digit SID from filename: {path.name}")
    return match.group(1)


def build_session_id(path: Path, rater: str) -> str:
    stem = path.stem
    stem = re.sub(rf"(?<![A-Za-z]){re.escape(rater)}(?![A-Za-z])", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[\(\)_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or path.stem


def extract_ella_participant_from_filename(path: Path) -> str:
    match = ELLA_PARTICIPANT_PATTERN.search(path.name)
    if not match:
        raise ValueError(
            f"Could not extract Ella participant name Chidera, Lily, or Katie from filename: {path.name}"
        )
    return match.group(1).strip().title()


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def compute_ita(l_value: float, b_value: float) -> float:
    if pd.isna(l_value) or pd.isna(b_value) or b_value == 0:
        return np.nan
    return float(np.degrees(np.arctan((l_value - 50.0) / b_value)))


def load_single_raw_file(path: Path, *, include_participant: bool) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type for {path.name}. Expected CSV or Excel.")

    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    rater = extract_rater_from_filename(path)
    konica = df.copy()
    konica["source_file"] = path.name
    konica["rater"] = rater
    konica["session_id"] = build_session_id(path, rater)
    konica["body_site"] = konica["Group"].astype(str).str.strip()
    konica["body_site"] = konica["body_site"].replace(BODY_SITE_NORMALIZATION)
    konica["L*"] = pd.to_numeric(konica["L*"], errors="coerce")
    konica["b*"] = pd.to_numeric(konica["b*"], errors="coerce")
    konica = konica.dropna(subset=["body_site", "L*", "b*"])
    konica = konica[konica["body_site"] != ""].copy()
    konica["ita"] = konica.apply(lambda row: compute_ita(row["L*"], row["b*"]), axis=1)

    if include_participant:
        konica["participant"] = extract_participant_from_filename(path)

    return konica


def select_median_ita_rows(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    scored = df.copy()
    scored["_median_ita"] = scored.groupby(group_columns)["ita"].transform("median")
    scored["_distance"] = (scored["ita"] - scored["_median_ita"]).abs()
    scored = scored.sort_values(group_columns + ["_distance", "ita"], kind="mergesort")
    reduced = (
        scored.groupby(group_columns, as_index=False)
        .head(1)
        .drop(columns=["_median_ita", "_distance"])
        .reset_index(drop=True)
    )
    reduced["measurement"] = reduced["ita"]
    return reduced


def participant_files(files: list[Path]) -> list[Path]:
    return [
        file
        for file in files
        if "monk scale" not in file.name.lower() and "test with" not in file.name.lower()
    ]


def monk_files(files: list[Path]) -> list[Path]:
    return [file for file in files if "monk scale" in file.name.lower()]


def ella_files(files: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for file in files:
        if "test with" not in file.name.lower():
            continue
        if extract_rater_from_filename(file) != "EB":
            continue
        selected.append(file)
    return selected


def load_participant_data(files: list[Path]) -> pd.DataFrame:
    frames = [load_single_raw_file(file, include_participant=True) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    return select_median_ita_rows(combined, ["participant", "rater", "body_site"])


def load_monk_data(files: list[Path]) -> pd.DataFrame:
    frames = [load_single_raw_file(file, include_participant=False) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    reduced = select_median_ita_rows(combined, ["session_id", "rater", "body_site"])
    reduced["body_site"] = pd.Categorical(reduced["body_site"], categories=MONK_GROUP_ORDER, ordered=True)
    return reduced.sort_values(["rater", "session_id", "body_site"]).reset_index(drop=True)


def load_ella_data(files: list[Path]) -> pd.DataFrame:
    frames = [load_single_raw_file(file, include_participant=False) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    participant_lookup = {
        file.name: extract_ella_participant_from_filename(file)
        for file in files
    }
    combined["ella_participant"] = combined["source_file"].map(participant_lookup)
    reduced = select_median_ita_rows(combined, ["session_id", "rater", "body_site"])
    reduced["ella_participant"] = reduced["source_file"].map(participant_lookup)
    present_order = [site for site in BODY_SITE_PLOT_ORDER if site in reduced["body_site"].unique()]
    reduced["body_site"] = pd.Categorical(reduced["body_site"], categories=present_order, ordered=True)
    return reduced.sort_values(["session_id", "body_site"]).reset_index(drop=True)


def fit_mixed_model(df: pd.DataFrame, formula: str, group_column: str) -> tuple:
    _, exog = patsy.dmatrices(formula, data=df, return_type="dataframe")
    keep_columns = [column for column in exog.columns if column == "Intercept" or exog[column].nunique() > 1]
    dropped_columns = [column for column in exog.columns if column not in keep_columns]
    exog = exog[keep_columns]
    endog = df["measurement"]
    model = sm.MixedLM(endog=endog, exog=exog, groups=df[group_column])
    result = model.fit(reml=False, method="lbfgs", disp=False)
    return result, dropped_columns


def fit_participant_models(df: pd.DataFrame) -> tuple:
    model_main, main_dropped = fit_mixed_model(df, MAIN_MODEL_FORMULA, "participant")
    model_interaction, interaction_dropped = fit_mixed_model(df, INTERACTION_MODEL_FORMULA, "participant")
    return model_main, main_dropped, model_interaction, interaction_dropped


def build_pairwise_dataset(
    df: pd.DataFrame,
    *,
    index_columns: list[str],
    pair_order: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    wide = df.pivot_table(index=index_columns, columns="rater", values="measurement", aggfunc="first").reset_index()
    rater_columns = [column for column in wide.columns if column not in index_columns]
    if pair_order is None:
        pair_order = list(itertools.combinations(sorted(rater_columns), 2))

    pair_frames: list[pd.DataFrame] = []
    for rater_a, rater_b in pair_order:
        if rater_a not in rater_columns or rater_b not in rater_columns:
            continue
        subset = wide[index_columns + [rater_a, rater_b]].dropna()
        if subset.empty:
            continue
        pair = subset.rename(columns={rater_a: "measurement_a", rater_b: "measurement_b"}).copy()
        pair["rater_pair"] = f"{rater_a} vs {rater_b}"
        pair["mean_measurement"] = (pair["measurement_a"] + pair["measurement_b"]) / 2
        pair["difference"] = pair["measurement_a"] - pair["measurement_b"]
        pair_frames.append(pair)

    if not pair_frames:
        raise ValueError("No paired measurements were found for the requested dataset.")
    return pd.concat(pair_frames, ignore_index=True)


def summarize_pairwise_differences(pairwise_df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        pairwise_df.groupby(group_columns)
        .agg(
            n_pairs=("difference", "size"),
            mean_difference=("difference", "mean"),
            sd_difference=("difference", "std"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def repeated_measures_limits(pair_df: pd.DataFrame, cluster_column: str) -> tuple[float, float, float]:
    try:
        model = smf.mixedlm("difference ~ 1", data=pair_df, groups=pair_df[cluster_column]).fit(reml=True)
        bias = float(model.fe_params["Intercept"])
        random_var = float(model.cov_re.iloc[0, 0]) if model.cov_re.size else 0.0
        residual_var = float(model.scale)
        total_sd = float(np.sqrt(max(random_var + residual_var, 0.0)))
    except Exception:
        bias = float(pair_df["difference"].mean())
        total_sd = float(pair_df["difference"].std(ddof=1))
        if np.isnan(total_sd):
            total_sd = 0.0
    loa_lower = bias - 1.96 * total_sd
    loa_upper = bias + 1.96 * total_sd
    return bias, loa_lower, loa_upper


def format_bland_altman_axis_labels(rater_pair: str) -> tuple[str, str]:
    rater_a, rater_b = [part.strip() for part in rater_pair.split("vs")]
    return f"({rater_a} + {rater_b}) / 2", f"{rater_a} - {rater_b}"


def format_bland_altman_title(section_label: str, rater_pair: str) -> str:
    return f"{section_label}: {rater_pair} ITA"


def make_bland_altman_plots(
    pairwise_df: pd.DataFrame,
    *,
    output_dir: Path,
    filename_prefix: str,
    section_label: str,
    cluster_column: str,
    hue_column: str,
    hue_order: list[str],
    palette: dict[str, str] | None = None,
    style_column: str | None = None,
    markers: dict[str, str] | None = None,
) -> None:
    sns.set_theme(style="whitegrid")
    for stale_path in output_dir.glob(f"{filename_prefix}_*.png"):
        stale_path.unlink()

    plot_limits: list[float] = []
    mean_limits: list[float] = []
    for _, pair_df in pairwise_df.groupby("rater_pair"):
        bias, loa_lower, loa_upper = repeated_measures_limits(pair_df, cluster_column)
        plot_limits.extend([pair_df["difference"].min(), pair_df["difference"].max(), bias, loa_lower, loa_upper])
        mean_limits.extend([pair_df["mean_measurement"].min(), pair_df["mean_measurement"].max()])

    y_min = min(plot_limits)
    y_max = max(plot_limits)
    y_padding = max((y_max - y_min) * 0.05, 1.0)
    x_min = min(mean_limits)
    x_max = max(mean_limits)
    x_padding = max((x_max - x_min) * 0.05, 1.0)

    for rater_pair, pair_df in pairwise_df.groupby("rater_pair"):
        bias, loa_lower, loa_upper = repeated_measures_limits(pair_df, cluster_column)
        x_label, y_label = format_bland_altman_axis_labels(rater_pair)
        plot_df = pair_df.copy()
        plot_df[hue_column] = pd.Categorical(plot_df[hue_column], categories=hue_order, ordered=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter_kwargs = {
            "data": plot_df,
            "x": "mean_measurement",
            "y": "difference",
            "hue": hue_column,
            "hue_order": hue_order,
            "s": 80,
            "ax": ax,
        }
        if palette is not None:
            scatter_kwargs["palette"] = palette
        if style_column is not None:
            scatter_kwargs["style"] = style_column
            scatter_kwargs["style_order"] = hue_order
        if markers is not None:
            scatter_kwargs["markers"] = markers
        sns.scatterplot(**scatter_kwargs)

        ax.axhline(bias, color="black", linestyle="-", linewidth=1.5, label="Bias")
        ax.axhline(loa_lower, color="firebrick", linestyle="--", linewidth=1.2, label="Lower LoA")
        ax.axhline(loa_upper, color="firebrick", linestyle="--", linewidth=1.2, label="Upper LoA")
        ax.set_title(format_bland_altman_title(section_label, rater_pair), fontsize=16, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        annotation_x = x_max + x_padding - (x_max - x_min + 2 * x_padding) * 0.02
        annotation_box = {
            "facecolor": "white",
            "edgecolor": "0.7",
            "boxstyle": "round,pad=0.3",
            "alpha": 0.9,
        }
        ax.text(annotation_x, bias, f"Mean diff = {bias:.2f}", ha="right", va="bottom", fontsize=11, bbox=annotation_box)
        ax.text(
            annotation_x,
            loa_upper,
            f"Upper LoA = {loa_upper:.2f}",
            ha="right",
            va="bottom",
            fontsize=11,
            bbox=annotation_box,
        )
        ax.text(
            annotation_x,
            loa_lower,
            f"Lower LoA = {loa_lower:.2f}",
            ha="right",
            va="top",
            fontsize=11,
            bbox=annotation_box,
        )
        ax.tick_params(axis="both", labelsize=12)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        output_path = output_dir / f"{filename_prefix}_{safe_slug(rater_pair)}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_model_summary(model, destination: Path) -> None:
    destination.write_text(model.summary().as_text())


def write_preprocessed_data(df: pd.DataFrame, destination: Path) -> None:
    preferred_columns = [
        "participant",
        "session_id",
        "source_file",
        "rater",
        "body_site",
        "measurement",
        "ita",
        "L*",
        "b*",
    ]
    leading = [column for column in preferred_columns if column in df.columns]
    trailing = [column for column in df.columns if column not in leading]
    df.loc[:, leading + trailing].to_csv(destination, index=False)


def format_display_label(value: str) -> str:
    label = value.replace("_", " ")
    replacements = {
        "mean difference": "mean ITA difference",
        "sd difference": "SD of ITA difference",
        "mean ita": "mean ITA",
        "sd ita": "SD of ITA",
        "within group sd": "within-group SD",
    }
    return replacements.get(label, label)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")

    headers = [format_display_label(str(column)) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def write_markdown_file(destination: Path, content: str) -> None:
    destination.write_text(content.rstrip() + "\n")


def format_p_value(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def build_data_overview_markdown(df: pd.DataFrame) -> str:
    participants = df["participant"].nunique()
    raters = ", ".join(sorted(df["rater"].unique()))
    body_sites = ", ".join(sorted(df["body_site"].unique()))
    return (
        f"Unique participants count: {participants}\n\n"
        f"Included raters: {raters}\n\n"
        f"Included body sites: {body_sites}"
    )


def build_main_model_markdown(model) -> str:
    eb_term = "C(rater, Treatment(reference='KH'))[T.EB]"
    rvz_term = "C(rater, Treatment(reference='KH'))[T.RVZ]"
    eb_coef = float(model.fe_params[eb_term])
    eb_p = float(model.pvalues[eb_term])
    rvz_coef = float(model.fe_params[rvz_term])
    rvz_p = float(model.pvalues[rvz_term])
    return (
        "By fitting a linear mixed-effects model "
        f"(`{MAIN_MODEL_FORMULA} + (1 | participant)`), EB vs KH had coefficient {eb_coef:.2f} "
        f"({format_p_value(eb_p)}), and RVZ vs KH had coefficient {rvz_coef:.2f} "
        f"({format_p_value(rvz_p)})."
    )


def build_interaction_model_markdown(model) -> str:
    eb_term = "C(rater, Treatment(reference='KH'))[T.EB]"
    rvz_term = "C(rater, Treatment(reference='KH'))[T.RVZ]"
    eb_coef = float(model.fe_params[eb_term])
    eb_p = float(model.pvalues[eb_term])
    rvz_coef = float(model.fe_params[rvz_term])
    rvz_p = float(model.pvalues[rvz_term])
    return (
        "By fitting a linear mixed-effects model "
        f"(`{INTERACTION_MODEL_FORMULA} + (1 | participant)`), EB vs KH had coefficient {eb_coef:.2f} "
        f"({format_p_value(eb_p)}), and RVZ vs KH had coefficient {rvz_coef:.2f} "
        f"({format_p_value(rvz_p)})."
    )


def build_pairwise_markdown(summary_df: pd.DataFrame, section_by: str | None = None) -> str:
    if section_by is None:
        return markdown_table(summary_df.reset_index(drop=True), digits=3)

    sections = []
    for value, group_df in summary_df.groupby(section_by):
        sections.extend([f"### {value}", "", markdown_table(group_df.reset_index(drop=True), digits=3), ""])
    return "\n".join(sections)


def build_bland_altman_markdown(output_dir: Path, filename_prefix: str) -> str:
    lines = []
    for plot_path in sorted(output_dir.glob(f"{filename_prefix}_*.png")):
        label = plot_path.stem.replace(f"{filename_prefix}_", "").replace("_vs_", " vs ").upper()
        lines.extend([f"### {label}", "", f"![]({plot_path.as_posix()})", ""])
    return "\n".join(lines)


def summarize_monk_intra_operator(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for rater, rater_df in df.groupby("rater"):
        centered = rater_df["measurement"] - rater_df.groupby("body_site")["measurement"].transform("mean")
        within_group_sd = centered.std(ddof=1)
        repeatability = 1.96 * np.sqrt(2) * within_group_sd if pd.notna(within_group_sd) else np.nan
        summary_rows.append(
            {
                "rater": rater,
                "n_sessions": rater_df["session_id"].nunique(),
                "n_groups": rater_df["body_site"].nunique(),
                "n_measurements": len(rater_df),
                "mean_ita": rater_df["measurement"].mean(),
                "sd_ita": rater_df["measurement"].std(ddof=1),
                "within_group_sd": within_group_sd,
                "repeatability_coefficient": repeatability,
            }
        )
    return pd.DataFrame(summary_rows).sort_values("rater").reset_index(drop=True)


def build_monk_intra_markdown(df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    file_count = df["source_file"].nunique()
    lines = [
        "Within-rater variation is summarized after removing the fixed Monk Scale level effect within each rater.",
        "The within-group SD is the standard deviation of `ITA - mean(ITA within rater and Group)`.",
        "The repeatability coefficient is `1.96 * sqrt(2) * within-group SD`.",
        "",
        markdown_table(summary_df, digits=3),
    ]
    if file_count == 3:
        lines.extend(
            [
                "",
                "Note: the current `data/` directory contains 3 Monk Scale files total, one per rater.",
                "That means the present report uses 10 median ITA values per rater, not 30.",
                "The code supports multiple Monk Scale sessions per rater if more files are added later.",
            ]
        )
    return "\n".join(lines)


def summarize_ella_intra_operator(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("body_site")
        .agg(
            n_sessions=("session_id", "nunique"),
            mean_ita=("measurement", "mean"),
            sd_ita=("measurement", "std"),
            min_ita=("measurement", "min"),
            max_ita=("measurement", "max"),
        )
        .reset_index()
        .sort_values("body_site")
    )


def make_ella_intra_plot(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = df.copy()
    present_participants = [name for name in ELLA_PARTICIPANT_ORDER if name in plot_df["ella_participant"].unique()]
    plot_df["ella_participant"] = pd.Categorical(
        plot_df["ella_participant"],
        categories=present_participants,
        ordered=True,
    )
    summary = (
        plot_df.groupby("body_site")
        .agg(mean_ita=("measurement", "mean"), sd_ita=("measurement", "std"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=plot_df,
        x="body_site",
        y="measurement",
        hue="ella_participant",
        style="ella_participant",
        hue_order=present_participants,
        style_order=present_participants,
        palette=ELLA_PARTICIPANT_COLORS,
        markers=ELLA_PARTICIPANT_MARKERS,
        s=90,
        ax=ax,
    )
    x_positions = np.arange(len(summary))
    ax.errorbar(
        x=x_positions,
        y=summary["mean_ita"],
        yerr=summary["sd_ita"].fillna(0.0),
        fmt="_",
        color="black",
        elinewidth=2,
        capsize=6,
        markersize=18,
        label="Mean ± SD",
    )
    ax.set_title("Ella EB Median ITA by Body Site", fontsize=15, fontweight="bold")
    ax.set_xlabel("Body site", fontsize=13, fontweight="bold")
    ax.set_ylabel("ITA", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="best", frameon=True, title="Participant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_ella_intra_markdown(summary_df: pd.DataFrame, plot_path: Path) -> str:
    return "\n".join(
        [
            "For Ella EB measurements, the most useful view is body-site specific repeatability rather than a single overall SD.",
            "The table reports the 3 median ITA values per body site through their mean, SD, and range; the plot shows the individual medians with mean ± SD overlay.",
            "",
            markdown_table(summary_df, digits=3),
            "",
            f"![]({plot_path.as_posix()})",
        ]
    )


def analyze_participants(files: list[Path], output_dir: Path) -> None:
    df = load_participant_data(files)
    model_main, _, model_interaction, _ = fit_participant_models(df)
    pairwise_df = build_pairwise_dataset(
        df,
        index_columns=["participant", "body_site"],
        pair_order=[("KH", "EB"), ("KH", "RVZ")],
    )
    summary_df = summarize_pairwise_differences(pairwise_df, ["body_site", "rater_pair"])
    present_body_sites = [site for site in BODY_SITE_PLOT_ORDER if site in pairwise_df["body_site"].unique()]
    marker_map = {site: BODY_SITE_MARKERS[site] for site in present_body_sites}

    write_preprocessed_data(df, output_dir / "participant_preprocessed_measurements.csv")
    write_model_summary(model_main, output_dir / "participant_model_main_effects_summary.txt")
    write_model_summary(model_interaction, output_dir / "participant_model_interaction_summary.txt")
    summary_df.to_csv(output_dir / "participant_pairwise_difference_summary.csv", index=False)
    make_bland_altman_plots(
        pairwise_df,
        output_dir=output_dir,
        filename_prefix="participant_bland_altman",
        section_label="Study Participant Bland-Altman Plot",
        cluster_column="participant",
        hue_column="body_site",
        hue_order=present_body_sites,
        palette=None,
        style_column="body_site",
        markers=marker_map,
    )
    write_markdown_file(output_dir / "report_participant_data_overview.md", build_data_overview_markdown(df))
    write_markdown_file(output_dir / "report_participant_model_main.md", build_main_model_markdown(model_main))
    write_markdown_file(
        output_dir / "report_participant_model_interaction.md",
        build_interaction_model_markdown(model_interaction),
    )
    write_markdown_file(
        output_dir / "report_participant_pairwise_summary.md",
        build_pairwise_markdown(summary_df, section_by="body_site"),
    )
    write_markdown_file(
        output_dir / "report_participant_bland_altman.md",
        build_bland_altman_markdown(output_dir, "participant_bland_altman"),
    )


def analyze_monk(files: list[Path], output_dir: Path) -> None:
    df = load_monk_data(files)
    pairwise_df = build_pairwise_dataset(df, index_columns=["session_id", "body_site"])
    summary_df = summarize_pairwise_differences(pairwise_df, ["rater_pair"])
    intra_df = summarize_monk_intra_operator(df)

    write_preprocessed_data(df, output_dir / "monk_preprocessed_measurements.csv")
    intra_df.to_csv(output_dir / "monk_intra_operator_summary.csv", index=False)
    summary_df.to_csv(output_dir / "monk_pairwise_difference_summary.csv", index=False)
    make_bland_altman_plots(
        pairwise_df,
        output_dir=output_dir,
        filename_prefix="monk_bland_altman",
        section_label="Monk Scale Bland-Altman Plot",
        cluster_column="body_site",
        hue_column="body_site",
        hue_order=MONK_GROUP_ORDER,
        palette=MONK_COLORS,
        style_column=None,
        markers=None,
    )
    write_markdown_file(output_dir / "report_monk_intra_operator.md", build_monk_intra_markdown(df, intra_df))
    write_markdown_file(output_dir / "report_monk_pairwise_summary.md", build_pairwise_markdown(summary_df))
    write_markdown_file(
        output_dir / "report_monk_bland_altman.md",
        build_bland_altman_markdown(output_dir, "monk_bland_altman"),
    )


def analyze_ella(files: list[Path], output_dir: Path) -> None:
    df = load_ella_data(files)
    summary_df = summarize_ella_intra_operator(df)
    plot_path = output_dir / "ella_intra_operator_variation.png"

    write_preprocessed_data(df, output_dir / "ella_preprocessed_measurements.csv")
    summary_df.to_csv(output_dir / "ella_intra_operator_summary.csv", index=False)
    make_ella_intra_plot(df, plot_path)
    write_markdown_file(output_dir / "report_ella_intra_operator.md", build_ella_intra_markdown(summary_df, plot_path))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list_raw_files(args.input_path)

    participant_input = participant_files(files)
    monk_input = monk_files(files)
    ella_input = ella_files(files)

    if not participant_input:
        raise ValueError("No participant files were found after excluding Monk Scale and Ella files.")
    if not monk_input:
        raise ValueError("No Monk Scale files were found.")
    if not ella_input:
        raise ValueError("No Ella EB files were found.")

    analyze_participants(participant_input, args.output_dir)
    analyze_monk(monk_input, args.output_dir)
    analyze_ella(ella_input, args.output_dir)


if __name__ == "__main__":
    main()
