"""Local-only Konica Minolta ITA and repeatability dashboard.

The app intentionally has no database client or network calls. Uploaded files are
processed in memory for the active browser session only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BODY_SITE_NORMALIZATION = {
    "Palmer": "Palmar",
    "Arm": "Inner Upper Arm",
    "Arms": "Inner Upper Arm",
    "Inner arm": "Inner Upper Arm",
    "Inner upper arm": "Inner Upper Arm",
    "Inner upperarm": "Inner Upper Arm",
    "Back of hand": "Back of Hand",
    "Fingernail (A)": "Fingernail",
    "Dorsal (B)": "Dorsal",
    "Dorsal - DIP (B)": "Dorsal",
    "Palmar (C)": "Palmar",
    "Inner Arn (D)": "Inner Upper Arm",
    "Inner Upper Arm (D)": "Inner Upper Arm",
    "Forehead (E)": "Forehead",
    "Forehead (G)": "Forehead",
}
FILENAME_PATTERN = re.compile(
    r"(?:subject\s*(?P<subject>\d+)|test\s+with\s+(?P<name>.+?)(?:-\d+)?)"
    r"(?:\s|_)+\(?(?P<operator>[A-Za-z]+)\)?(?:\s|_)+(?P<repeat>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FileMetadata:
    subject: str
    operator: str
    repeat: int


def find_column(columns: pd.Index, *candidates: str) -> str | None:
    normalized = {str(column).strip().casefold(): str(column) for column in columns}
    for candidate in candidates:
        found = normalized.get(candidate.casefold())
        if found:
            return found
    return None


def deduplicate_uploads(uploaded_files: list[object]) -> list[object]:
    """Keep the first copy of each byte-identical CSV upload."""
    unique_files: list[object] = []
    seen_hashes: set[str] = set()
    for uploaded_file in uploaded_files:
        content_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_files.append(uploaded_file)
    return unique_files


def infer_metadata(filename: str) -> FileMetadata | None:
    stem = filename.rsplit(".", 1)[0]
    match = FILENAME_PATTERN.search(stem)
    if not match:
        return None
    subject = match.group("subject")
    if subject:
        subject = f"Subject {int(subject)}"
    else:
        subject = match.group("name").strip().title()
    return FileMetadata(subject, match.group("operator").upper(), int(match.group("repeat")))


def read_konica_file(uploaded_file: object, metadata: FileMetadata) -> pd.DataFrame:
    raw = pd.read_csv(uploaded_file)
    group_col = find_column(raw.columns, "Group", "group")
    l_col = find_column(raw.columns, "L*", "lab_l", "L")
    a_col = find_column(raw.columns, "a*", "lab_a", "a")
    b_col = find_column(raw.columns, "b*", "lab_b", "b")
    required_columns = (("Group", group_col), ("L*", l_col), ("b*", b_col))
    missing = [name for name, column in required_columns if not column]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    data = pd.DataFrame(
        {
            "source_file": uploaded_file.name,
            "subject": metadata.subject,
            "operator": metadata.operator,
            "repeat": metadata.repeat,
            "site": raw[group_col].astype("string").str.strip().replace(BODY_SITE_NORMALIZATION),
            "lab_l": pd.to_numeric(raw[l_col], errors="coerce"),
            "lab_b": pd.to_numeric(raw[b_col], errors="coerce"),
        }
    )
    data["lab_a"] = pd.to_numeric(raw[a_col], errors="coerce") if a_col else np.nan
    data = data.dropna(subset=["site", "lab_l", "lab_b"]).copy()
    data = data[data["site"] != ""].copy()
    data["ita"] = np.degrees(np.arctan2(data["lab_l"] - 50.0, data["lab_b"]))
    return data


def reduce_to_median_ita(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["source_file", "subject", "operator", "repeat", "site"], dropna=False)[
        "ita"
    ]
    return grouped.median().rename("median_ita").reset_index()


def summarize_repeatability(medians: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = (
        medians.groupby(["subject", "site", "operator"], dropna=False)["median_ita"]
        .agg(n_repeats="size", mean_median_ita="mean", cell_sd="std")
        .reset_index()
        .sort_values(["subject", "site", "operator"])
    )
    rows: list[dict[str, object]] = []
    for (operator, site), group in cells.groupby(["operator", "site"], dropna=False):
        valid_sd = group["cell_sd"].dropna()
        pooled_sd = float(np.sqrt(np.mean(np.square(valid_sd)))) if not valid_sd.empty else np.nan
        rows.append(
            {
                "operator": operator,
                "site": site,
                "n_subject_site_cells": int(valid_sd.size),
                "pooled_within_sd": pooled_sd,
                "repeatability_coefficient": 1.96 * np.sqrt(2) * pooled_sd
                if pd.notna(pooled_sd)
                else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["operator", "site"]).reset_index(drop=True)
    return cells.reset_index(drop=True), summary


def insufficient_triplicate_uploads(medians: pd.DataFrame) -> pd.DataFrame:
    counts = (
        medians.groupby(["subject", "operator"], dropna=False)["source_file"]
        .agg(
            uploaded_triplicate_files="nunique",
            uploaded_filenames=lambda filenames: ", ".join(sorted(set(filenames))),
        )
        .reset_index()
    )
    return counts[counts["uploaded_triplicate_files"] < 3].sort_values(["subject", "operator"])


def build_flagged_cells(medians: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    median_values = medians.pivot_table(
        index=["subject", "site", "operator"],
        columns="repeat",
        values="median_ita",
        aggfunc="first",
    ).reset_index()
    median_columns = [
        column for column in median_values.columns if isinstance(column, (int, np.integer))
    ]
    median_values = median_values.rename(
        columns={column: f"median_ita_round_{column}" for column in median_columns}
    )
    flagged = cells[cells["cell_sd"] > 5].merge(
        median_values,
        on=["subject", "site", "operator"],
        how="left",
    )
    return (
        flagged.drop(columns=["mean_median_ita"])
        .sort_values(["subject", "site", "operator"])
        .reset_index(drop=True)
    )


def round_for_display(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy().round(2)


def csv_bytes(data: pd.DataFrame) -> bytes:
    return round_for_display(data).to_csv(index=False).encode("utf-8")


def highlight_high_pooled_sd(value: float) -> str:
    if pd.notna(value) and value > 5:
        return "background-color: #f8d7da; color: #842029; font-weight: bold;"
    return ""


def highlight_high_repeatability_coefficient(value: float) -> str:
    if pd.notna(value) and value > 10:
        return "background-color: #f8d7da; color: #842029; font-weight: bold;"
    return ""


def subject_label(subject: str) -> str:
    """Show numeric Subject labels compactly while retaining participant-name labels."""
    match = re.fullmatch(r"Subject\s*(\d+)", str(subject), re.IGNORECASE)
    return match.group(1) if match else str(subject)


def subject_sort_key(subject: str) -> tuple[int, int, str]:
    """Keep Katie and Rene first, then sort numbered subjects numerically."""
    value = str(subject)
    preferred_order = {"katie": 0, "rene": 1}
    if value.casefold() in preferred_order:
        return (0, preferred_order[value.casefold()], value.casefold())
    match = re.fullmatch(r"Subject\s*(\d+)", value, re.IGNORECASE)
    if match:
        return (1, int(match.group(1)), value.casefold())
    return (2, 0, value.casefold())


def summarize_inter_operator(medians: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    collapsed = (
        medians.groupby(["subject", "site", "operator"], dropna=False)["median_ita"]
        .median()
        .rename("operator_median_ita")
        .reset_index()
    )
    wide = collapsed.pivot(
        index=["subject", "site"], columns="operator", values="operator_median_ita"
    )
    operators = sorted(collapsed["operator"].unique())
    pairs: list[pd.DataFrame] = []
    for index, operator_a in enumerate(operators):
        for operator_b in operators[index + 1 :]:
            paired = wide[[operator_a, operator_b]].dropna().reset_index()
            if paired.empty:
                continue
            paired = paired.rename(columns={operator_a: "ita_a", operator_b: "ita_b"})
            paired["operator_pair"] = f"{operator_a} vs {operator_b}"
            paired["mean_ita"] = (paired["ita_a"] + paired["ita_b"]) / 2
            paired["difference"] = paired["ita_a"] - paired["ita_b"]
            pairs.append(paired)
    return collapsed, pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()


def summarize_pairwise_agreement(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for grouping, group in pairwise.groupby(["operator_pair", "site"], dropna=False):
        operator_pair, site = grouping
        mean_difference = group["difference"].mean()
        sd_difference = group["difference"].std(ddof=1)
        rows.append(
            {
                "operator_pair": operator_pair,
                "site": site,
                "n_pairs": len(group),
                "mean_difference": mean_difference,
                "sd_difference": sd_difference,
                "loa_lower": mean_difference - 1.96 * sd_difference,
                "loa_upper": mean_difference + 1.96 * sd_difference,
            }
        )
    for operator_pair, group in pairwise.groupby("operator_pair", dropna=False):
        mean_difference = group["difference"].mean()
        sd_difference = group["difference"].std(ddof=1)
        rows.append(
            {
                "operator_pair": operator_pair,
                "site": "Overall",
                "n_pairs": len(group),
                "mean_difference": mean_difference,
                "sd_difference": sd_difference,
                "loa_lower": mean_difference - 1.96 * sd_difference,
                "loa_upper": mean_difference + 1.96 * sd_difference,
            }
        )
    return pd.DataFrame(rows).sort_values(["operator_pair", "site"]).reset_index(drop=True)


def show_repeatability(medians: pd.DataFrame) -> None:
    st.header("Intra-operator Variation")
    st.write(
        "For each subject x site x operator cell, the SD is calculated across repeated files. "
        "The repeatability coefficient shows the expected maximum difference between two repeated "
        "measurements under the same conditions (same operator and same site)."
    )
    cells, summary = summarize_repeatability(medians)
    st.caption("Summary by operator and site")
    operators = sorted(summary["operator"].unique())
    operator_columns = st.columns(len(operators))
    for column, operator in zip(operator_columns, operators):
        operator_summary = summary[summary["operator"] == operator][
            ["site", "n_subject_site_cells", "pooled_within_sd", "repeatability_coefficient"]
        ].rename(columns={"n_subject_site_cells": "n_subject_site_pair"})
        styled_summary = (
            operator_summary.style.map(
                highlight_high_pooled_sd,
                subset=["pooled_within_sd"],
            )
            .map(
                highlight_high_repeatability_coefficient,
                subset=["repeatability_coefficient"],
            )
            .format(precision=2, na_rep="")
        )
        with column:
            st.markdown(f"**{operator}**")
            st.dataframe(styled_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Pooled within SD greater than 5 and repeatability coefficient greater than 10 are "
        "flagged in red."
    )

    flagged = build_flagged_cells(medians, cells)
    if flagged.empty:
        return
    st.caption("Median ITA values for subject x operator pair with SD greater than 5")
    st.dataframe(round_for_display(flagged), use_container_width=True, hide_index=True)


def show_inter_operator_agreement(medians: pd.DataFrame) -> None:
    st.header("Inter-operator Agreement")
    st.write(
        "For each subject and site, each operator's triplicate median ITA is compared with every "
        "other operator. Bland-Altman differences are calculated as the first operator minus "
        "the second."
    )
    _, pairwise = summarize_inter_operator(medians)
    if pairwise.empty:
        st.warning("No paired subject-and-site measurements were found across operators.")
        return
    summary = summarize_pairwise_agreement(pairwise)
    st.caption("Pairwise agreement summary")
    st.dataframe(round_for_display(summary), use_container_width=True, hide_index=True)
    operator_pair = st.selectbox(
        "Bland-Altman operator pair", sorted(pairwise["operator_pair"].unique())
    )
    operator_a, operator_b = operator_pair.split(" vs ")
    plot_data = pairwise[pairwise["operator_pair"] == operator_pair].copy()
    plot_data["subject_label"] = plot_data["subject"].map(subject_label)
    bias = plot_data["difference"].mean()
    difference_sd = plot_data["difference"].std(ddof=1)
    plot = px.scatter(
        plot_data,
        x="mean_ita",
        y="difference",
        color="site",
        symbol="site",
        text="subject_label",
        hover_data={"subject": True, "site": True, "ita_a": ":.2f", "ita_b": ":.2f"},
        title=f"Bland-Altman agreement: {operator_pair}",
    )
    plot.add_hline(y=bias, line_dash="dash", line_color="black", annotation_text="Mean difference")
    if pd.notna(difference_sd):
        plot.add_hline(
            y=bias - 1.96 * difference_sd,
            line_dash="dot",
            line_color="gray",
            annotation_text="Lower 95% LoA",
        )
        plot.add_hline(
            y=bias + 1.96 * difference_sd,
            line_dash="dot",
            line_color="gray",
            annotation_text="Upper 95% LoA",
        )
    plot.update_traces(textposition="top center")
    plot.update_layout(
        xaxis_title=f"({operator_a} + {operator_b}) / 2 ITA (degrees)",
        yaxis_title=f"{operator_a} - {operator_b} ITA (degrees)",
        legend_title="Site",
    )
    st.plotly_chart(plot, use_container_width=True)


st.set_page_config(page_title="Konica Local Dashboard", layout="wide")
st.title("Konica Minolta Local Analysis Dashboard")
st.caption("Local-only: files are processed in memory and are never sent to the remote database.")

if "upload_widget_version" not in st.session_state:
    st.session_state.upload_widget_version = 0

if st.button("Clear uploaded files and restart"):
    st.session_state.upload_widget_version += 1
    st.rerun()

st.subheader("Please upload three sets of triplicates per participant and operator")
uploaded_files = st.file_uploader(
    "Konica Minolta CSV files",
    type=["csv"],
    accept_multiple_files=True,
    help=(
        "Upload raw Konica CSVs named like `9-02-2026 Subject 5 KH 1.csv`. "
        "The subject, operator, and triplicate number are read from the filename."
    ),
    key=f"konica_csv_upload_{st.session_state.upload_widget_version}",
)

if not uploaded_files:
    st.info("Upload one or more raw Konica CSV exports to calculate and plot ITA by site.")
    st.stop()
uploaded_files = deduplicate_uploads(uploaded_files)

frames: list[pd.DataFrame] = []
errors: list[str] = []
for uploaded_file in uploaded_files:
    metadata = infer_metadata(uploaded_file.name)
    if metadata is None:
        errors.append(
            f"{uploaded_file.name}: filename must include subject, operator, "
            "and triplicate number, "
            "for example `9-02-2026 Subject 5 KH 1.csv`."
        )
        continue
    try:
        frames.append(read_konica_file(uploaded_file, metadata))
    except (ValueError, pd.errors.ParserError) as error:
        errors.append(f"{uploaded_file.name}: {error}")

for error in errors:
    st.error(error)
if not frames:
    st.stop()

data = pd.concat(frames, ignore_index=True)
medians = reduce_to_median_ita(data)
insufficient_uploads = insufficient_triplicate_uploads(medians)
if not insufficient_uploads.empty:
    st.error(
        "At least three triplicate CSV files are required for every participant and operator. "
        "Please see table below for Subject # and operators that had incomplete data uploaded."
    )
    st.caption("Participants and operators with fewer than three uploaded triplicate CSV files")
    st.dataframe(
        insufficient_uploads.rename(
            columns={
                "subject": "Subject #",
                "operator": "Operator",
                "uploaded_filenames": "Uploaded triplicate CSV files",
            }
        )[["Subject #", "Operator", "Uploaded triplicate CSV files"]],
        use_container_width=True,
        hide_index=True,
    )
    st.stop()

st.caption(f"{len(uploaded_files)} uploaded file(s) | {len(data)} valid measurements")
with st.expander("View processed measurements"):
    subject_filter, operator_filter, site_filter = st.columns(3)
    with subject_filter:
        selected_subject = st.selectbox(
            "Subject #",
            ["All", *sorted(data["subject"].unique(), key=subject_sort_key)],
        )
    with operator_filter:
        selected_operator = st.selectbox("Operator", ["All", *sorted(data["operator"].unique())])
    with site_filter:
        selected_site = st.selectbox("Site", ["All", *sorted(data["site"].unique())])

    filtered_data = data.copy()
    if selected_subject != "All":
        filtered_data = filtered_data[filtered_data["subject"] == selected_subject]
    if selected_operator != "All":
        filtered_data = filtered_data[filtered_data["operator"] == selected_operator]
    if selected_site != "All":
        filtered_data = filtered_data[filtered_data["site"] == selected_site]
    subject_order = sorted(data["subject"].unique(), key=subject_sort_key)
    subject_rank = {subject: index for index, subject in enumerate(subject_order)}
    filtered_data = (
        filtered_data.assign(_subject_order=filtered_data["subject"].map(subject_rank))
        .sort_values(["_subject_order", "operator", "site", "repeat"])
        .drop(columns="_subject_order")
    )

    st.dataframe(round_for_display(filtered_data), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered processed ITA CSV",
        csv_bytes(filtered_data),
        "konica_ita_measurements.csv",
        "text/csv",
    )

show_repeatability(medians)
if medians["operator"].nunique() > 1:
    show_inter_operator_agreement(medians)
