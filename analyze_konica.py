from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


RATER_PATTERN = re.compile(r"(?<![A-Za-z])(KH|EB|RVZ|LER|PE)(?![A-Za-z])", re.IGNORECASE)
DATE_PATTERN = re.compile(r"(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<year>\d{2,4})")
SID_PATTERN = re.compile(r"SID\s*([0-9]{3,4})", re.IGNORECASE)
LEGACY_SUBJECT_ID_PATTERN = re.compile(r"Subject\s+\d+\s+\(([0-9]{3,4})\)", re.IGNORECASE)
OPERATOR_SUFFIX_PATTERN = re.compile(r"(KH|EB|RVZ|LER|PE)\)?(?:\s*\(\d+\))?\s*$", re.IGNORECASE)
TRIPLICATE_FILENAME_PATTERN = re.compile(
    r"(?:\d{1,2}-\d{1,2}-\d{2,4}\s+)?Subject\s*(\d+)(?:\s+|_)(EB|KH|RVZ)(?:\s+|_)(\d+)$",
    re.IGNORECASE,
)
UGANDA_FILENAME_PATTERN = re.compile(r"Subject\s*(\d+)_(EB|PE)_(\d+)$", re.IGNORECASE)
REQUIRED_RAW_COLUMNS = ("Group", "L*", "b*")
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
PARTICIPANT_BODY_SITE_ORDER = ["Inner Upper Arm", "Dorsal", "Forehead", "Palmar"]
UGANDA_BODY_SITE_ORDER = ["Inner Upper Arm", "Dorsal", "Forehead", "Back of Hand", "Palmar"]
PREVIOUS_KM_BODY_SITE_ORDER = ["Fingernail", "Dorsal", "Inner Upper Arm", "Forehead", "Palmar"]
EQUIOX_BODY_SITE_ORDER = ["Dorsal", "Inner Upper Arm", "Forehead"]
BODY_SITE_PLOT_ORDER = ["Arm", "Chest", "Dorsal", "Ear", "Forehead", "Palmar"]
BODY_SITE_MARKERS = {
    "Fingernail": "o",
    "Inner Upper Arm": "o",
    "Chest": "s",
    "Dorsal": "^",
    "Back of Hand": "D",
    "Ear": "D",
    "Forehead": "P",
    "Palmar": "X",
}
BODY_SITE_COLORS = {
    "Fingernail": "#4daf4a",
    "Inner Upper Arm": "#1f78b4",
    "Dorsal": "#e31a1c",
    "Forehead": "#33a02c",
    "Back of Hand": "#ff7f00",
    "Palmar": "#6a3d9a",
}
TRIPLICATE_OPERATOR_ORDER = ["EB", "KH", "RVZ"]
UGANDA_OPERATOR_ORDER = ["EB", "PE"]
FRED_OPERATOR_ORDER = ["RB", "FB"]
EQUIOX_INTER_OPERATOR_ORDER = ["CC", "EB", "LO", "RVZ", "SE"]
EQUIOX_VARIABLE_CHANGE_ORDER = ["Control", "Pressure: Hard", "Lifiting"]
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
FRED_FILENAME_PATTERN = re.compile(r"(.+?)_(rb|fb)_([a-z]+)$", re.IGNORECASE)
EQUIOX_TEST_FILENAME_PATTERN = re.compile(r"Test\s+(\d+)$", re.IGNORECASE)
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
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
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if not files:
        raise ValueError(f"No CSV or Excel files found in directory: {path}")
    return deduplicate_raw_files(files)


def extract_rater_from_filename(path: Path) -> str:
    match = OPERATOR_SUFFIX_PATTERN.search(path.stem)
    if not match:
        match = RATER_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not extract rater code KH, EB, RVZ, or LER from filename: {path.name}")
    return match.group(1).upper()


def extract_participant_from_filename(path: Path) -> str:
    match = SID_PATTERN.search(path.name)
    if match:
        return match.group(1)
    match = LEGACY_SUBJECT_ID_PATTERN.search(path.name)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract participant SID/ID from filename: {path.name}")


def extract_date_from_filename(path: Path) -> str:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not extract date from filename: {path.name}")
    month = int(match.group("month"))
    day = int(match.group("day"))
    year_text = match.group("year")
    year = int(year_text)
    if len(year_text) == 2:
        year += 2000
    return f"{year:04d}-{month:02d}-{day:02d}"


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


def read_simple_xlsx_rows(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as workbook:
        shared_strings: list[str] = []
        shared_path = "xl/sharedStrings.xml"
        if shared_path in workbook.namelist():
            root = ET.fromstring(workbook.read(shared_path))
            for item in root.findall("a:si", XLSX_NS):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS)))

        root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        sheet_rows = root.findall(".//a:sheetData/a:row", XLSX_NS)
        records: list[dict[str, str]] = []
        headers: list[str] = []
        expected_columns = ["A", "B", "C", "D", "E", "F", "G"]

        for row_index, row in enumerate(sheet_rows):
            values: dict[str, str] = {}
            for cell in row.findall("a:c", XLSX_NS):
                cell_ref = cell.get("r", "")
                column = "".join(character for character in cell_ref if character.isalpha())
                cell_type = cell.get("t")
                raw_value = cell.find("a:v", XLSX_NS)
                value = ""
                if cell_type == "s" and raw_value is not None:
                    value = shared_strings[int(raw_value.text)]
                elif raw_value is not None and raw_value.text is not None:
                    value = raw_value.text
                values[column] = value

            ordered_values = [values.get(column, "") for column in expected_columns]
            if row_index == 0:
                headers = ordered_values
                continue
            record = dict(zip(headers, ordered_values))
            if record.get("Test"):
                records.append(record)

    return records


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
    konica["operator"] = rater
    konica["session_id"] = build_session_id(path, rater)
    konica["body_site"] = konica["Group"].astype(str).str.strip()
    konica["body_site"] = konica["body_site"].replace(BODY_SITE_NORMALIZATION)
    konica["L*"] = pd.to_numeric(konica["L*"], errors="coerce")
    konica["b*"] = pd.to_numeric(konica["b*"], errors="coerce")
    konica = konica.dropna(subset=["body_site", "L*", "b*"])
    konica = konica[konica["body_site"] != ""].copy()
    konica["ita"] = konica.apply(lambda row: compute_ita(row["L*"], row["b*"]), axis=1)

    if include_participant:
        sid = extract_participant_from_filename(path)
        konica["participant"] = sid
        konica["SID"] = sid
        konica["Date"] = extract_date_from_filename(path)

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
    excluded_directory_tokens = {
        "triplicates",
        "ella and philip ita repeatability",
        "ella and katie and rene ita repeatability",
        "rene and lea ita repeatability",
        "fred and ronald ita repeatability",
        "equiox ita repeatability",
        "ita measurement trials",
    }
    return [
        file
        for file in files
        if "subject" in file.name.lower()
        and not any(
            any(token in part.lower() for token in excluded_directory_tokens)
            for part in file.parts
        )
    ]


def participant_reference_rater(raters: list[str]) -> str:
    for candidate in ["KH", "LER", "EB", "RVZ"]:
        if candidate in raters:
            return candidate
    return sorted(raters)[0]


def participant_pair_order(raters: list[str]) -> list[tuple[str, str]]:
    excluded_pairs = {frozenset({"EB", "LER"}), frozenset({"EB", "RVZ"})}
    return [
        pair
        for pair in itertools.combinations(sorted(raters), 2)
        if frozenset(pair) not in excluded_pairs
    ]


def participant_model_formulas(reference_rater: str) -> tuple[str, str]:
    main_formula = f"measurement ~ C(rater, Treatment(reference='{reference_rater}')) + C(body_site)"
    interaction_formula = f"measurement ~ C(rater, Treatment(reference='{reference_rater}')) * C(body_site)"
    return main_formula, interaction_formula


def monk_files(files: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for file in files:
        if "monk scale" not in file.name.lower():
            continue
        try:
            rater = extract_rater_from_filename(file)
        except ValueError:
            continue
        if rater in {"KH", "EB", "RVZ"}:
            selected.append(file)
    return selected


def ella_files(files: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for file in files:
        if "test with" not in file.name.lower():
            continue
        if extract_rater_from_filename(file) != "EB":
            continue
        selected.append(file)
    return selected


def triplicates_files(files: list[Path]) -> list[Path]:
    selected_roots = {"triplicates", "ella and katie and rene ita repeatability"}
    return [
        file
        for file in files
        if any(part.lower() in selected_roots for part in file.parts) and TRIPLICATE_FILENAME_PATTERN.search(file.stem)
    ]


def uganda_files(files: list[Path]) -> list[Path]:
    return [
        file
        for file in files
        if "Ella and Philip ITA Repeatability" in file.parts and UGANDA_FILENAME_PATTERN.search(file.stem)
    ]


def uganda_files_from_root(root: Path) -> list[Path]:
    uganda_root = root / "Ella and Philip ITA Repeatability"
    if not uganda_root.exists():
        return []
    return sorted(
        file for file in uganda_root.glob("*.csv") if file.is_file() and UGANDA_FILENAME_PATTERN.search(file.stem)
    )


def fred_files(files: list[Path]) -> list[Path]:
    return [
        file
        for file in files
        if "Fred and Ronald ITA Repeatability" in file.parts and FRED_FILENAME_PATTERN.search(file.stem)
    ]


def equiox_files(files: list[Path]) -> list[Path]:
    return [
        file
        for file in files
        if "EquiOx ITA Repeatability" in file.parts and EQUIOX_TEST_FILENAME_PATTERN.search(file.stem)
    ]


def load_participant_data(files: list[Path]) -> pd.DataFrame:
    frames = [load_single_raw_file(file, include_participant=True) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["body_site"].isin(PARTICIPANT_BODY_SITE_ORDER)].copy()
    return select_median_ita_rows(combined, ["body_site", "Date", "SID", "operator"])


def extract_triplicates_metadata(path: Path) -> tuple[str, str, int]:
    match = TRIPLICATE_FILENAME_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Could not extract triplicates metadata from filename: {path.name}")
    subject = f"Subject{int(match.group(1))}"
    operator = match.group(2).upper()
    repeat_version = int(match.group(3))
    return subject, operator, repeat_version


def load_repeatability_data(
    files: list[Path],
    *,
    filename_pattern: re.Pattern[str],
    body_site_order: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in files:
        match = filename_pattern.search(file.stem)
        if not match:
            raise ValueError(f"Could not extract repeatability metadata from filename: {file.name}")
        subject = f"Subject{int(match.group(1))}"
        operator = match.group(2).upper()
        repeat_version = int(match.group(3))
        df = pd.read_csv(file)
        missing = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{file.name} is missing required columns: {', '.join(missing)}")

        trip = df.copy()
        trip["source_file"] = file.name
        trip["subject"] = subject
        trip["participant"] = subject
        trip["operator"] = operator
        trip["rater"] = operator
        trip["repeat_version"] = repeat_version
        trip["body_site"] = trip["Group"].astype(str).str.strip().replace(BODY_SITE_NORMALIZATION)
        trip["L*"] = pd.to_numeric(trip["L*"], errors="coerce")
        trip["b*"] = pd.to_numeric(trip["b*"], errors="coerce")
        trip = trip.dropna(subset=["body_site", "L*", "b*"])
        trip = trip[trip["body_site"].isin(body_site_order)].copy()
        trip["ita"] = trip.apply(lambda row: compute_ita(row["L*"], row["b*"]), axis=1)
        frames.append(trip)

    combined = pd.concat(frames, ignore_index=True)
    reduced = select_median_ita_rows(combined, ["body_site", "subject", "operator", "repeat_version"])
    reduced["body_site"] = pd.Categorical(reduced["body_site"], categories=body_site_order, ordered=True)
    return reduced.sort_values(["subject", "body_site", "operator", "repeat_version"]).reset_index(drop=True)


def load_triplicates_data(files: list[Path]) -> pd.DataFrame:
    return load_repeatability_data(
        files,
        filename_pattern=TRIPLICATE_FILENAME_PATTERN,
        body_site_order=PARTICIPANT_BODY_SITE_ORDER,
    )


def load_uganda_data(files: list[Path]) -> pd.DataFrame:
    return load_repeatability_data(
        files,
        filename_pattern=UGANDA_FILENAME_PATTERN,
        body_site_order=UGANDA_BODY_SITE_ORDER,
    )


def load_fred_data(files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in files:
        match = FRED_FILENAME_PATTERN.match(file.stem)
        if not match:
            continue
        test_id, operator, subject = match.groups()
        operator = operator.upper()
        if operator not in {"RB", "FB"}:
            continue
        df = pd.read_csv(file)
        missing = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{file.name} is missing required columns: {', '.join(missing)}")

        fred = df.copy()
        fred["source_file"] = file.name
        fred["subject"] = subject.upper()
        fred["participant"] = subject.upper()
        fred["operator"] = operator
        fred["rater"] = operator
        fred["test_id"] = test_id
        fred["body_site"] = fred["Group"].astype(str).str.strip().replace(BODY_SITE_NORMALIZATION)
        fred["L*"] = pd.to_numeric(fred["L*"], errors="coerce")
        fred["b*"] = pd.to_numeric(fred["b*"], errors="coerce")
        fred = fred.dropna(subset=["body_site", "L*", "b*"])
        fred = fred[fred["body_site"].isin(PREVIOUS_KM_BODY_SITE_ORDER)].copy()
        fred["ita"] = fred.apply(lambda row: compute_ita(row["L*"], row["b*"]), axis=1)
        frames.append(fred)

    combined = pd.concat(frames, ignore_index=True)
    reduced = select_median_ita_rows(combined, ["body_site", "subject", "operator", "test_id"])
    reduced["body_site"] = pd.Categorical(reduced["body_site"], categories=PREVIOUS_KM_BODY_SITE_ORDER, ordered=True)
    reduced = reduced.sort_values(["subject", "body_site", "operator"]).reset_index(drop=True)
    paired_subjects = (
        reduced.groupby("subject")["operator"].nunique().loc[lambda series: series >= 2].index.tolist()
    )
    return reduced[reduced["subject"].isin(paired_subjects)].reset_index(drop=True)


def load_equiox_data(files: list[Path], metadata_path: Path) -> pd.DataFrame:
    metadata_rows = read_simple_xlsx_rows(metadata_path)
    metadata_by_test: dict[int, dict[str, str]] = {}
    for row in metadata_rows:
        test_id = int(float(row["Test"]))
        metadata_by_test[test_id] = {
            "subject": row["Subject"].strip().upper(),
            "operator": row["Operator"].strip().upper(),
            "pressure": row["Pressure"].strip(),
            "lifting": row["Lifting"].strip(),
            "condition": row["Variable Changes (compares to Control)"].strip(),
            "date_label": row["Date / Time"].strip(),
        }

    frames: list[pd.DataFrame] = []
    for file in files:
        match = EQUIOX_TEST_FILENAME_PATTERN.search(file.stem)
        if not match:
            continue
        test_id = int(match.group(1))
        if test_id not in metadata_by_test:
            continue
        metadata = metadata_by_test[test_id]
        df = pd.read_csv(file)
        missing = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{file.name} is missing required columns: {', '.join(missing)}")

        equiox = df.copy()
        equiox["source_file"] = file.name
        equiox["test_id"] = test_id
        equiox["subject"] = metadata["subject"]
        equiox["participant"] = metadata["subject"]
        equiox["operator"] = metadata["operator"]
        equiox["rater"] = metadata["operator"]
        equiox["pressure"] = metadata["pressure"]
        equiox["lifting"] = metadata["lifting"]
        equiox["condition"] = metadata["condition"]
        equiox["date_label"] = metadata["date_label"]
        equiox["body_site"] = equiox["Group"].astype(str).str.strip().replace(BODY_SITE_NORMALIZATION)
        equiox["L*"] = pd.to_numeric(equiox["L*"], errors="coerce")
        equiox["b*"] = pd.to_numeric(equiox["b*"], errors="coerce")
        equiox = equiox.dropna(subset=["body_site", "L*", "b*"])
        equiox = equiox[equiox["body_site"].isin(PREVIOUS_KM_BODY_SITE_ORDER)].copy()
        equiox["ita"] = equiox.apply(lambda row: compute_ita(row["L*"], row["b*"]), axis=1)
        frames.append(equiox)

    combined = pd.concat(frames, ignore_index=True)
    reduced = select_median_ita_rows(combined, ["body_site", "test_id", "subject", "operator", "condition"])
    reduced["body_site"] = pd.Categorical(reduced["body_site"], categories=PREVIOUS_KM_BODY_SITE_ORDER, ordered=True)
    return reduced.sort_values(["test_id", "body_site", "operator"]).reset_index(drop=True)


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


def select_full_rank_columns(exog: pd.DataFrame) -> tuple[list[str], list[str]]:
    keep_columns: list[str] = []
    current_rank = 0

    for column in exog.columns:
        candidate_columns = keep_columns + [column]
        candidate_rank = int(np.linalg.matrix_rank(exog[candidate_columns].to_numpy()))
        if candidate_rank > current_rank:
            keep_columns.append(column)
            current_rank = candidate_rank

    dropped_columns = [column for column in exog.columns if column not in keep_columns]
    return keep_columns, dropped_columns


def fit_mixed_model(df: pd.DataFrame, formula: str, group_column: str) -> tuple:
    _, exog = patsy.dmatrices(formula, data=df, return_type="dataframe")
    varying_columns = [column for column in exog.columns if column == "Intercept" or exog[column].nunique() > 1]
    exog = exog[varying_columns]
    keep_columns, dropped_columns = select_full_rank_columns(exog)
    exog = exog[keep_columns]
    endog = df["measurement"]
    model = sm.MixedLM(endog=endog, exog=exog, groups=df[group_column])
    result = model.fit(reml=False, method="lbfgs", disp=False)
    return result, dropped_columns


def fit_participant_models(df: pd.DataFrame, reference_rater: str) -> tuple:
    main_formula, interaction_formula = participant_model_formulas(reference_rater)
    model_main, main_dropped = fit_mixed_model(df, main_formula, "participant")
    model_interaction, interaction_dropped = fit_mixed_model(df, interaction_formula, "participant")
    return model_main, main_dropped, model_interaction, interaction_dropped, main_formula, interaction_formula


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
    legend_kwargs: dict | None = None,
    annotate_column: str | None = None,
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

        if annotate_column is not None:
            for _, row in plot_df.iterrows():
                ax.annotate(
                    str(row[annotate_column]),
                    (row["mean_measurement"], row["difference"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                )

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
        final_legend_kwargs = {"loc": "best", "frameon": True}
        if legend_kwargs is not None:
            final_legend_kwargs.update(legend_kwargs)
        ax.legend(**final_legend_kwargs)
        fig.tight_layout()

        output_path = output_dir / f"{filename_prefix}_{safe_slug(rater_pair)}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_model_summary(model, destination: Path) -> None:
    destination.write_text(model.summary().as_text())


def write_preprocessed_data(df: pd.DataFrame, destination: Path) -> None:
    preferred_columns = [
        "participant",
        "SID",
        "Date",
        "operator",
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
    rows = [[str(value) for value in row] for row in formatted.astype(object).values.tolist()]
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
    operator_column = "operator" if "operator" in df.columns else "rater"
    raters = ", ".join(sorted(df[operator_column].unique()))
    body_sites = ", ".join(sorted(df["body_site"].unique()))
    lines = [
        f"Unique participants count: {participants}",
        f"Included operators: {raters}",
        f"Included body sites: {body_sites}",
    ]
    return "\n\n".join(lines)


def build_model_markdown(model, formula: str, reference_rater: str) -> str:
    comparisons: list[str] = []
    prefix = f"C(rater, Treatment(reference='{reference_rater}'))[T."
    for term, coefficient in model.fe_params.items():
        if not term.startswith(prefix) or ":" in term:
            continue
        comparison_rater = term.removeprefix(prefix).removesuffix("]")
        p_value = float(model.pvalues[term])
        comparisons.append(
            f"{comparison_rater} vs {reference_rater} had coefficient {float(coefficient):.2f} ({format_p_value(p_value)})"
        )

    if not comparisons:
        return (
            "By fitting a linear mixed-effects model "
            f"(`{formula} + (1 | participant)`), no rater contrast terms remained after removing non-varying columns."
        )

    return (
        "By fitting a linear mixed-effects model "
        f"(`{formula} + (1 | participant)`), " + ", and ".join(comparisons) + "."
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


def make_ella_intra_plot(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = df.copy()
    present_participants = [name for name in ELLA_PARTICIPANT_ORDER if name in plot_df["ella_participant"].unique()]
    plot_df["ella_participant"] = pd.Categorical(
        plot_df["ella_participant"],
        categories=present_participants,
        ordered=True,
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
    ax.set_xlabel("Body site", fontsize=13, fontweight="bold")
    ax.set_ylabel("ITA", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="best", frameon=True, title="Participant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_ella_intra_markdown(plot_path: Path) -> str:
    return "\n".join(
        [
            "The current Ella EB sessions come from different participants, so across-participant mean and SD by body site are not reported here.",
            "This section shows the individual participant-level median ITA values by body site. Per-participant summary statistics can be added later as more repeated sessions become available.",
            "",
            f"![]({plot_path.as_posix()})",
        ]
    )


def summarize_triplicates_intra_operator(
    df: pd.DataFrame,
    *,
    operator_order: list[str] = TRIPLICATE_OPERATOR_ORDER,
    by_body_site: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_df = (
        df.groupby(["subject", "body_site", "operator"])
        .agg(
            n_repeats=("measurement", "size"),
            mean_median_ita=("measurement", "mean"),
            cell_sd=("measurement", "std"),
        )
        .reset_index()
        .sort_values(["subject", "body_site", "operator"])
        .reset_index(drop=True)
    )

    summary_rows: list[dict] = []
    summary_groups = ["operator", "body_site"] if by_body_site else ["operator"]
    for group_values, group_df in cell_df.groupby(summary_groups, observed=True):
        if by_body_site:
            operator, body_site = group_values
        else:
            operator = group_values
            body_site = None

        valid = group_df["cell_sd"].dropna()
        pooled_within_sd = float(np.sqrt(np.mean(np.square(valid)))) if not valid.empty else np.nan
        repeatability = 1.96 * np.sqrt(2) * pooled_within_sd if pd.notna(pooled_within_sd) else np.nan
        row = {
            "operator": operator,
            "n_subject_site_cells": int(group_df["cell_sd"].notna().sum()),
            "pooled_within_sd": pooled_within_sd,
            "repeatability_coefficient": repeatability,
        }
        if by_body_site:
            row["body_site"] = body_site
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["operator"] = pd.Categorical(summary_df["operator"], categories=operator_order, ordered=True)
        if by_body_site:
            summary_df["body_site"] = pd.Categorical(
                summary_df["body_site"], categories=PARTICIPANT_BODY_SITE_ORDER, ordered=True
            )
            summary_df = summary_df.sort_values(["operator", "body_site"]).reset_index(drop=True)
            summary_df = summary_df[
                [
                    "operator",
                    "body_site",
                    "n_subject_site_cells",
                    "pooled_within_sd",
                    "repeatability_coefficient",
                ]
            ]
        else:
            summary_df = summary_df.sort_values("operator").reset_index(drop=True)

    return cell_df, summary_df


def build_triplicates_flagged_cells(df: pd.DataFrame, cell_df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index=["subject", "body_site", "operator"],
            columns="repeat_version",
            values="measurement",
            aggfunc="first",
        )
        .reset_index()
    )
    version_numbers = [int(column) for column in wide.columns if isinstance(column, (int, np.integer))]
    version_columns = {column: f"median_ita_v{int(column)}" for column in wide.columns if isinstance(column, (int, np.integer))}
    wide = wide.rename(columns=version_columns)
    flagged = cell_df[cell_df["cell_sd"] > threshold].copy()
    if flagged.empty:
        return flagged

    merged = flagged.merge(wide, on=["subject", "body_site", "operator"], how="left")
    version_labels = [f"median_ita_v{version}" for version in sorted(version_numbers)]
    desired_columns = ["subject", "body_site", "operator", "cell_sd", *version_labels]
    present_columns = [column for column in desired_columns if column in merged.columns]
    return merged[present_columns].sort_values(["subject", "body_site", "operator"]).reset_index(drop=True)


def make_triplicates_heatmap(
    cell_df: pd.DataFrame,
    output_path: Path,
    operator_order: list[str],
) -> None:
    heatmap_df = cell_df.copy()
    subject_order = sorted(heatmap_df["subject"].unique(), key=lambda value: int(re.search(r"\d+", value).group(0)))
    body_site_order = sorted(heatmap_df["body_site"].unique())
    heatmap_df = heatmap_df.sort_values(["subject", "body_site", "operator"]).reset_index(drop=True)
    heatmap_df["subject_group"] = heatmap_df["subject"].astype(str) + " | " + heatmap_df["body_site"].astype(str)
    subject_group_order = [
        f"{subject} | {body_site}"
        for subject in subject_order
        for body_site in body_site_order
        if ((heatmap_df["subject"] == subject) & (heatmap_df["body_site"] == body_site)).any()
    ]

    matrix = (
        heatmap_df.pivot(index="subject_group", columns="operator", values="cell_sd")
        .reindex(index=subject_group_order)
        .reindex(columns=operator_order)
    )
    annot = matrix.apply(lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value:.2f}"))

    fig_height = max(4.5, len(matrix) * 0.45)
    fig, ax = plt.subplots(figsize=(6.5, fig_height))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "SD of median ITA"},
        ax=ax,
    )
    ax.set_xlabel("Operator", fontsize=12, fontweight="bold")
    ax.set_ylabel("Subject | Body site", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_repeatability_heatmap(
    cell_df: pd.DataFrame,
    *,
    output_path: Path,
    body_site_order: list[str],
    operator_order: list[str],
) -> None:
    heatmap_df = cell_df.copy()
    heatmap_df["body_site"] = pd.Categorical(
        heatmap_df["body_site"], categories=body_site_order, ordered=True
    )
    subject_order = sorted(heatmap_df["subject"].unique(), key=lambda value: int(re.search(r"\d+", value).group(0)))
    heatmap_df["subject"] = pd.Categorical(heatmap_df["subject"], categories=subject_order, ordered=True)
    heatmap_df = heatmap_df.sort_values(["subject", "body_site", "operator"]).reset_index(drop=True)
    heatmap_df["subject_group"] = heatmap_df["subject"].astype(str) + " | " + heatmap_df["body_site"].astype(str)

    matrix = heatmap_df.pivot(index="subject_group", columns="operator", values="cell_sd").reindex(columns=operator_order)
    annot = matrix.apply(lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value:.2f}"))

    fig_height = max(4.5, len(matrix) * 0.45)
    fig, ax = plt.subplots(figsize=(6.5, fig_height))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "SD of median ITA"},
        ax=ax,
    )
    ax.set_xlabel("Operator", fontsize=12, fontweight="bold")
    ax.set_ylabel("Subject | Body site", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_triplicates_markdown(summary_df: pd.DataFrame, heatmap_path: Path, flagged_df: pd.DataFrame) -> str:
    lines = [
        "Each triplicates file is first reduced to one median ITA per body site.",
        "For each Subject x Body site x Operator cell, the SD is then calculated from the 3 session-level median ITA values.",
        "The pooled within-subject SD and repeatability coefficient are reported separately for each Operator x Body site. The repeatability coefficient is `1.96 * sqrt(2) * pooled within-SD`.",
        "",
        markdown_table(summary_df, digits=3),
        "",
        f"![]({heatmap_path.as_posix()})",
    ]
    if flagged_df.empty:
        lines.extend(["", "No Subject x Body site x Operator cells had SD of median ITA greater than 5."])
    else:
        lines.extend(
            [
                "",
                "## Cells With SD of Median ITA Greater Than 5",
                "",
                markdown_table(flagged_df, digits=3),
            ]
        )
    return "\n".join(lines)


def build_uganda_intra_markdown(summary_df: pd.DataFrame, heatmap_path: Path, flagged_df: pd.DataFrame) -> str:
    lines = [
        markdown_table(summary_df, digits=3),
        "",
        f"![]({heatmap_path.as_posix()})",
    ]
    if flagged_df.empty:
        lines.extend(["", "No Subject x Body site x Operator cells had SD of median ITA greater than 5."])
    else:
        lines.extend(
            [
                "",
                "## Cells With SD of Median ITA Greater Than 5",
                "",
                markdown_table(flagged_df, digits=3),
            ]
        )
    return "\n".join(lines)


def build_uganda_inter_markdown(summary_df: pd.DataFrame, output_dir: Path) -> str:
    by_site_df = summary_df[summary_df["body_site"] != "Overall"].reset_index(drop=True)
    lines = [
        markdown_table(by_site_df, digits=3),
        "",
        build_bland_altman_markdown(output_dir, "uganda_bland_altman").rstrip(),
    ]
    return "\n".join(lines)


def build_triplicates_inter_markdown(summary_df: pd.DataFrame, output_dir: Path) -> str:
    by_site_df = summary_df[summary_df["body_site"] != "Overall"].reset_index(drop=True)
    lines = [
        "Each file is reduced to one median ITA per body site. For each Subject x Body site x Operator cell, the median of the 3 round-level median ITA values is used for the inter-operator comparison.",
        "",
        markdown_table(by_site_df, digits=3),
        "",
        build_bland_altman_markdown(output_dir, "triplicates_inter_operator_bland_altman").rstrip(),
    ]
    return "\n".join(lines)


def build_uganda_raw_data_markdown(df: pd.DataFrame) -> str:
    raw_df = df.copy()
    raw_df["subject_sort"] = raw_df["subject"].str.extract(r"(\d+)").astype(int)
    raw_df = raw_df.sort_values(["subject_sort", "body_site", "operator", "repeat_version"]).reset_index(drop=True)
    display_df = raw_df[
        ["subject", "operator", "repeat_version", "body_site", "L*", "a*", "b*", "measurement"]
    ].rename(
        columns={
            "subject": "Subject #",
            "operator": "Operator",
            "repeat_version": "Round #",
            "body_site": "Group",
            "L*": "L*",
            "a*": "a*",
            "b*": "b*",
            "measurement": "ITA",
        }
    )
    return markdown_table(display_df, digits=3)


def summarize_uganda_inter_operator(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    by_site = summarize_pairwise_differences(pairwise_df, ["body_site", "rater_pair"])
    overall = (
        pairwise_df.groupby("rater_pair")
        .agg(
            n_pairs=("difference", "size"),
            mean_difference=("difference", "mean"),
            sd_difference=("difference", "std"),
        )
        .reset_index()
    )
    overall.insert(0, "body_site", "Overall")
    summary_df = pd.concat([overall, by_site], ignore_index=True)
    summary_df["body_site"] = pd.Categorical(
        summary_df["body_site"],
        categories=["Overall", *UGANDA_BODY_SITE_ORDER],
        ordered=True,
    )
    return summary_df.sort_values(["body_site", "rater_pair"]).reset_index(drop=True)


def collapse_uganda_inter_operator(df: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        df.groupby(["subject", "body_site", "operator", "rater"], as_index=False)
        .agg(
            measurement=("measurement", "median"),
            ita=("ita", "median"),
            **{
                "L*": ("L*", "median"),
                "b*": ("b*", "median"),
            },
        )
        .sort_values(["subject", "body_site", "operator"])
        .reset_index(drop=True)
    )
    collapsed["participant"] = collapsed["subject"]
    collapsed["repeat_version"] = "collapsed_median"
    collapsed["source_file"] = "collapsed_across_versions"
    return collapsed


def collapse_equiox_inter_operator(df: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        df.groupby(["subject", "body_site", "operator", "rater"], as_index=False)
        .agg(
            measurement=("measurement", "median"),
            ita=("ita", "median"),
            **{
                "L*": ("L*", "median"),
                "a*": ("a*", "median"),
                "b*": ("b*", "median"),
            },
        )
        .sort_values(["body_site", "operator"])
        .reset_index(drop=True)
    )
    collapsed["test_id"] = "collapsed_median"
    collapsed["source_file"] = "collapsed_across_tests"
    return collapsed


def add_replicate_index(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    ranked = df.sort_values(sort_columns).copy()
    ranked["replicate_index"] = ranked.groupby(group_columns).cumcount() + 1
    return ranked


def summarize_pairwise_by_site_only(pairwise_df: pd.DataFrame, body_site_order: list[str]) -> pd.DataFrame:
    summary_df = summarize_pairwise_differences(pairwise_df, ["body_site", "rater_pair"])
    summary_df["body_site"] = pd.Categorical(summary_df["body_site"], categories=body_site_order, ordered=True)
    return summary_df.sort_values(["body_site", "rater_pair"]).reset_index(drop=True)


def build_previous_km_inter_markdown(summary_df: pd.DataFrame, output_dir: Path, filename_prefix: str) -> str:
    display_df = summary_df.drop(columns=["sd_difference"]).rename(columns={"mean_difference": "ITA difference"})
    return "\n".join(
        [
            markdown_table(display_df, digits=3),
            "",
            build_bland_altman_markdown(output_dir, filename_prefix).rstrip(),
        ]
    )


def analyze_participants(files: list[Path], output_dir: Path) -> None:
    df = load_participant_data(files)
    raters = sorted(df["rater"].unique())
    reference_rater = participant_reference_rater(raters)
    pair_order = participant_pair_order(raters)
    model_main, _, model_interaction, _, main_formula, interaction_formula = fit_participant_models(
        df, reference_rater
    )
    pairwise_df = build_pairwise_dataset(
        df,
        index_columns=["body_site", "Date", "SID"],
        pair_order=pair_order,
    )
    summary_df = summarize_pairwise_differences(pairwise_df, ["body_site", "rater_pair"])
    present_body_sites = [site for site in PARTICIPANT_BODY_SITE_ORDER if site in pairwise_df["body_site"].unique()]
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
    write_markdown_file(
        output_dir / "report_participant_model_main.md",
        build_model_markdown(model_main, main_formula, reference_rater),
    )
    write_markdown_file(
        output_dir / "report_participant_model_interaction.md",
        build_model_markdown(model_interaction, interaction_formula, reference_rater),
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
        legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.12), "ncol": 5},
    )
    write_markdown_file(output_dir / "report_monk_intra_operator.md", build_monk_intra_markdown(df, intra_df))
    write_markdown_file(output_dir / "report_monk_pairwise_summary.md", build_pairwise_markdown(summary_df))
    write_markdown_file(
        output_dir / "report_monk_bland_altman.md",
        build_bland_altman_markdown(output_dir, "monk_bland_altman"),
    )


def analyze_ella(files: list[Path], output_dir: Path) -> None:
    df = load_ella_data(files)
    plot_path = output_dir / "ella_intra_operator_variation.png"
    summary_path = output_dir / "ella_intra_operator_summary.csv"

    write_preprocessed_data(df, output_dir / "ella_preprocessed_measurements.csv")
    if summary_path.exists():
        summary_path.unlink()
    make_ella_intra_plot(df, plot_path)
    write_markdown_file(output_dir / "report_ella_intra_operator.md", build_ella_intra_markdown(plot_path))


def analyze_triplicates(files: list[Path], output_dir: Path) -> None:
    df = load_triplicates_data(files)
    operator_order = sorted(df["operator"].unique())
    cell_df, summary_df = summarize_triplicates_intra_operator(
        df,
        operator_order=operator_order,
        by_body_site=True,
    )
    flagged_df = build_triplicates_flagged_cells(df, cell_df, threshold=5.0)
    heatmap_path = output_dir / "triplicates_intra_operator_heatmap.png"
    inter_df = collapse_uganda_inter_operator(df)
    pairwise_df = build_pairwise_dataset(
        inter_df,
        index_columns=["subject", "body_site"],
        pair_order=[("KH", "RVZ")],
    )
    inter_summary_df = summarize_uganda_inter_operator(pairwise_df)
    present_body_sites = [site for site in PARTICIPANT_BODY_SITE_ORDER if site in pairwise_df["body_site"].unique()]
    marker_map = {site: BODY_SITE_MARKERS[site] for site in present_body_sites}
    palette = {site: BODY_SITE_COLORS[site] for site in present_body_sites}

    write_preprocessed_data(df, output_dir / "triplicates_preprocessed_measurements.csv")
    write_preprocessed_data(inter_df, output_dir / "triplicates_inter_operator_collapsed_measurements.csv")
    cell_df.to_csv(output_dir / "triplicates_cell_level_sd.csv", index=False)
    summary_df.to_csv(output_dir / "triplicates_operator_summary.csv", index=False)
    flagged_df.to_csv(output_dir / "triplicates_flagged_cells_sd_gt_5.csv", index=False)
    pairwise_df.to_csv(output_dir / "triplicates_inter_operator_pairwise_measurements.csv", index=False)
    inter_summary_df.to_csv(output_dir / "triplicates_inter_operator_pairwise_difference_summary.csv", index=False)
    make_triplicates_heatmap(cell_df, heatmap_path, operator_order)
    make_bland_altman_plots(
        pairwise_df,
        output_dir=output_dir,
        filename_prefix="triplicates_inter_operator_bland_altman",
        section_label="ITA Measurement Trials 09-01-2026 Inter-operator Bland-Altman Plot",
        cluster_column="subject",
        hue_column="body_site",
        hue_order=present_body_sites,
        palette=palette,
        style_column="body_site",
        markers=marker_map,
        annotate_column="subject",
    )
    write_markdown_file(
        output_dir / "report_triplicates_intra_operator.md",
        build_triplicates_markdown(summary_df, heatmap_path, flagged_df),
    )
    write_markdown_file(
        output_dir / "report_triplicates_inter_operator.md",
        build_triplicates_inter_markdown(inter_summary_df, output_dir),
    )


def analyze_uganda(files: list[Path], output_dir: Path) -> None:
    df = load_uganda_data(files)
    intra_cell_df, intra_summary_df = summarize_triplicates_intra_operator(
        df, operator_order=UGANDA_OPERATOR_ORDER
    )
    intra_flagged_df = build_triplicates_flagged_cells(df, intra_cell_df, threshold=5.0)
    heatmap_path = output_dir / "uganda_intra_operator_heatmap.png"
    inter_df = collapse_uganda_inter_operator(df)
    pairwise_df = build_pairwise_dataset(
        inter_df,
        index_columns=["subject", "body_site"],
        pair_order=[("EB", "PE")],
    )
    inter_summary_df = summarize_uganda_inter_operator(pairwise_df)
    present_body_sites = [site for site in UGANDA_BODY_SITE_ORDER if site in pairwise_df["body_site"].unique()]
    marker_map = {site: BODY_SITE_MARKERS[site] for site in present_body_sites}
    palette = {site: BODY_SITE_COLORS[site] for site in present_body_sites}

    write_preprocessed_data(df, output_dir / "uganda_preprocessed_measurements.csv")
    write_preprocessed_data(inter_df, output_dir / "uganda_inter_operator_collapsed_measurements.csv")
    intra_cell_df.to_csv(output_dir / "uganda_cell_level_sd.csv", index=False)
    intra_summary_df.to_csv(output_dir / "uganda_operator_summary.csv", index=False)
    intra_flagged_df.to_csv(output_dir / "uganda_flagged_cells_sd_gt_5.csv", index=False)
    pairwise_df.to_csv(output_dir / "uganda_pairwise_measurements.csv", index=False)
    inter_summary_df.to_csv(output_dir / "uganda_pairwise_difference_summary.csv", index=False)
    make_repeatability_heatmap(
        intra_cell_df,
        output_path=heatmap_path,
        body_site_order=UGANDA_BODY_SITE_ORDER,
        operator_order=UGANDA_OPERATOR_ORDER,
    )
    make_bland_altman_plots(
        pairwise_df,
        output_dir=output_dir,
        filename_prefix="uganda_bland_altman",
        section_label="Uganda Inter-operator Bland-Altman Plot",
        cluster_column="subject",
        hue_column="body_site",
        hue_order=present_body_sites,
        palette=palette,
        style_column="body_site",
        markers=marker_map,
        annotate_column="subject",
    )
    write_markdown_file(
        output_dir / "report_uganda_intra_operator.md",
        build_uganda_intra_markdown(intra_summary_df, heatmap_path, intra_flagged_df),
    )
    write_markdown_file(
        output_dir / "report_uganda_inter_operator.md",
        build_uganda_inter_markdown(inter_summary_df, output_dir),
    )
    write_markdown_file(
        output_dir / "report_uganda_inter_operator_raw_data.md",
        build_uganda_raw_data_markdown(df),
    )


def analyze_fred(files: list[Path], output_dir: Path) -> None:
    df = load_fred_data(files)
    df = df[df["body_site"] != "Fingernail"].copy()
    pairwise_df = build_pairwise_dataset(df, index_columns=["subject", "body_site"], pair_order=[("RB", "FB")])
    fred_body_site_order = [site for site in PREVIOUS_KM_BODY_SITE_ORDER if site != "Fingernail"]
    summary_df = summarize_pairwise_by_site_only(pairwise_df, fred_body_site_order)
    present_body_sites = [site for site in fred_body_site_order if site in pairwise_df["body_site"].unique()]
    marker_map = {site: BODY_SITE_MARKERS[site] for site in present_body_sites}
    palette = {site: BODY_SITE_COLORS[site] for site in present_body_sites}

    write_preprocessed_data(df, output_dir / "fred_preprocessed_measurements.csv")
    pairwise_df.to_csv(output_dir / "fred_pairwise_measurements.csv", index=False)
    summary_df.to_csv(output_dir / "fred_pairwise_difference_summary.csv", index=False)
    make_bland_altman_plots(
        pairwise_df,
        output_dir=output_dir,
        filename_prefix="fred_bland_altman",
        section_label="Fred vs Ronald Bland-Altman Plot",
        cluster_column="subject",
        hue_column="body_site",
        hue_order=present_body_sites,
        palette=palette,
        style_column="body_site",
        markers=marker_map,
    )
    write_markdown_file(
        output_dir / "report_fred_inter_operator.md",
        build_previous_km_inter_markdown(summary_df, output_dir, "fred_bland_altman"),
    )


def analyze_equiox(files: list[Path], output_dir: Path) -> None:
    metadata_path = Path("data/EquiOx ITA Repeatability/Copy of Minolta Testing.xlsx")
    df = load_equiox_data(files, metadata_path)
    df = df[df["body_site"].isin(EQUIOX_BODY_SITE_ORDER)].copy()

    inter_df = df[
        (df["operator"].isin(EQUIOX_INTER_OPERATOR_ORDER))
        & (
            (df["operator"] != "CC")
            | (df["condition"] == "Control")
        )
    ].copy()
    inter_collapsed_df = collapse_equiox_inter_operator(inter_df)
    inter_pairwise_df = build_pairwise_dataset(
        inter_collapsed_df,
        index_columns=["subject", "body_site"],
        pair_order=list(itertools.combinations(EQUIOX_INTER_OPERATOR_ORDER, 2)),
    )
    inter_summary_df = summarize_pairwise_by_site_only(inter_pairwise_df, EQUIOX_BODY_SITE_ORDER)
    present_inter_sites = [site for site in EQUIOX_BODY_SITE_ORDER if site in inter_pairwise_df["body_site"].unique()]
    inter_marker_map = {site: BODY_SITE_MARKERS[site] for site in present_inter_sites}
    inter_palette = {site: BODY_SITE_COLORS[site] for site in present_inter_sites}

    variable_df = df[(df["operator"] == "CC") & (df["condition"].isin(EQUIOX_VARIABLE_CHANGE_ORDER))].copy()
    variable_df = add_replicate_index(
        variable_df,
        group_columns=["condition", "body_site"],
        sort_columns=["condition", "body_site", "test_id"],
    )
    variable_pair_input = variable_df.drop(columns=["rater"]).rename(columns={"condition": "rater"})
    variable_pairwise_df = build_pairwise_dataset(
        variable_pair_input,
        index_columns=["subject", "body_site", "replicate_index"],
        pair_order=[("Control", "Pressure: Hard"), ("Control", "Lifiting")],
    )
    variable_pairwise_df["rater_pair"] = variable_pairwise_df["rater_pair"].replace(
        {
            "Control vs Pressure: Hard": "Control vs Hard Pressure",
            "Control vs Lifiting": "Control vs Lifting",
        }
    )
    variable_summary_df = summarize_pairwise_by_site_only(variable_pairwise_df, EQUIOX_BODY_SITE_ORDER)
    present_variable_sites = [site for site in EQUIOX_BODY_SITE_ORDER if site in variable_pairwise_df["body_site"].unique()]
    variable_marker_map = {site: BODY_SITE_MARKERS[site] for site in present_variable_sites}
    variable_palette = {site: BODY_SITE_COLORS[site] for site in present_variable_sites}

    write_preprocessed_data(df, output_dir / "equiox_preprocessed_measurements.csv")
    write_preprocessed_data(inter_collapsed_df, output_dir / "equiox_inter_operator_preprocessed_measurements.csv")
    inter_pairwise_df.to_csv(output_dir / "equiox_inter_operator_pairwise_measurements.csv", index=False)
    inter_summary_df.to_csv(output_dir / "equiox_inter_operator_pairwise_difference_summary.csv", index=False)
    write_preprocessed_data(variable_df, output_dir / "equiox_variable_change_preprocessed_measurements.csv")
    variable_pairwise_df.to_csv(output_dir / "equiox_variable_change_pairwise_measurements.csv", index=False)
    variable_summary_df.to_csv(output_dir / "equiox_variable_change_pairwise_difference_summary.csv", index=False)

    make_bland_altman_plots(
        inter_pairwise_df,
        output_dir=output_dir,
        filename_prefix="equiox_inter_operator_bland_altman",
        section_label="EquiOx CRC Inter-operator Bland-Altman Plot",
        cluster_column="subject",
        hue_column="body_site",
        hue_order=present_inter_sites,
        palette=inter_palette,
        style_column="body_site",
        markers=inter_marker_map,
    )
    make_bland_altman_plots(
        variable_pairwise_df,
        output_dir=output_dir,
        filename_prefix="equiox_variable_change_bland_altman",
        section_label="EquiOx Condition Comparison Bland-Altman Plot",
        cluster_column="replicate_index",
        hue_column="body_site",
        hue_order=present_variable_sites,
        palette=variable_palette,
        style_column="body_site",
        markers=variable_marker_map,
    )
    write_markdown_file(
        output_dir / "report_equiox_inter_operator.md",
        build_previous_km_inter_markdown(inter_summary_df, output_dir, "equiox_inter_operator_bland_altman"),
    )
    write_markdown_file(
        output_dir / "report_equiox_variable_change.md",
        build_previous_km_inter_markdown(variable_summary_df, output_dir, "equiox_variable_change_bland_altman"),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list_raw_files(args.input_path)

    participant_input = participant_files(files)
    monk_input = monk_files(files)
    ella_input = ella_files(files)
    triplicates_input = triplicates_files(files)
    uganda_input = uganda_files_from_root(args.input_path)
    fred_input = fred_files(files)
    equiox_input = equiox_files(files)

    if not participant_input:
        raise ValueError("No participant files were found with 'Subject' in the filename.")
    if not monk_input:
        raise ValueError("No Monk Scale files were found.")
    if not ella_input:
        raise ValueError("No Ella EB files were found.")
    if not triplicates_input:
        raise ValueError("No triplicates files were found.")
    if not uganda_input:
        raise ValueError("No Uganda repeatability files were found.")
    if not fred_input:
        raise ValueError("No Fred and Ronald repeatability files were found.")
    if not equiox_input:
        raise ValueError("No EquiOx repeatability files were found.")

    analyze_participants(participant_input, args.output_dir)
    analyze_monk(monk_input, args.output_dir)
    analyze_ella(ella_input, args.output_dir)
    analyze_triplicates(triplicates_input, args.output_dir)
    analyze_uganda(uganda_input, args.output_dir)
    analyze_fred(fred_input, args.output_dir)
    analyze_equiox(equiox_input, args.output_dir)


if __name__ == "__main__":
    main()
