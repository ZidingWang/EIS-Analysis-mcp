import glob
import os
import re

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = (".csv", ".txt", ".tsv", ".xlsx", ".xls")


class EISData(object):
    def __init__(
        self,
        path,
        freq_hz,
        z,
        minus_z_imag=None,
        frame=None,
        columns=None,
        imag_multiplier_to_minus=None,
    ):
        self.path = path
        self.freq_hz = np.asarray(freq_hz, dtype=float)
        self.z = np.asarray(z, dtype=complex)
        if minus_z_imag is None:
            minus_z_imag = -np.imag(self.z)
        self.minus_z_imag = np.asarray(minus_z_imag, dtype=float)
        self.frame = frame
        self.columns = dict(columns or {})
        self.imag_multiplier_to_minus = imag_multiplier_to_minus

    def sorted(self, descending=True):
        order = np.argsort(self.freq_hz)
        if descending:
            order = order[::-1]
        frame = self.frame
        if frame is not None:
            frame = frame.iloc[order].reset_index(drop=True)
        return EISData(
            self.path,
            self.freq_hz[order],
            self.z[order],
            minus_z_imag=self.minus_z_imag[order],
            frame=frame,
            columns=self.columns,
            imag_multiplier_to_minus=self.imag_multiplier_to_minus,
        )


def read_eis_file(
    path,
    freq_col=None,
    real_col=None,
    imag_col=None,
    imag_sign=None,
    sheet_name=0,
):
    frame = _read_table(path, sheet_name=sheet_name)
    if _columns_look_like_data(frame.columns):
        frame = _read_table(path, sheet_name=sheet_name, header=None)

    detected = detect_eis_columns(frame, freq_col=freq_col, real_col=real_col, imag_col=imag_col)
    freq_col = detected["freq_col"]
    real_col = detected["real_col"]
    imag_col = detected["imag_col"]

    freq_values = pd.to_numeric(frame[freq_col], errors="coerce")
    real_values = pd.to_numeric(frame[real_col], errors="coerce")
    imag_values = pd.to_numeric(frame[imag_col], errors="coerce")

    raw = pd.DataFrame(
        {
            "freq_hz": freq_values,
            "z_real_ohm": real_values,
            "source_imag_ohm": imag_values,
        }
    ).dropna()
    raw = raw[raw["freq_hz"] > 0]
    if raw.empty:
        raise ValueError("No valid EIS rows found in %s" % path)

    if imag_sign is None:
        multiplier = _detect_imag_multiplier_to_minus(
            raw["freq_hz"].values,
            raw["source_imag_ohm"].values,
        )
    else:
        multiplier = float(imag_sign)

    minus_z_imag = raw["source_imag_ohm"].values * multiplier
    z_physical_imag = -minus_z_imag
    data = pd.DataFrame(
        {
            "freq_hz": raw["freq_hz"].values,
            "z_real_ohm": raw["z_real_ohm"].values,
            "z_imag_ohm": minus_z_imag,
            "z_imag_physical_ohm": z_physical_imag,
        }
    )
    data = data[data["freq_hz"] > 0]
    if data.empty:
        raise ValueError("No valid EIS rows found in %s" % path)

    z = data["z_real_ohm"].values + 1j * data["z_imag_physical_ohm"].values
    return EISData(
        path,
        data["freq_hz"].values,
        z,
        minus_z_imag=data["z_imag_ohm"].values,
        frame=data,
        columns=detected,
        imag_multiplier_to_minus=multiplier,
    ).sorted(descending=True)


def detect_eis_columns(frame, freq_col=None, real_col=None, imag_col=None):
    columns = list(frame.columns)
    requested = {
        "freq_col": _normalize_requested_column(columns, freq_col),
        "real_col": _normalize_requested_column(columns, real_col),
        "imag_col": _normalize_requested_column(columns, imag_col),
    }

    detected = dict(requested)
    used = set([value for value in detected.values() if value is not None])

    for key, role in (("freq_col", "freq"), ("real_col", "real"), ("imag_col", "imag")):
        if detected[key] is None:
            column = _find_column_by_name(columns, role, used)
            if column is not None:
                detected[key] = column
                used.add(column)

    if detected["freq_col"] is None:
        detected["freq_col"] = _guess_frequency_column(frame, used)
        used.add(detected["freq_col"])

    if detected["real_col"] is None or detected["imag_col"] is None:
        real_guess, imag_guess = _guess_impedance_columns(
            frame,
            detected["freq_col"],
            real_col=detected["real_col"],
            imag_col=detected["imag_col"],
        )
        if detected["real_col"] is None:
            detected["real_col"] = real_guess
        if detected["imag_col"] is None:
            detected["imag_col"] = imag_guess

    for key in ("freq_col", "real_col", "imag_col"):
        if detected[key] is None:
            raise KeyError("Could not detect %s. Available: %s" % (key, columns))

    if len(set(detected.values())) != 3:
        raise ValueError("Detected EIS columns must be different: %s" % detected)

    return detected


def expand_inputs(inputs, recursive=False):
    if isinstance(inputs, str):
        inputs = [inputs]
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            pattern = "**/*" if recursive else "*"
            for ext in SUPPORTED_EXTENSIONS:
                paths.extend(glob.glob(os.path.join(item, pattern + ext), recursive=recursive))
        else:
            matched = glob.glob(item, recursive=recursive)
            if matched:
                paths.extend(matched)
            else:
                paths.append(item)

    unique = []
    seen = set()
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path not in seen and os.path.isfile(abs_path):
            unique.append(abs_path)
            seen.add(abs_path)
    return unique


def _read_table(path, sheet_name=0, header=0):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name, header=header)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", header=header)
    if ext == ".txt":
        frame = pd.read_csv(path, sep=None, engine="python", header=header)
        if len(frame.columns) <= 1:
            frame = pd.read_csv(path, sep=r"\s+", engine="python", header=header)
        return frame
    return pd.read_csv(path, sep=None, engine="python", header=header)


def _find_column_by_name(columns, role, used=None):
    used = used or set()
    scored = []
    for column in columns:
        if column in used:
            continue
        score = _column_score(column, role)
        if score > 0:
            scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _normalize_requested_column(columns, requested):
    if requested is None or requested == "":
        return None
    if requested in columns:
        return requested
    if isinstance(requested, str):
        for column in columns:
            if str(column) == requested:
                return column
        try:
            index = int(requested)
        except ValueError:
            index = None
        if index is not None and 0 <= index < len(columns):
            return columns[index]
    raise KeyError("Column '%s' not found. Available: %s" % (requested, list(columns)))


def _column_score(column, role):
    raw = str(column).lower().strip()
    norm = re.sub(r"[^a-z0-9]+", "", raw)

    if role == "freq":
        if norm in ("f", "freq", "frequency", "frequencyhz", "freqhz", "hz"):
            return 10
        if "freq" in norm or norm.endswith("hz"):
            return 5
        return 0

    if role == "real":
        if norm in ("zreal", "zre", "rez", "realz", "zprime", "zp"):
            return 10
        if "real" in norm and "z" in norm:
            return 8
        if "zre" in norm or "rez" in norm:
            return 8
        if "z'" in raw or "z prime" in raw:
            return 7
        return 0

    if role == "imag":
        if norm in ("zimag", "zim", "zimg", "imz", "imagz", "zdoubleprime", "zpp"):
            return 10
        if "imag" in norm and "z" in norm:
            return 8
        if "zim" in norm or "imz" in norm:
            return 8
        if "z''" in raw or "z double" in raw:
            return 7
        return 0

    return 0


def _guess_frequency_column(frame, used):
    candidates = []
    for column in frame.columns:
        if column in used:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").values
        score = _frequency_score(values)
        if score > 0:
            candidates.append((score, column))
    if not candidates:
        raise KeyError("Could not detect frequency column from numeric values")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _guess_impedance_columns(frame, freq_col, real_col=None, imag_col=None):
    columns = list(frame.columns)
    numeric_columns = [
        column
        for column in columns
        if column != freq_col and _numeric_fraction(frame[column]) >= 0.6
    ]
    freq_index = columns.index(freq_col)
    after_freq = [column for column in numeric_columns if columns.index(column) > freq_index]
    ordered = after_freq + [column for column in numeric_columns if column not in after_freq]

    if real_col is not None and imag_col is not None:
        return real_col, imag_col

    if real_col is None and imag_col is None:
        if len(ordered) >= 2:
            return ordered[0], ordered[1]
        raise KeyError("Could not detect real/imag impedance columns")

    if real_col is None:
        real_index = columns.index(imag_col)
        before_imag = [
            column
            for column in ordered
            if column != imag_col and columns.index(column) < real_index
        ]
        if before_imag:
            return before_imag[-1], imag_col
        for column in ordered:
            if column != imag_col:
                return column, imag_col
        raise KeyError("Could not detect real impedance column")

    imag_index = columns.index(real_col)
    after_real = [
        column
        for column in ordered
        if column != real_col and columns.index(column) > imag_index
    ]
    if after_real:
        return real_col, after_real[0]
    for column in ordered:
        if column != real_col:
            return real_col, column
    raise KeyError("Could not detect imaginary impedance column")


def _frequency_score(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0

    positive = values[values > 0]
    positive_fraction = float(positive.size) / float(values.size)
    if positive.size < 3 or positive_fraction < 0.8:
        return 0.0

    score = 20.0 * positive_fraction
    min_value = float(np.nanmin(positive))
    max_value = float(np.nanmax(positive))
    if min_value <= 0 or max_value <= min_value:
        return 0.0

    decades = np.log10(max_value / min_value)
    score += min(decades, 8.0) * 6.0

    diffs = np.diff(positive)
    if np.all(diffs >= 0) or np.all(diffs <= 0):
        score += 20.0

    if 1e-6 <= min_value and max_value <= 1e8:
        score += 8.0
    if min_value <= 1.0 <= max_value:
        score += 8.0

    unique_fraction = float(np.unique(positive).size) / float(positive.size)
    score += 5.0 * unique_fraction
    return score


def _numeric_fraction(series):
    values = pd.to_numeric(series, errors="coerce")
    if len(values) == 0:
        return 0.0
    return float(values.notnull().sum()) / float(len(values))


def _columns_look_like_data(columns):
    if len(columns) < 3:
        return False
    numeric_count = 0
    for column in columns:
        try:
            value = float(column)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            numeric_count += 1
    return numeric_count >= min(3, len(columns))


def _detect_imag_multiplier_to_minus(freq_hz, imag_values):
    freq_hz = np.asarray(freq_hz, dtype=float)
    imag_values = np.asarray(imag_values, dtype=float)
    mask = np.isfinite(freq_hz) & np.isfinite(imag_values) & (freq_hz > 0)
    freq_hz = freq_hz[mask]
    imag_values = imag_values[mask]
    if imag_values.size == 0:
        return 1.0

    order = np.argsort(freq_hz)
    low_count = max(1, int(np.ceil(0.1 * imag_values.size)))
    low_count = min(max(low_count, 3), imag_values.size)
    low_freq_imag = imag_values[order[:low_count]]
    sign_value = float(np.nanmedian(low_freq_imag))
    if abs(sign_value) < 1e-30:
        sign_value = float(np.nanmedian(imag_values))
    if sign_value < 0:
        return -1.0
    return 1.0
