import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .circuit import CircuitModel


ECM_MODEL_PRESETS = [
    ("R0", "R0", "Series ohmic resistance only."),
    ("R0-RC", "R0-(R1||C1)", "One ideal RC arc."),
    ("R0-RQ", "R0-(R1||CPE1)", "One depressed arc with a CPE."),
    ("R0-2RC", "R0-(R1||C1)-(R2||C2)", "Two ideal RC arcs."),
    ("R0-RQ-RC", "R0-(R1||CPE1)-(R2||C2)", "One CPE arc plus one ideal RC arc."),
    ("R0-2RQ", "R0-(R1||CPE1)-(R2||CPE2)", "Two depressed arcs with CPEs."),
    ("R0-RQ-W", "R0-(R1||CPE1)-W1", "One CPE arc plus Warburg diffusion."),
    ("R0-2RQ-W", "R0-(R1||CPE1)-(R2||CPE2)-W1", "Two CPE arcs plus Warburg diffusion."),
    ("L-R0-RQ", "L1-R0-(R1||CPE1)", "Series inductance plus one depressed arc."),
    ("L-R0-RC", "L1-R0-(R1||C1)", "Series inductance plus one ideal RC arc."),
    ("L-R0-2RC", "L1-R0-(R1||C1)-(R2||C2)", "Series inductance plus two ideal RC arcs."),
    ("L-R0-RQ-RC", "L1-R0-(R1||CPE1)-(R2||C2)", "Series inductance, one CPE arc, and one ideal RC arc."),
    ("L-R0-2RQ", "L1-R0-(R1||CPE1)-(R2||CPE2)", "Series inductance plus two depressed arcs."),
    ("L-R0-RWQ", "L1-R0-((R1-W1)||CPE1)", "Series inductance with an R-W series branch in parallel with one CPE."),
    ("L-R0-RQ-RWQ", "L1-R0-(R1||CPE1)-((R2-W1)||CPE2)", "Series inductance, one RQ arc, and one R-W series branch in parallel with a CPE."),
    ("L-R0-RQ-W", "L1-R0-((R1-W1)||CPE1)", "Backward-compatible alias for L-R0-RWQ."),
    ("L-R0-2RQ-W", "L1-R0-(R1||CPE1)-((R2-W1)||CPE2)", "Backward-compatible alias for L-R0-RQ-RWQ."),
]

PUBLIC_ECM_MODEL_IDS = (
    "L-R0-RWQ",
    "L-R0-RQ",
    "L-R0-RC",
    "L-R0-2RC",
    "L-R0-2RQ",
    "L-R0-RQ-RWQ",
)


class ECMConfig(object):
    def __init__(
        self,
        model="R0-RQ",
        initial=None,
        bounds=None,
        weighting="modulus",
        max_nfev=20000,
        strict_bounds=True,
        acceptable_relative_rmse=0.08,
    ):
        self.model_name, self.model = resolve_ecm_model(model)
        self.initial = dict(initial or {})
        self.bounds = dict(bounds or {})
        self.weighting = weighting
        self.max_nfev = int(max_nfev)
        self.strict_bounds = bool(strict_bounds)
        self.acceptable_relative_rmse = float(acceptable_relative_rmse)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        lambda_model = data.get("model", "R0-RQ")
        return cls(
            model=lambda_model,
            initial=data.get("initial"),
            bounds=data.get("bounds"),
            weighting=data.get("weighting", "modulus"),
            max_nfev=data.get("max_nfev", 20000),
            strict_bounds=data.get("strict_bounds", True),
            acceptable_relative_rmse=data.get("acceptable_relative_rmse", 0.08),
        )


class ECMFitResult(object):
    def __init__(
        self,
        circuit,
        parameter_names,
        parameters,
        bounds,
        z_fit,
        success,
        message,
        cost,
        metrics,
    ):
        self.circuit = circuit
        self.parameter_names = parameter_names
        self.parameters = parameters
        self.bounds = bounds
        self.z_fit = z_fit
        self.success = bool(success)
        self.message = message
        self.cost = float(cost)
        self.metrics = metrics

    def parameters_frame(self):
        rows = []
        for name in self.parameter_names:
            lower, upper = self.bounds[name]
            rows.append(
                {
                    "parameter": name,
                    "value": self.parameters[name],
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            )
        return pd.DataFrame(rows)


def fit_ecm(freq_hz, z, config=None):
    config = config or ECMConfig()
    freq_hz = np.asarray(freq_hz, dtype=float)
    z = np.asarray(z, dtype=complex)

    _validate_eis(freq_hz, z)

    circuit = CircuitModel(config.model)
    names = circuit.parameter_names

    bounds = circuit.default_bounds(freq_hz, z)
    for key, value in config.bounds.items():
        if key not in bounds:
            if config.strict_bounds:
                raise KeyError("Bounds contain unknown parameter: %s" % key)
            continue
        if len(value) != 2:
            raise ValueError("Bounds for %s must be [lower, upper]" % key)
        bounds[key] = (float(value[0]), float(value[1]))

    lower = np.array([bounds[name][0] for name in names], dtype=float)
    upper = np.array([bounds[name][1] for name in names], dtype=float)

    def objective(x):
        params = dict(zip(names, x))
        z_model = circuit.impedance(freq_hz, params)
        return _weighted_residual(z_model, z, config.weighting)

    best = None
    for initial in _initial_candidates(circuit, freq_hz, z, config.initial):
        x0 = np.array([float(initial[name]) for name in names], dtype=float)
        x0 = _clip_initial(x0, lower, upper)
        result = least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            max_nfev=config.max_nfev,
        )
        params = dict(zip(names, result.x))
        z_fit = circuit.impedance(freq_hz, params)
        metrics = _fit_metrics(z, z_fit)
        candidate = (metrics.get("relative_rmse", np.inf), result, params, z_fit, metrics)
        if best is None or candidate[0] < best[0]:
            best = candidate

    _score, result, params, z_fit, metrics = best

    success = bool(result.success or _is_acceptable_ecm_fit(metrics, config))
    message = str(result.message)
    if success and not result.success:
        message = (
            message
            + " Accepted because relative_rmse is below acceptable_relative_rmse."
        )

    return ECMFitResult(
        circuit=circuit,
        parameter_names=names,
        parameters=params,
        bounds=bounds,
        z_fit=z_fit,
        success=success,
        message=message,
        cost=result.cost,
        metrics=metrics,
    )


def ecm_model_presets():
    rows = []
    for name, expression, description in ECM_MODEL_PRESETS:
        rows.append(
            {
                "name": name,
                "expression": expression,
                "description": description,
            }
        )
    return rows


def resolve_ecm_model(model):
    value = str(model or "").strip()
    if not value:
        value = "R0-RQ"
    normalized = value.lower()
    for name, expression, _description in ECM_MODEL_PRESETS:
        if normalized == name.lower() or normalized == expression.lower():
            return name, expression
    return value, value


def ecm_configs_from_dict(data):
    data = data or {}
    models = data.get("models")
    if models is None:
        models = data.get("model", "R0-RQ")
    models = parse_model_list(models)
    if not models:
        models = ["R0-RQ"]

    strict_bounds = data.get("strict_bounds")
    if strict_bounds is None:
        strict_bounds = len(models) == 1

    configs = []
    for model in models:
        configs.append(
            ECMConfig(
                model=model,
                initial=data.get("initial"),
                bounds=data.get("bounds"),
                weighting=data.get("weighting", "modulus"),
                max_nfev=data.get("max_nfev", 20000),
                strict_bounds=strict_bounds,
                acceptable_relative_rmse=data.get("acceptable_relative_rmse", 0.08),
            )
        )
    return configs


def parse_model_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(parse_model_list(item))
        return output

    text = str(value).strip()
    if not text:
        return []
    return _split_model_list_text(text)


def _split_model_list_text(text):
    parts = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    part = text[start:].strip()
    if part:
        parts.append(part)
    return parts


def _weighted_residual(z_model, z_measured, weighting):
    diff_real = np.real(z_model) - np.real(z_measured)
    diff_imag = np.imag(z_model) - np.imag(z_measured)

    if weighting in (None, "none", "None"):
        return np.concatenate([diff_real, diff_imag])
    if weighting == "modulus":
        scale = 1.0 / np.maximum(np.abs(z_measured), 1e-30)
        return np.concatenate([diff_real * scale, diff_imag * scale])
    if weighting == "real-imag":
        real_scale = 1.0 / np.maximum(np.abs(np.real(z_measured)), 1e-30)
        imag_scale = 1.0 / np.maximum(np.abs(np.imag(z_measured)), 1e-30)
        return np.concatenate([diff_real * real_scale, diff_imag * imag_scale])
    raise ValueError("Unknown weighting: %s" % weighting)


def _initial_candidates(circuit, freq_hz, z, user_initial):
    base = circuit.guess_parameters(freq_hz, z)
    user_initial = dict(user_initial or {})
    first = dict(base)
    first.update(user_initial)

    candidates = [first]
    names = circuit.parameter_names
    cpe_q_names = [name for name in names if name.endswith("_Q")]
    cpe_n_names = [name for name in names if name.endswith("_n")]
    if not cpe_q_names:
        return candidates

    real_z = np.real(np.asarray(z, dtype=complex))
    real_min = max(float(np.nanmin(real_z)), 0.0)
    real_max = float(np.nanmax(real_z))
    real_range = max(real_max - real_min, 0.0)
    resistance_scale = max(real_range, 0.1 * abs(real_max), 0.1 * abs(real_min), 1e-12)
    positive_freq = np.asarray(freq_hz, dtype=float)
    positive_freq = positive_freq[positive_freq > 0]
    if positive_freq.size:
        f_mid = float(np.exp(np.mean(np.log(positive_freq))))
    else:
        f_mid = 1.0
    q_scale = 1.0 / (2.0 * np.pi * max(f_mid, 1e-30) * max(resistance_scale, 1e-30))
    q_values = _positive_unique([q_scale, q_scale * 0.2, q_scale * 2.0, 100.0, 10.0, 1000.0])
    n_values = [0.98, 0.9, 0.75]

    r_names = [name for name in names if _parameter_prefix(name) == "R" and name.lower() not in ("r0", "rs", "rinf")]
    w_names = [name for name in names if _parameter_prefix(name) == "W"]
    for q_value, n_value in zip(q_values[:6], (n_values * 3)[:6]):
        candidate = dict(base)
        candidate.update(user_initial)
        for name in r_names:
            if name not in user_initial:
                candidate[name] = max(resistance_scale / max(len(r_names), 1), 1e-12)
        for name in w_names:
            if name not in user_initial:
                candidate[name] = max(resistance_scale * np.sqrt(2.0 * np.pi * max(f_mid, 1e-30)) / 4.0, 1e-12)
        for name in cpe_q_names:
            if name not in user_initial:
                candidate[name] = q_value
        for name in cpe_n_names:
            if name not in user_initial:
                candidate[name] = n_value
        candidates.append(candidate)

    return _dedupe_initials(candidates, names)


def _is_acceptable_ecm_fit(metrics, config):
    rel = metrics.get("relative_rmse")
    if rel is None or not np.isfinite(rel):
        return False
    return float(rel) <= float(config.acceptable_relative_rmse)


def _parameter_prefix(name):
    text = str(name)
    if text.startswith("CPE"):
        return "CPE"
    return text[:1].upper()


def _positive_unique(values):
    output = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if value <= 0 or not np.isfinite(value):
            continue
        if not any(abs(np.log(value) - np.log(old)) < 1e-6 for old in output):
            output.append(value)
    return output


def _dedupe_initials(candidates, names):
    output = []
    seen = set()
    for candidate in candidates:
        key = tuple(round(float(candidate[name]), 14) for name in names)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _fit_metrics(z_measured, z_fit):
    residual = z_fit - z_measured
    abs_z = np.maximum(np.abs(z_measured), 1e-30)
    rmse = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    rel_rmse = float(np.sqrt(np.mean((np.abs(residual) / abs_z) ** 2)))
    ss_res = float(np.sum(np.abs(residual) ** 2))
    centered = z_measured - np.mean(z_measured)
    ss_tot = float(np.sum(np.abs(centered) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "rmse_ohm": rmse,
        "relative_rmse": rel_rmse,
        "r2_complex": float(r2),
    }


def _clip_initial(x0, lower, upper):
    clipped = np.array(x0, copy=True)
    for i in range(clipped.size):
        lo = lower[i]
        hi = upper[i]
        if clipped[i] <= lo:
            clipped[i] = lo + max(1e-12, abs(lo) * 1e-6)
        if clipped[i] >= hi:
            clipped[i] = hi - max(1e-12, abs(hi) * 1e-6)
    return clipped


def _validate_eis(freq_hz, z):
    if freq_hz.ndim != 1 or z.ndim != 1:
        raise ValueError("freq_hz and z must be one-dimensional arrays")
    if freq_hz.size != z.size:
        raise ValueError("freq_hz and z must have the same length")
    if freq_hz.size < 3:
        raise ValueError("At least three EIS points are required")
    if np.any(freq_hz <= 0):
        raise ValueError("All frequencies must be positive")
