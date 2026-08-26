import numpy as np
from scipy.optimize import lsq_linear, nnls


class DRTConfig(object):
    """Configuration for non-negative Tikhonov regularized RBF DRT."""

    def __init__(
        self,
        lambda_value=1e-3,
        lambda_selection="mgcv",
        lambda_min=1e-7,
        lambda_max=1.0,
        lambda_count=41,
        n_tau=750,
        tau_min=None,
        tau_max=None,
        tau_padding_decades=0.5,
        regularization_order=1,
        basis_function="gaussian",
        shape_factor=0.5,
        n_basis=None,
        polarization_removal="none",
        boundary_suppression_factor=0.0,
        nonnegative=True,
        fit_r_inf=True,
        fit_inductance=True,
        weighting="modulus",
        normalization="polarization_resistance",
        max_nfev=50000,
    ):
        self.lambda_value = float(lambda_value)
        self.lambda_selection = str(lambda_selection or "fixed").lower()
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_count = int(lambda_count)
        self.n_tau = int(n_tau)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_padding_decades = float(tau_padding_decades)
        self.regularization_order = int(regularization_order)
        self.basis_function = basis_function
        # DRTtools-style coefficient: FWHM = log(tau) spacing / shape_factor.
        self.shape_factor = float(shape_factor)
        self.n_basis = None if n_basis in (None, "", 0) else int(n_basis)
        self.polarization_removal = polarization_removal
        self.boundary_suppression_factor = float(boundary_suppression_factor)
        self.nonnegative = bool(nonnegative)
        self.fit_r_inf = bool(fit_r_inf)
        self.fit_inductance = bool(fit_inductance)
        self.weighting = weighting
        self.normalization = str(normalization or "none").lower()
        self.max_nfev = int(max_nfev)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            lambda_value=data.get("lambda_value", data.get("lambda", 1e-3)),
            lambda_selection=data.get("lambda_selection", "mgcv"),
            lambda_min=data.get("lambda_min", 1e-7),
            lambda_max=data.get("lambda_max", 1.0),
            lambda_count=data.get("lambda_count", 41),
            n_tau=data.get("n_tau", 750),
            tau_min=data.get("tau_min"),
            tau_max=data.get("tau_max"),
            tau_padding_decades=data.get("tau_padding_decades", 0.5),
            regularization_order=data.get("regularization_order", 1),
            basis_function=data.get("basis_function", "gaussian"),
            shape_factor=data.get("shape_factor", 0.5),
            n_basis=data.get("n_basis"),
            polarization_removal=data.get("polarization_removal", "none"),
            boundary_suppression_factor=data.get("boundary_suppression_factor", 0.0),
            nonnegative=data.get("nonnegative", True),
            fit_r_inf=data.get("fit_r_inf", True),
            fit_inductance=data.get("fit_inductance", True),
            weighting=data.get("weighting", "modulus"),
            normalization=data.get("normalization", "polarization_resistance"),
            max_nfev=data.get("max_nfev", 50000),
        )


class DRTResult(object):
    def __init__(self, tau, gamma, weights, r_inf, inductance, z_fit, success,
                 message, cost, lambda_value, lambda_selection, lambda_score,
                 normalization_ohm, n_basis, supported_tau_min, supported_tau_max):
        self.tau = tau
        self.gamma = gamma
        self.weights = weights
        self.r_inf = r_inf
        self.inductance = inductance
        self.z_fit = z_fit
        self.success = bool(success)
        self.message = message
        self.cost = float(cost)
        self.lambda_value = float(lambda_value)
        self.lambda_selection = str(lambda_selection)
        self.lambda_score = None if lambda_score is None else float(lambda_score)
        self.normalization_ohm = float(normalization_ohm)
        self.n_basis = int(n_basis)
        self.supported_tau_min = float(supported_tau_min)
        self.supported_tau_max = float(supported_tau_max)
        self.total_polarization_resistance = float(np.sum(gamma * weights))

    def frame(self):
        import pandas as pd
        supported = (self.tau >= self.supported_tau_min) & (self.tau <= self.supported_tau_max)
        return pd.DataFrame({
            "tau_s": self.tau,
            "log_tau": np.log(self.tau),
            "gamma_ohm": self.gamma,
            "integration_weight": self.weights,
            "frequency_supported": supported,
        })


def solve_drt(freq_hz, z, config=None):
    """Solve combined real/imaginary TR-RBF DRT on a frequency-derived grid."""
    config = config or DRTConfig()
    freq_hz = np.asarray(freq_hz, dtype=float)
    z = np.asarray(z, dtype=complex)
    _validate_eis(freq_hz, z)
    _validate_config(config)

    omega = 2.0 * np.pi * freq_hz
    supported_tau_min, supported_tau_max = _supported_tau_bounds(freq_hz, config)
    tau = build_tau_grid(freq_hz, config)
    weights = integration_weights(np.log(tau))
    coeff_tau = _build_coefficient_tau_grid(freq_hz, supported_tau_min, supported_tau_max, config)
    output_basis = _basis_matrix(np.log(tau), np.log(coeff_tau), config)
    kernel = _build_drt_kernel(tau, weights, omega, output_basis)
    n_drt_variables = int(output_basis.shape[1])

    normalization_ohm = _normalization_scale(z, config.normalization)
    z_normalized = z / normalization_ohm
    inductance_reference = normalization_ohm / float(np.max(omega))
    data_matrix, data_target = _build_linear_system(
        kernel, omega, z_normalized, config, normalization_ohm, inductance_reference
    )
    data_matrix, data_target = _apply_weighting(data_matrix, data_target, z_normalized, config.weighting)
    regularization = _regularization_matrix(output_basis, tau, data_matrix.shape[1], config)
    boundary = _boundary_matrix(output_basis, data_matrix.shape[1], config)
    if boundary.size:
        regularization = np.vstack([regularization, boundary])

    lambda_value, lambda_score = _select_lambda(data_matrix, data_target, regularization, config)
    matrix, target = _append_penalty(data_matrix, data_target, regularization, lambda_value)
    x, success, message, cost = _solve_linear_system(matrix, target, n_drt_variables, config)

    gamma_coeff = x[:n_drt_variables]
    gamma = normalization_ohm * np.dot(output_basis, gamma_coeff)
    cursor = n_drt_variables
    r_inf = 0.0
    inductance = 0.0
    if config.fit_r_inf:
        r_inf = float(x[cursor] * normalization_ohm)
        cursor += 1
    if config.fit_inductance:
        inductance = float(x[cursor] * inductance_reference)
    z_fit = normalization_ohm * np.dot(kernel, gamma_coeff) + r_inf + 1j * omega * inductance

    return DRTResult(
        tau, gamma, weights, r_inf, inductance, z_fit, success, message, cost,
        lambda_value, config.lambda_selection, lambda_score, normalization_ohm,
        n_drt_variables, supported_tau_min, supported_tau_max,
    )


def build_tau_grid(freq_hz, config):
    supported_min, supported_max = _supported_tau_bounds(freq_hz, config)
    padding = max(float(getattr(config, "tau_padding_decades", 0.5)), 0.0)
    tau_min = supported_min / (10.0 ** padding)
    tau_max = supported_max * (10.0 ** padding)
    if config.n_tau < 3:
        raise ValueError("n_tau must be at least 3")
    return np.logspace(np.log10(tau_min), np.log10(tau_max), config.n_tau)


def integration_weights(log_tau):
    log_tau = np.asarray(log_tau, dtype=float)
    if log_tau.size < 2:
        raise ValueError("At least two tau points are required")
    weights = np.empty_like(log_tau)
    weights[0] = 0.5 * (log_tau[1] - log_tau[0])
    weights[-1] = 0.5 * (log_tau[-1] - log_tau[-2])
    if log_tau.size > 2:
        weights[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
    return weights


def _supported_tau_bounds(freq_hz, config):
    positive = np.asarray(freq_hz, dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError("All frequencies must be positive")
    tau_min = config.tau_min
    tau_max = config.tau_max
    if tau_min is None:
        tau_min = 1.0 / (2.0 * np.pi * float(np.max(positive)))
    if tau_max is None:
        tau_max = 1.0 / (2.0 * np.pi * float(np.min(positive)))
    tau_min, tau_max = float(tau_min), float(tau_max)
    if tau_min <= 0 or tau_max <= 0:
        raise ValueError("tau_min and tau_max must be positive")
    if tau_min >= tau_max:
        raise ValueError("tau_min must be smaller than tau_max")
    return tau_min, tau_max


def _build_coefficient_tau_grid(freq_hz, tau_min, tau_max, config):
    natural = np.sort(np.unique(1.0 / (2.0 * np.pi * np.asarray(freq_hz, dtype=float))))
    natural = natural[(natural >= tau_min) & (natural <= tau_max)]
    if config.n_basis is None:
        return natural if natural.size >= 3 else np.logspace(np.log10(tau_min), np.log10(tau_max), 3)
    if config.n_basis < 3:
        raise ValueError("n_basis must be at least 3 when provided")
    return np.logspace(np.log10(tau_min), np.log10(tau_max), int(config.n_basis))


def _basis_matrix(output_log_tau, coeff_log_tau, config):
    name = str(config.basis_function or "gaussian").lower()
    output_log_tau = np.asarray(output_log_tau, dtype=float)
    coeff_log_tau = np.asarray(coeff_log_tau, dtype=float)
    n_output, n_coeff = output_log_tau.size, coeff_log_tau.size
    if name in ("delta", "dirac", "identity", "none"):
        if n_output != n_coeff or not np.allclose(output_log_tau, coeff_log_tau):
            raise ValueError("delta basis requires matching coefficient and output grids")
        return np.eye(n_output)
    if name != "gaussian":
        raise ValueError("Unsupported DRT basis_function: %s" % config.basis_function)
    coefficient = max(float(config.shape_factor), 1e-12)
    spacing = float(np.median(np.abs(np.diff(coeff_log_tau))))
    fwhm = max(spacing / coefficient, np.finfo(float).eps)
    distances = output_log_tau[:, np.newaxis] - coeff_log_tau[np.newaxis, :]
    return np.exp(-4.0 * np.log(2.0) * (distances / fwhm) ** 2)


def _build_drt_kernel(tau, weights, omega, output_basis):
    base = weights[np.newaxis, :] / (1.0 + 1j * omega[:, np.newaxis] * tau[np.newaxis, :])
    return np.dot(base, output_basis)


def _normalization_scale(z, method):
    if method in (None, "none", "None"):
        return 1.0
    if method not in ("polarization_resistance", "polarization", "r_pol"):
        raise ValueError("Unsupported DRT normalization: %s" % method)
    real = np.real(z)
    candidates = [
        abs(float(real[-1] - real[0])), float(np.ptp(real)),
        float(np.percentile(np.abs(z - z[0]), 90.0)), float(np.median(np.abs(z))),
    ]
    finite = [value for value in candidates if np.isfinite(value) and value > 1e-30]
    return max(finite[0] if finite else 1.0, 1e-30)


def _build_linear_system(kernel, omega, z_normalized, config, scale, inductance_reference):
    real_cols, imag_cols = [np.real(kernel)], [np.imag(kernel)]
    if config.fit_r_inf:
        real_cols.append(np.ones((kernel.shape[0], 1)))
        imag_cols.append(np.zeros((kernel.shape[0], 1)))
    if config.fit_inductance:
        normalized_l_column = omega * inductance_reference / scale
        real_cols.append(np.zeros((kernel.shape[0], 1)))
        imag_cols.append(normalized_l_column[:, np.newaxis])
    matrix = np.vstack([np.hstack(real_cols), np.hstack(imag_cols)])
    target = np.concatenate([np.real(z_normalized), np.imag(z_normalized)])
    return matrix, target


def _apply_weighting(matrix, target, z, weighting):
    if weighting in (None, "none", "None"):
        return matrix, target
    if weighting != "modulus":
        raise ValueError("DRT only supports weighting='modulus' or 'none'")
    point_scale = 1.0 / np.maximum(np.abs(z), 1e-12)
    row_scale = np.concatenate([point_scale, point_scale])
    return matrix * row_scale[:, None], target * row_scale


def _regularization_matrix(output_basis, tau, n_variables, config):
    order = int(config.regularization_order)
    if order not in (0, 1, 2):
        raise ValueError("regularization_order must be 0, 1, or 2")
    spacing = float(np.median(np.diff(np.log(np.asarray(tau, dtype=float)))))
    if order == 0:
        derivative = output_basis * np.sqrt(spacing)
    else:
        derivative = np.diff(output_basis, n=order, axis=0) / (spacing ** order)
        derivative = derivative * np.sqrt(spacing)
    regularization = np.zeros((derivative.shape[0], n_variables))
    regularization[:, : output_basis.shape[1]] = derivative
    return regularization


def _boundary_matrix(output_basis, n_variables, config):
    if not _uses_boundary_suppression(config) or config.boundary_suppression_factor <= 0:
        return np.zeros((0, n_variables))
    boundary = np.zeros((2, n_variables))
    boundary[0, : output_basis.shape[1]] = np.sqrt(config.boundary_suppression_factor) * output_basis[0]
    boundary[1, : output_basis.shape[1]] = np.sqrt(config.boundary_suppression_factor) * output_basis[-1]
    return boundary


def _uses_boundary_suppression(config):
    value = str(config.polarization_removal or "").lower()
    return value in ("ignore", "ignore_polarization", "ignore-polarization", "boundary_suppression",
                     "boundary-suppression", "remove_boundary", "remove-boundary")


def _select_lambda(matrix, target, regularization, config):
    method = str(config.lambda_selection or "fixed").lower()
    if method in ("fixed", "manual", "none"):
        return max(float(config.lambda_value), 0.0), None
    if method not in ("gcv", "mgcv"):
        raise ValueError("lambda_selection must be 'mgcv', 'gcv', or 'fixed'")
    if config.lambda_min <= 0 or config.lambda_max <= 0 or config.lambda_min >= config.lambda_max:
        raise ValueError("lambda bounds must be positive and increasing")
    if config.lambda_count < 3:
        raise ValueError("lambda_count must be at least 3")
    lambdas = np.logspace(np.log10(config.lambda_min), np.log10(config.lambda_max), config.lambda_count)
    ata, atb = np.dot(matrix.T, matrix), np.dot(matrix.T, target)
    penalty = np.dot(regularization.T, regularization)
    n_rows = float(matrix.shape[0])
    rho = 1.3 if (method == "mgcv" and matrix.shape[0] < 50) else (2.0 if method == "mgcv" else 1.0)
    scores = []
    for value in lambdas:
        normal = ata + value * penalty
        ridge = np.finfo(float).eps * max(float(np.trace(normal)), 1.0)
        normal = normal + ridge * np.eye(normal.shape[0])
        try:
            coefficient = np.linalg.solve(normal, atb)
            influence = np.linalg.solve(normal, matrix.T)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(normal)
            coefficient, influence = np.dot(inverse, atb), np.dot(inverse, matrix.T)
        residual = target - np.dot(matrix, coefficient)
        trace_hat = float(np.sum(matrix * influence.T))
        numerator = float(np.dot(residual, residual)) / n_rows
        denominator = ((n_rows - rho * trace_hat) / n_rows) ** 2
        score = numerator / max(denominator, np.finfo(float).tiny)
        scores.append(score if np.isfinite(score) else np.inf)
    index = int(np.argmin(scores))
    return float(lambdas[index]), float(scores[index])


def _append_penalty(matrix, target, regularization, lambda_value):
    if lambda_value <= 0 or regularization.size == 0:
        return matrix, target
    penalty = np.sqrt(lambda_value) * regularization
    return np.vstack([matrix, penalty]), np.concatenate([target, np.zeros(penalty.shape[0])])


def _solve_linear_system(matrix, target, n_drt_variables, config):
    n_variables = matrix.shape[1]
    lower, upper = np.full(n_variables, -np.inf), np.full(n_variables, np.inf)
    if config.nonnegative:
        lower[:n_drt_variables] = 0.0
    cursor = n_drt_variables
    if config.fit_r_inf:
        lower[cursor] = 0.0
        cursor += 1
    if config.fit_inductance:
        lower[cursor] = 0.0
    if np.all(lower == 0.0) and np.all(np.isposinf(upper)):
        x, residual_norm = nnls(matrix, target)
        return x, True, "Solved by nonnegative TR-RBF least squares.", 0.5 * float(residual_norm ** 2)
    if np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
        x, _residuals, _rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
        residual = np.dot(matrix, x) - target
        return x, True, "Solved by unconstrained TR-RBF least squares.", 0.5 * float(np.dot(residual, residual))
    result = lsq_linear(matrix, target, bounds=(lower, upper), max_iter=config.max_nfev)
    return result.x, result.success, result.message, result.cost


def _validate_config(config):
    if config.n_tau < 3:
        raise ValueError("n_tau must be at least 3")
    if config.shape_factor <= 0:
        raise ValueError("shape_factor/FWHM coefficient must be positive")


def _validate_eis(freq_hz, z):
    if freq_hz.ndim != 1 or z.ndim != 1:
        raise ValueError("freq_hz and z must be one-dimensional arrays")
    if freq_hz.size != z.size:
        raise ValueError("freq_hz and z must have the same length")
    if freq_hz.size < 3:
        raise ValueError("At least three EIS points are required")
    if np.any(freq_hz <= 0):
        raise ValueError("All frequencies must be positive")
    if not np.all(np.isfinite(freq_hz)) or not np.all(np.isfinite(z)):
        raise ValueError("EIS data must be finite")
