import numpy as np
from scipy.optimize import lsq_linear, nnls


class DRTConfig(object):
    def __init__(
        self,
        lambda_value=10.0,
        n_tau=750,
        tau_min=None,
        tau_max=None,
        regularization_order=1,
        basis_function="gaussian",
        shape_factor=4.0,
        n_basis=120,
        polarization_removal="ignore_polarization",
        boundary_suppression_factor=10.0,
        nonnegative=True,
        fit_r_inf=True,
        fit_inductance=True,
        weighting="modulus",
        max_nfev=50000,
    ):
        self.lambda_value = float(lambda_value)
        self.n_tau = int(n_tau)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.regularization_order = int(regularization_order)
        self.basis_function = basis_function
        self.shape_factor = float(shape_factor)
        self.n_basis = None if n_basis in (None, "") else int(n_basis)
        self.polarization_removal = polarization_removal
        self.boundary_suppression_factor = float(boundary_suppression_factor)
        self.nonnegative = bool(nonnegative)
        self.fit_r_inf = bool(fit_r_inf)
        self.fit_inductance = bool(fit_inductance)
        self.weighting = weighting
        self.max_nfev = int(max_nfev)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        lambda_value = data.get("lambda_value", data.get("lambda", 10.0))
        return cls(
            lambda_value=lambda_value,
            n_tau=data.get("n_tau", 750),
            tau_min=data.get("tau_min"),
            tau_max=data.get("tau_max"),
            regularization_order=data.get("regularization_order", 1),
            basis_function=data.get("basis_function", "gaussian"),
            shape_factor=data.get("shape_factor", 4.0),
            n_basis=data.get("n_basis", 120),
            polarization_removal=data.get("polarization_removal", "ignore_polarization"),
            boundary_suppression_factor=data.get("boundary_suppression_factor", 10.0),
            nonnegative=data.get("nonnegative", True),
            fit_r_inf=data.get("fit_r_inf", True),
            fit_inductance=data.get("fit_inductance", True),
            weighting=data.get("weighting", "modulus"),
            max_nfev=data.get("max_nfev", 50000),
        )


class DRTResult(object):
    def __init__(
        self,
        tau,
        gamma,
        weights,
        r_inf,
        inductance,
        z_fit,
        success,
        message,
        cost,
        lambda_value,
        n_basis,
    ):
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
        self.n_basis = int(n_basis)
        self.total_polarization_resistance = float(np.sum(gamma * weights))

    def frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "tau_s": self.tau,
                "log_tau": np.log(self.tau),
                "gamma_ohm": self.gamma,
                "integration_weight": self.weights,
            }
        )


def solve_drt(freq_hz, z, config=None):
    config = config or DRTConfig()
    freq_hz = np.asarray(freq_hz, dtype=float)
    z = np.asarray(z, dtype=complex)
    _validate_eis(freq_hz, z)

    tau = build_tau_grid(freq_hz, config)
    weights = integration_weights(np.log(tau))
    omega = 2.0 * np.pi * freq_hz
    kernel, output_basis, _coeff_tau = _build_drt_kernel(tau, weights, omega, config)
    n_drt_variables = int(output_basis.shape[1])

    matrix, target = _build_linear_system(kernel, omega, z, config)
    matrix, target = _apply_weighting(matrix, target, z, config.weighting)
    matrix, target = _append_boundary_suppression(matrix, target, output_basis, z, config)
    matrix, target = _append_regularization(matrix, target, n_drt_variables, config)

    n_variables = matrix.shape[1]
    lower = np.full(n_variables, -np.inf)
    upper = np.full(n_variables, np.inf)
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
        success = True
        message = "Solved by nonnegative linear least squares."
        cost = 0.5 * float(residual_norm ** 2)
    elif np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
        x, residuals, _rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
        residual = np.dot(matrix, x) - target
        success = True
        message = "Solved by unconstrained linear least squares."
        cost = 0.5 * float(np.dot(residual, residual))
    else:
        result = lsq_linear(
            matrix,
            target,
            bounds=(lower, upper),
            max_iter=config.max_nfev,
        )
        x = result.x
        success = result.success
        message = result.message
        cost = result.cost

    gamma_coeff = x[:n_drt_variables]
    gamma = np.dot(output_basis, gamma_coeff)
    cursor = n_drt_variables
    r_inf = 0.0
    inductance = 0.0
    if config.fit_r_inf:
        r_inf = float(x[cursor])
        cursor += 1
    if config.fit_inductance:
        inductance = float(x[cursor])

    z_fit = np.dot(kernel, gamma_coeff) + r_inf + 1j * omega * inductance

    return DRTResult(
        tau=tau,
        gamma=gamma,
        weights=weights,
        r_inf=r_inf,
        inductance=inductance,
        z_fit=z_fit,
        success=success,
        message=message,
        cost=cost,
        lambda_value=config.lambda_value,
        n_basis=n_drt_variables,
    )


def build_tau_grid(freq_hz, config):
    freq_hz = np.asarray(freq_hz, dtype=float)
    positive = freq_hz[freq_hz > 0]
    if positive.size == 0:
        raise ValueError("All frequencies must be positive")

    tau_min = config.tau_min
    tau_max = config.tau_max
    if tau_min is None:
        tau_min = 0.1 / (2.0 * np.pi * float(np.max(positive)))
    if tau_max is None:
        tau_max = 10.0 / (2.0 * np.pi * float(np.min(positive)))

    tau_min = float(tau_min)
    tau_max = float(tau_max)
    if tau_min <= 0 or tau_max <= 0:
        raise ValueError("tau_min and tau_max must be positive")
    if tau_min >= tau_max:
        raise ValueError("tau_min must be smaller than tau_max")
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


def _build_drt_kernel(tau, weights, omega, config):
    base_kernel = weights[np.newaxis, :] / (
        1.0 + 1j * omega[:, np.newaxis] * tau[np.newaxis, :]
    )
    coeff_tau = _build_coefficient_tau_grid(tau, config)
    basis = _basis_matrix(np.log(tau), np.log(coeff_tau), config)
    return np.dot(base_kernel, basis), basis, coeff_tau


def _build_coefficient_tau_grid(output_tau, config):
    n_output = int(output_tau.size)
    n_basis = getattr(config, "n_basis", None)
    if n_basis is None or n_basis <= 0 or n_basis >= n_output:
        return output_tau
    if n_basis < 3:
        raise ValueError("n_basis must be at least 3 when provided")
    return np.logspace(np.log10(float(output_tau[0])), np.log10(float(output_tau[-1])), int(n_basis))


def _basis_matrix(output_log_tau, coeff_log_tau, config):
    name = str(getattr(config, "basis_function", "gaussian") or "gaussian").lower()
    output_log_tau = np.asarray(output_log_tau, dtype=float)
    coeff_log_tau = np.asarray(coeff_log_tau, dtype=float)
    n_output = output_log_tau.size
    if name in ("delta", "dirac", "identity", "none"):
        if n_output != n_coeff or not np.allclose(output_log_tau, coeff_log_tau):
            raise ValueError("n_basis can only differ from n_tau when basis_function='gaussian'")
        return np.eye(n_output)
    if name != "gaussian":
        raise ValueError("Unsupported DRT basis_function: %s" % config.basis_function)

    distances = output_log_tau[:, np.newaxis] - coeff_log_tau[np.newaxis, :]
    sigma = _gaussian_sigma(coeff_log_tau, config.shape_factor)
    basis = np.exp(-0.5 * (distances / sigma) ** 2)
    row_sum = np.sum(basis, axis=1)
    row_sum = np.maximum(row_sum, 1e-300)
    return basis / row_sum[:, np.newaxis]


def _gaussian_sigma(log_tau, shape_factor):
    shape = max(float(shape_factor), 1e-12)
    if log_tau.size < 2:
        return shape
    spacing = float(np.median(np.abs(np.diff(log_tau))))
    # Expose shape as a dimensionless smoothing width. Treat values below one
    # log-spacing as grid-relative;
    # otherwise use the value directly in natural-log tau units.
    if shape <= 2.0 and shape * spacing < 0.05:
        return max(shape, spacing)
    return max(shape * spacing, spacing)


def _build_linear_system(kernel, omega, z, config):
    real_cols = [np.real(kernel)]
    imag_cols = [np.imag(kernel)]

    if config.fit_r_inf:
        real_cols.append(np.ones((kernel.shape[0], 1)))
        imag_cols.append(np.zeros((kernel.shape[0], 1)))
    if config.fit_inductance:
        real_cols.append(np.zeros((kernel.shape[0], 1)))
        imag_cols.append(omega[:, np.newaxis])

    real_matrix = np.hstack(real_cols)
    imag_matrix = np.hstack(imag_cols)
    matrix = np.vstack([real_matrix, imag_matrix])
    target = np.concatenate([np.real(z), np.imag(z)])
    return matrix, target


def _apply_weighting(matrix, target, z, weighting):
    if weighting in (None, "none", "None"):
        return matrix, target
    if weighting != "modulus":
        raise ValueError("DRT only supports weighting='modulus' or 'none'")

    n = z.size
    scale = 1.0 / np.maximum(np.abs(z), 1e-30)
    row_scale = np.concatenate([scale, scale])
    weighted_matrix = matrix.copy()
    weighted_target = target.copy()
    weighted_matrix[: 2 * n, :] = weighted_matrix[: 2 * n, :] * row_scale[:, None]
    weighted_target[: 2 * n] = weighted_target[: 2 * n] * row_scale
    return weighted_matrix, weighted_target


def _append_boundary_suppression(matrix, target, output_basis, z, config):
    if not _uses_boundary_suppression(config):
        return matrix, target

    factor = float(getattr(config, "boundary_suppression_factor", 0.25))
    if factor <= 0:
        return matrix, target

    strength = _boundary_suppression_strength(z, config.weighting) * factor
    if strength <= 0:
        return matrix, target

    n_drt_variables = output_basis.shape[1]
    extra = np.zeros((2, matrix.shape[1]))
    scale = np.sqrt(strength)
    extra[0, :n_drt_variables] = scale * output_basis[0, :]
    extra[1, :n_drt_variables] = scale * output_basis[-1, :]
    return np.vstack([matrix, extra]), np.concatenate([target, np.zeros(2)])


def _uses_boundary_suppression(config):
    value = str(getattr(config, "polarization_removal", "") or "").lower()
    return value in (
        "ignore",
        "ignore_polarization",
        "ignore-polarization",
        "boundary_suppression",
        "boundary-suppression",
        "remove_boundary",
        "remove-boundary",
    )


def _boundary_suppression_strength(z, weighting):
    if weighting == "modulus":
        scale = 1.0 / np.maximum(np.abs(z), 1e-30)
        return float(np.median(scale) ** 2)
    return 1.0


def _append_regularization(matrix, target, n_tau, config):
    if config.lambda_value <= 0:
        return matrix, target

    order = config.regularization_order
    if order == 0:
        diff = np.eye(n_tau)
    elif order in (1, 2):
        diff = np.diff(np.eye(n_tau), n=order, axis=0)
    else:
        raise ValueError("regularization_order must be 0, 1, or 2")

    reg = np.zeros((diff.shape[0], matrix.shape[1]))
    reg[:, :n_tau] = np.sqrt(config.lambda_value) * diff
    return np.vstack([matrix, reg]), np.concatenate([target, np.zeros(reg.shape[0])])


def _validate_eis(freq_hz, z):
    if freq_hz.ndim != 1 or z.ndim != 1:
        raise ValueError("freq_hz and z must be one-dimensional arrays")
    if freq_hz.size != z.size:
        raise ValueError("freq_hz and z must have the same length")
    if freq_hz.size < 3:
        raise ValueError("At least three EIS points are required")
    if np.any(freq_hz <= 0):
        raise ValueError("All frequencies must be positive")
