"""User-facing ECM and DRT analysis choices."""

from copy import deepcopy


ECM_MODELS = [
    {"id": "L-R0-RWQ", "display_name": "R-W branch with CPE",
     "circuit": "L1 + R0 + ((R1 + W1) || CPE1)", "expression": "L1-R0-((R1-W1)||CPE1)",
     "description": "R1 and one-parameter W1 are in series, then in parallel with CPE1.", "recommended": True},
    {"id": "L-R0-RQ", "display_name": "Single CPE arc",
     "circuit": "L1 + R0 + (R1 || CPE1)", "expression": "L1-R0-(R1||CPE1)",
     "description": "One non-ideal polarization arc without Warburg diffusion.", "recommended": False},
    {"id": "L-R0-RC", "display_name": "Single ideal RC arc",
     "circuit": "L1 + R0 + (R1 || C1)", "expression": "L1-R0-(R1||C1)",
     "description": "One ideal RC polarization arc.", "recommended": False},
    {"id": "L-R0-2RC", "display_name": "Double ideal RC arcs",
     "circuit": "L1 + R0 + (R1 || C1) + (R2 || C2)", "expression": "L1-R0-(R1||C1)-(R2||C2)",
     "description": "Two ideal RC polarization arcs.", "recommended": False},
    {"id": "L-R0-2RQ", "display_name": "Double CPE arcs",
     "circuit": "L1 + R0 + (R1 || CPE1) + (R2 || CPE2)", "expression": "L1-R0-(R1||CPE1)-(R2||CPE2)",
     "description": "Two non-ideal polarization arcs without Warburg diffusion.", "recommended": False},
    {"id": "L-R0-RQ-RWQ", "display_name": "RQ plus R-W/CPE arc",
     "circuit": "L1 + R0 + (R1 || CPE1) + ((R2 + W1) || CPE2)",
     "expression": "L1-R0-(R1||CPE1)-((R2-W1)||CPE2)",
     "description": "Two non-ideal processes; R2 and W1 are in the same series branch.", "recommended": False},
]

CUSTOM_ECM_EXAMPLES = [
    "L1-R0-((R1-W1)||CPE1)",
    "L1-R0-(R1||CPE1)-(R2||CPE2)",
    "R0-(R1||C1)-W1",
]

_DRT_SHARED = {
    "tau_min": None, "tau_max": None, "regularization_order": 1,
    "tau_padding_decades": 0.5, "basis_function": "gaussian",
    "polarization_removal": "none", "boundary_suppression_factor": 0.0,
    "nonnegative": True,
    "fit_r_inf": True, "fit_inductance": True, "weighting": "modulus",
    "normalization": "polarization_resistance", "lambda_min": 1e-7,
    "lambda_max": 1.0, "lambda_count": 41,
}

DRT_PRESETS = [
    {"id": "balanced", "display_name": "Automatic mGCV / common",
     "description": "Recommended TR-RBF analysis; mGCV selects regularization from each spectrum.",
     "recommended": True, "config": dict(_DRT_SHARED, lambda_value=1e-3, lambda_selection="mgcv", n_tau=750, shape_factor=0.5, n_basis=None)},
    {"id": "smooth", "display_name": "Smooth / noise resistant",
     "description": "Fixed stronger regularization for noisy data; may merge weak nearby peaks.",
     "recommended": False, "config": dict(_DRT_SHARED, lambda_value=1e-2, lambda_selection="fixed", n_tau=750, shape_factor=0.5, n_basis=None)},
    {"id": "high_resolution", "display_name": "High resolution",
     "description": "Narrower RBFs and weak fixed regularization; retains detail with more over-fitting risk.",
     "recommended": False, "config": dict(_DRT_SHARED, lambda_value=1e-4, lambda_selection="fixed", n_tau=1000, shape_factor=0.75, n_basis=None)},
    {"id": "fast_preview", "display_name": "Fast preview", "description": "Reduced grid for quick first inspection.",
     "recommended": False, "config": dict(_DRT_SHARED, lambda_value=1e-3, lambda_selection="fixed", n_tau=300, shape_factor=0.5, n_basis=60)},
]

REQUIRED_DRT_FIELDS = [
    "lambda_value", "lambda_selection", "n_tau", "regularization_order", "basis_function", "shape_factor",
    "polarization_removal", "boundary_suppression_factor", "nonnegative", "fit_r_inf",
    "fit_inductance", "weighting",
]

PRESENTATION_REQUIREMENTS = [
    "Show all six ECM models before asking the user to choose.",
    "Mark L-R0-RWQ as recommended, but do not hide the other models.",
    "State that the user may enter a custom ECM expression.",
    "Show all four DRT presets before asking the user to choose.",
    "Mark balanced as recommended, but do not hide the other presets.",
    "State that the user may provide a complete custom DRT configuration.",
    "Do not run analysis until the user explicitly confirms both choices.",
]


def ecm_choices_payload():
    models = deepcopy(ECM_MODELS)
    return {
        "models": models,
        "primary_models": deepcopy(models),
        "recommended_model_id": "L-R0-RWQ",
        "custom_supported": True,
        "custom_expression_examples": list(CUSTOM_ECM_EXAMPLES),
        "selection_required": True,
    }


def drt_choices_payload():
    presets = deepcopy(DRT_PRESETS)
    recommended = next(item for item in presets if item["recommended"])
    return {
        "presets": presets,
        "recommended_preset_id": recommended["id"],
        "recommended_config": deepcopy(recommended["config"]),
        "custom_supported": True,
        "custom_config_schema": {
            "required_fields": list(REQUIRED_DRT_FIELDS),
            "optional_fields": ["n_basis", "tau_min", "tau_max", "tau_padding_decades",
                                "lambda_min", "lambda_max", "lambda_count", "normalization", "max_nfev"],
        },
        "required_fields": list(REQUIRED_DRT_FIELDS),
        "choices": {
            "basis_function": ["gaussian", "delta", "none"],
            "regularization_order": [0, 1, 2],
            "lambda_selection": ["mgcv", "gcv", "fixed"],
            "polarization_removal": ["ignore_polarization", "none", "boundary_suppression"],
            "weighting": ["modulus", "none"],
        },
        "optional_fields": {
            "n_basis": "Internal Gaussian basis count; null uses one basis center per measured frequency.",
            "tau_min": "Leave null to derive it from the highest input frequency.",
            "tau_max": "Leave null to derive it from the lowest input frequency.",
            "tau_padding_decades": "Display padding only; coefficient support still follows the input frequency range.",
        },
        "notes": [
            "Every preset derives tau_min and tau_max from each input EIS frequency range.",
            "n_tau controls CSV/plot sampling only; it does not add experimental resolution.",
            "shape_factor is the Gaussian FWHM coefficient: FWHM = log-tau spacing / shape_factor.",
            "A recommended preset is not user confirmation; explicit selection is still required.",
        ],
        "selection_required": True,
    }


def analysis_choices_payload():
    return {
        "action_required": "Present every choice below and ask the user to select both ECM and DRT.",
        "presentation_requirements": list(PRESENTATION_REQUIREMENTS),
        "ecm": ecm_choices_payload(),
        "drt": drt_choices_payload(),
    }


def drt_config_to_dict(config):
    return {
        "lambda_value": config.lambda_value, "lambda_selection": config.lambda_selection,
        "lambda_min": config.lambda_min, "lambda_max": config.lambda_max,
        "lambda_count": config.lambda_count, "n_tau": config.n_tau,
        "tau_min": config.tau_min, "tau_max": config.tau_max,
        "tau_padding_decades": config.tau_padding_decades,
        "regularization_order": config.regularization_order, "basis_function": config.basis_function,
        "shape_factor": config.shape_factor, "n_basis": config.n_basis,
        "polarization_removal": config.polarization_removal,
        "boundary_suppression_factor": config.boundary_suppression_factor,
        "nonnegative": config.nonnegative, "fit_r_inf": config.fit_r_inf,
        "fit_inductance": config.fit_inductance, "weighting": config.weighting,
        "normalization": config.normalization,
        "max_nfev": config.max_nfev,
    }


def identify_drt_preset(config):
    actual = drt_config_to_dict(config)
    for preset in DRT_PRESETS:
        if all(actual.get(key) == value for key, value in preset["config"].items()):
            return preset["id"]
    return "custom"
