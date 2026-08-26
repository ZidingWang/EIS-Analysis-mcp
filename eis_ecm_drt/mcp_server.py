import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .choices import (
    REQUIRED_DRT_FIELDS,
    analysis_choices_payload,
    drt_choices_payload,
    drt_config_to_dict,
    ecm_choices_payload,
)
from .drt import DRTConfig, solve_drt
from .ecm import ECMConfig, ecm_configs_from_dict, fit_ecm as fit_ecm_core
from .io import EISData, expand_inputs, read_eis_file
from .pipeline import _zview_compatibility_fields, analyze_batch
from .publishing import publish_analysis_results as publish_results_core


try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - exercised only when MCP is installed.
    FastMCP = None


MCP_INSTRUCTIONS = (
    "An EIS task must not finish with only a chat attachment. "
    "If analyze_eis_file or analyze_eis_batch returned a desktop output_dir, report it. "
    "Otherwise call publish_analysis_results on the generated ZIP or result directory, "
    "verify the returned desktop output_dir, and only then report completion. "
    "Do not move or delete the source results."
)


def _require_mcp():
    if FastMCP is None:
        raise RuntimeError("MCP Python SDK is not installed. Install it with: pip install -e .[mcp]")


def create_mcp_server():
    _require_mcp()
    server = FastMCP("eis-ecm-drt", instructions=MCP_INSTRUCTIONS)

    @server.tool()
    def get_analysis_choices() -> Dict[str, Any]:
        """MANDATORY first step: show every ECM model and every DRT preset, including custom options, then ask the user to explicitly select both before analysis."""
        return analysis_choices_payload()

    @server.tool()
    def list_ecm_models() -> Dict[str, Any]:
        """Show all six fixed ECM choices, the recommendation, and custom-expression support."""
        return ecm_choices_payload()

    @server.tool()
    def list_drt_options() -> Dict[str, Any]:
        """Show all four DRT presets, the recommendation, and complete custom parameters."""
        return drt_choices_payload()

    @server.tool()
    def publish_analysis_results(
        source_path: str,
        folder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mandatory finalizer for task-local EIS results: copy a directory or safely extract a ZIP into the current user's desktop EIS Analysis output folder."""
        return publish_results_core(source_path, folder_name)

    @server.tool()
    def validate_eis(
        input_path: Optional[str] = None,
        frequency_hz: Optional[List[float]] = None,
        z_real_ohm: Optional[List[float]] = None,
        z_imag_ohm: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Validate one EIS file or one in-memory EIS dataset."""
        if input_path:
            data = read_eis_file(input_path)
            return _data_response(data, source_type="file", source_path=input_path)

        if frequency_hz is None or z_real_ohm is None or z_imag_ohm is None:
            raise ValueError("Provide input_path or all of frequency_hz, z_real_ohm, and z_imag_ohm.")

        freq = np.asarray(frequency_hz, dtype=float)
        z = np.asarray(z_real_ohm, dtype=float) + 1j * np.asarray(z_imag_ohm, dtype=float)
        data = EISData(None, freq, z).sorted(descending=True)
        return _data_response(data, source_type="array", source_path=None)

    @server.tool()
    def validate_eis_batch(inputs: List[str], recursive: bool = True) -> Dict[str, Any]:
        """Validate multiple Excel, TXT, CSV, or TSV EIS files and auto-detect columns."""
        files = expand_inputs(inputs, recursive=recursive)
        valid_items = []
        invalid_items = []
        for path in files:
            try:
                valid_items.append(_data_response(read_eis_file(path), source_type="file", source_path=path))
            except Exception as exc:
                invalid_items.append({"source_path": path, "error": str(exc)})
        return {
            "success": bool(valid_items),
            "valid": bool(valid_items),
            "input_count": len(files),
            "valid_count": len(valid_items),
            "invalid_count": len(invalid_items),
            "valid_items": valid_items,
            "invalid_items": invalid_items,
        }

    @server.tool()
    def fit_ecm(
        frequency_hz: List[float],
        z_real_ohm: List[float],
        z_imag_ohm: List[float],
        ecm_model: str,
        ecm_model_confirmed_by_user: bool = False,
        ecm_weighting: str = "modulus",
        ecm_max_nfev: int = 1000,
        ecm_initial: Optional[Dict[str, Any]] = None,
        ecm_bounds: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call get_analysis_choices first, show every option, obtain explicit ECM confirmation, then fit one in-memory EIS dataset."""
        if not ecm_model:
            raise ValueError("ecm_model is required; call list_ecm_models first and let the user choose.")
        _require_ecm_selection_confirmation(ecm_model_confirmed_by_user)
        freq, z = _arrays_to_eis(frequency_hz, z_real_ohm, z_imag_ohm)
        config = ECMConfig(
            model=ecm_model,
            weighting=ecm_weighting,
            max_nfev=ecm_max_nfev,
            initial=ecm_initial,
            bounds=ecm_bounds,
            strict_bounds=False,
        )
        result = fit_ecm_core(freq, z, config)
        return _ecm_result_response(result, config, freq)

    @server.tool()
    def calculate_drt(
        frequency_hz: List[float],
        z_real_ohm: List[float],
        z_imag_ohm: List[float],
        drt_config: Dict[str, Any],
        drt_config_confirmed_by_user: bool = False,
    ) -> Dict[str, Any]:
        """Call get_analysis_choices first, show every option, obtain explicit DRT confirmation, then calculate one in-memory EIS dataset."""
        _require_drt_selection_confirmation(drt_config_confirmed_by_user)
        freq, z = _arrays_to_eis(frequency_hz, z_real_ohm, z_imag_ohm)
        config = _drt_config_from_user(drt_config)
        result = solve_drt(freq, z, config)
        return _drt_result_response(result, freq, z, drt_config_to_dict(config))

    @server.tool()
    def analyze_eis_file(
        input_path: str,
        ecm_models: List[str],
        drt_config: Dict[str, Any],
        ecm_models_confirmed_by_user: bool = False,
        drt_config_confirmed_by_user: bool = False,
        output_dir: Optional[str] = None,
        ecm_weighting: str = "modulus",
        ecm_max_nfev: int = 1000,
        ecm_acceptable_relative_rmse: float = 0.08,
        ecm_initial: Optional[Dict[str, Any]] = None,
        ecm_bounds: Optional[Dict[str, Any]] = None,
        make_plots: bool = True,
        freq_col: Optional[str] = None,
        real_col: Optional[str] = None,
        imag_col: Optional[str] = None,
        imag_sign: Optional[float] = None,
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call get_analysis_choices first, show all ECM/DRT choices, and obtain explicit confirmation of both before analyzing one EIS file."""
        return analyze_batch_for_mcp(
            inputs=[input_path],
            ecm_models=ecm_models,
            drt_config=drt_config,
            ecm_models_confirmed_by_user=ecm_models_confirmed_by_user,
            drt_config_confirmed_by_user=drt_config_confirmed_by_user,
            output_dir=output_dir,
            ecm_weighting=ecm_weighting,
            ecm_max_nfev=ecm_max_nfev,
            ecm_acceptable_relative_rmse=ecm_acceptable_relative_rmse,
            ecm_initial=ecm_initial,
            ecm_bounds=ecm_bounds,
            make_plots=make_plots,
            recursive=False,
            continue_on_error=False,
            freq_col=freq_col,
            real_col=real_col,
            imag_col=imag_col,
            imag_sign=imag_sign,
            sheet_name=sheet_name,
        )

    @server.tool()
    def analyze_eis_batch(
        inputs: List[str],
        ecm_models: List[str],
        drt_config: Dict[str, Any],
        ecm_models_confirmed_by_user: bool = False,
        drt_config_confirmed_by_user: bool = False,
        output_dir: Optional[str] = None,
        ecm_weighting: str = "modulus",
        ecm_max_nfev: int = 1000,
        ecm_acceptable_relative_rmse: float = 0.08,
        ecm_initial: Optional[Dict[str, Any]] = None,
        ecm_bounds: Optional[Dict[str, Any]] = None,
        make_plots: bool = True,
        recursive: bool = False,
        continue_on_error: bool = True,
        freq_col: Optional[str] = None,
        real_col: Optional[str] = None,
        imag_col: Optional[str] = None,
        imag_sign: Optional[float] = None,
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call get_analysis_choices first, show all ECM/DRT choices, and obtain explicit confirmation of both before analyzing multiple EIS files."""
        return analyze_batch_for_mcp(
            inputs=inputs,
            ecm_models=ecm_models,
            drt_config=drt_config,
            ecm_models_confirmed_by_user=ecm_models_confirmed_by_user,
            drt_config_confirmed_by_user=drt_config_confirmed_by_user,
            output_dir=output_dir,
            ecm_weighting=ecm_weighting,
            ecm_max_nfev=ecm_max_nfev,
            ecm_acceptable_relative_rmse=ecm_acceptable_relative_rmse,
            ecm_initial=ecm_initial,
            ecm_bounds=ecm_bounds,
            make_plots=make_plots,
            recursive=recursive,
            continue_on_error=continue_on_error,
            freq_col=freq_col,
            real_col=real_col,
            imag_col=imag_col,
            imag_sign=imag_sign,
            sheet_name=sheet_name,
        )

    return server


def analyze_batch_for_mcp(
    inputs,
    ecm_models,
    drt_config,
    output_dir=None,
    ecm_weighting="modulus",
    ecm_max_nfev=1000,
    ecm_acceptable_relative_rmse=0.08,
    ecm_initial=None,
    ecm_bounds=None,
    make_plots=True,
    recursive=False,
    continue_on_error=True,
    freq_col=None,
    real_col=None,
    imag_col=None,
    imag_sign=None,
    sheet_name=None,
    ecm_models_confirmed_by_user=False,
    drt_config_confirmed_by_user=False,
):
    if isinstance(inputs, str):
        inputs = [inputs]
    if not inputs:
        raise ValueError("inputs must contain at least one file, folder, or glob pattern.")
    if not ecm_models:
        raise ValueError("ecm_models is required; call list_ecm_models first and let the user choose.")
    _require_analysis_selection_confirmation(
        ecm_models_confirmed_by_user=ecm_models_confirmed_by_user,
        drt_config_confirmed_by_user=drt_config_confirmed_by_user,
    )

    selected_drt = _drt_config_from_user(drt_config)
    ecm_data = {
        "models": list(ecm_models),
        "weighting": ecm_weighting,
        "max_nfev": ecm_max_nfev,
        "acceptable_relative_rmse": ecm_acceptable_relative_rmse,
        "initial": ecm_initial,
        "bounds": ecm_bounds,
        "strict_bounds": False,
    }
    io_options = _io_options(
        freq_col=freq_col,
        real_col=real_col,
        imag_col=imag_col,
        imag_sign=imag_sign,
        sheet_name=sheet_name,
    )

    result = analyze_batch(
        inputs,
        output_dir=output_dir,
        ecm_config=ecm_configs_from_dict(ecm_data),
        drt_config=selected_drt,
        io_options=io_options,
        make_plots=make_plots,
        recursive=recursive,
        continue_on_error=continue_on_error,
    )
    return _batch_response(result)


_analyze_batch_tool = analyze_batch_for_mcp


def _require_analysis_selection_confirmation(
    ecm_models_confirmed_by_user=False,
    drt_config_confirmed_by_user=False,
):
    _require_ecm_selection_confirmation(ecm_models_confirmed_by_user)
    _require_drt_selection_confirmation(drt_config_confirmed_by_user)


def _require_ecm_selection_confirmation(confirmed):
    if not confirmed:
        raise ValueError(
            "ECM model selection must be confirmed by the user first. "
            "Call get_analysis_choices, show all ECM and DRT choices, let the user choose, then pass "
            "ecm_model_confirmed_by_user=true for fit_ecm or "
            "ecm_models_confirmed_by_user=true for analyze_eis_file/analyze_eis_batch."
        )


def _require_drt_selection_confirmation(confirmed):
    if not confirmed:
        raise ValueError(
            "DRT parameter selection must be confirmed by the user first. "
            "Call get_analysis_choices, show all ECM and DRT choices, let the user choose a preset or custom config, "
            "then pass drt_config_confirmed_by_user=true."
        )


def _drt_config_from_user(drt_config):
    if not drt_config:
        raise ValueError("drt_config is required; call list_drt_options first and let the user choose.")
    missing = [key for key in REQUIRED_DRT_FIELDS if key not in drt_config]
    if missing:
        raise ValueError("drt_config is missing required field(s): %s" % ", ".join(missing))
    return DRTConfig.from_dict(drt_config)


def _data_response(data, source_type, source_path):
    cleaned, duplicate_count = _merge_duplicate_frequencies(data)
    return {
        "success": True,
        "valid": True,
        "source_type": source_type,
        "source_path": source_path,
        "sample_name": _sample_name(source_path),
        "columns": getattr(data, "columns", {}),
        "frequency_hz": cleaned.freq_hz.tolist(),
        "z_real_ohm": np.real(cleaned.z).tolist(),
        "z_imag_ohm": np.imag(cleaned.z).tolist(),
        "imaginary_format": "physical_zimag",
        "cleaning_actions": [
            {
                "action": "normalize_imaginary_sign",
                "imag_multiplier_to_minus_zimag": data.imag_multiplier_to_minus,
            },
            {"action": "sort_frequency_descending", "points": int(cleaned.freq_hz.size)},
        ],
        "warnings": (
            ["Merged %d duplicate frequency point(s)." % duplicate_count]
            if duplicate_count
            else []
        ),
        "summary": {
            "points": int(cleaned.freq_hz.size),
            "freq_min_hz": float(np.min(cleaned.freq_hz)),
            "freq_max_hz": float(np.max(cleaned.freq_hz)),
            "frequency_order": "descending",
        },
    }


def _merge_duplicate_frequencies(data):
    frame = pd.DataFrame(
        {
            "freq_hz": data.freq_hz,
            "z_real_ohm": np.real(data.z),
            "z_imag_ohm": np.imag(data.z),
        }
    )
    before = len(frame)
    frame = frame.groupby("freq_hz", as_index=False).mean()
    frame = frame.sort_values("freq_hz", ascending=False)
    z = frame["z_real_ohm"].values + 1j * frame["z_imag_ohm"].values
    return EISData(data.path, frame["freq_hz"].values, z), before - len(frame)


def _arrays_to_eis(frequency_hz, z_real_ohm, z_imag_ohm):
    freq = np.asarray(frequency_hz, dtype=float)
    z = np.asarray(z_real_ohm, dtype=float) + 1j * np.asarray(z_imag_ohm, dtype=float)
    return freq, z


def _ecm_result_response(result, config, freq_hz):
    zview_fields = _zview_compatibility_fields(result)
    return {
        "success": bool(result.success),
        "model_name": config.model_name,
        "model": config.model,
        "parameters": {key: float(value) for key, value in result.parameters.items()},
        "parameter_names": list(result.parameter_names),
        "parameter_order": zview_fields.pop("ecm_process_order"),
        "zview_compatibility": {
            key: float(value) if isinstance(value, (int, float, np.number)) else value
            for key, value in zview_fields.items()
        },
        "metrics": result.metrics,
        "message": result.message,
        "reconstructed": {
            "freq_hz": np.asarray(freq_hz, dtype=float).tolist(),
            "z_real_ohm": np.real(result.z_fit).tolist(),
            "z_imag_ohm": np.imag(result.z_fit).tolist(),
        },
    }


def _drt_result_response(result, freq_hz, z, requested_config):
    residual = result.z_fit - z
    rel_rmse = float(np.sqrt(np.mean((np.abs(residual) / np.maximum(np.abs(z), 1e-30)) ** 2)))
    return {
        "success": bool(result.success),
        "config": requested_config,
        "message": result.message,
        "metrics": {
            "relative_rmse": rel_rmse,
            "cost": result.cost,
            "polarization_resistance_ohm": result.total_polarization_resistance,
            "n_basis_effective": result.n_basis,
            "lambda_selected": result.lambda_value,
            "lambda_selection": result.lambda_selection,
            "lambda_score": result.lambda_score,
            "normalization_ohm": result.normalization_ohm,
        },
        "r_inf_ohm": result.r_inf,
        "inductance_h": result.inductance,
        "curve": {
            "tau_s": result.tau.tolist(),
            "gamma_ohm": result.gamma.tolist(),
            "integration_weight": result.weights.tolist(),
            "frequency_supported": (
                (result.tau >= result.supported_tau_min)
                & (result.tau <= result.supported_tau_max)
            ).tolist(),
        },
        "frequency_supported_tau_range_s": [
            result.supported_tau_min,
            result.supported_tau_max,
        ],
        "peaks": _find_drt_peaks(result.tau, result.gamma, result.weights),
        "reconstructed": {
            "freq_hz": np.asarray(freq_hz, dtype=float).tolist(),
            "z_real_ohm": np.real(result.z_fit).tolist(),
            "z_imag_ohm": np.imag(result.z_fit).tolist(),
        },
    }


def _find_drt_peaks(tau, gamma, weights):
    peaks = []
    for index in range(1, len(gamma) - 1):
        if gamma[index] >= gamma[index - 1] and gamma[index] >= gamma[index + 1] and gamma[index] > 0:
            peaks.append(
                {
                    "index": index,
                    "tau_s": float(tau[index]),
                    "frequency_hz": float(1.0 / (2.0 * np.pi * tau[index])),
                    "height_ohm": float(gamma[index]),
                    "area_ohm": float(gamma[index] * weights[index]),
                }
            )
    peaks.sort(key=lambda item: item["height_ohm"], reverse=True)
    return peaks[:10]


def _io_options(freq_col=None, real_col=None, imag_col=None, imag_sign=None, sheet_name=None):
    options = {}
    if freq_col is not None:
        options["freq_col"] = freq_col
    if real_col is not None:
        options["real_col"] = real_col
    if imag_col is not None:
        options["imag_col"] = imag_col
    if imag_sign is not None:
        options["imag_sign"] = imag_sign
    if sheet_name is not None:
        options["sheet_name"] = _parse_sheet_name(sheet_name)
    return options


def _parse_sheet_name(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _batch_response(result):
    rows = []
    for record in result.get("rows", []):
        rows.append({key: _json_value(value) for key, value in record.items()})
    output_dir = result["output_dir"]
    readme_path = result.get("readme_path") or os.path.join(output_dir, "README_输出说明.txt")
    return {
        "processed_count": len(result["files"]),
        "output_dir": os.path.abspath(output_dir),
        "readme_path": os.path.abspath(readme_path),
        "input_files": [os.path.abspath(path) for path in result["files"]],
        "samples": rows,
    }


def _json_value(value):
    if pd.isnull(value):
        return None
    try:
        item = value.item()
    except AttributeError:
        item = value
    return item


def _sample_name(path):
    if not path:
        return None
    return os.path.splitext(os.path.basename(path))[0]


def main():
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
