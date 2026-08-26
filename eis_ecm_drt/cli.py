import argparse
import json
import os

from .drt import DRTConfig
from .ecm import PUBLIC_ECM_MODEL_IDS, ecm_configs_from_dict, ecm_model_presets, parse_model_list
from .pipeline import analyze_batch, config_from_dict, load_json_config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_ecm_models:
        _print_ecm_models()
        return

    raw_config = load_json_config(args.config) if args.config else {}
    config = config_from_dict(raw_config)

    inputs = args.inputs or config["inputs"]
    if not inputs:
        parser.error("Please provide at least one input file, folder, or glob pattern")

    output_dir = args.output_dir or config["output_dir"]
    io_options = dict(config["io_options"])
    _override_io_options(io_options, args)

    ecm_data = dict(raw_config.get("ecm", {}))
    drt_data = dict(raw_config.get("drt", {}))
    _override_ecm_options(ecm_data, args)
    _override_drt_options(drt_data, args)

    make_plots = config["make_plots"] and not args.no_plots
    recursive = bool(args.recursive or config["recursive"])

    result = analyze_batch(
        inputs,
        output_dir=output_dir,
        ecm_config=ecm_configs_from_dict(ecm_data),
        drt_config=DRTConfig.from_dict(drt_data),
        io_options=io_options,
        make_plots=make_plots,
        recursive=recursive,
        continue_on_error=not args.stop_on_error,
    )

    print("Processed %d input file(s)." % len(result["files"]))
    print("Output directory: %s" % os.path.abspath(result["output_dir"]))
    print("README: %s" % os.path.abspath(result["readme_path"]))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fit ECM parameters and calculate DRT curves from EIS data."
    )
    parser.add_argument("inputs", nargs="*", help="Input file(s), folder(s), or glob(s).")
    parser.add_argument("-c", "--config", help="JSON config file.")
    parser.add_argument("-o", "--output-dir", help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first error.")

    parser.add_argument("--freq-col", help="Frequency column name.")
    parser.add_argument("--real-col", help="Real impedance column name.")
    parser.add_argument("--imag-col", help="Imaginary impedance column name.")
    parser.add_argument(
        "--imag-sign",
        type=float,
        help="Multiplier that converts the source imaginary column to -Zimag. Usually auto-detected.",
    )
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name or index.")

    parser.add_argument(
        "--list-ecm-models",
        action="store_true",
        help="List built-in ECM model choices and exit.",
    )
    parser.add_argument(
        "--ecm-model",
        action="append",
        help="ECM model choice or custom expression. Repeat this option to fit multiple models.",
    )
    parser.add_argument(
        "--ecm-models",
        help='Comma-separated ECM model choices, e.g. L-R0-RWQ,L-R0-2RQ,"L1-R0-((R1-W1)||CPE1)".',
    )
    parser.add_argument("--ecm-initial", help="JSON object/file for initial ECM parameters.")
    parser.add_argument("--ecm-bounds", help="JSON object/file for ECM parameter bounds.")
    parser.add_argument(
        "--weighting",
        choices=["modulus", "none", "real-imag"],
        help="ECM fitting weighting.",
    )
    parser.add_argument("--ecm-max-nfev", type=int, help="Maximum ECM optimizer evaluations.")
    parser.add_argument(
        "--ecm-acceptable-relative-rmse",
        type=float,
        help="Treat ECM as successful when relative RMSE is below this value.",
    )

    parser.add_argument("--drt-lambda", type=float, help="DRT regularization strength.")
    parser.add_argument(
        "--drt-lambda-selection",
        choices=["mgcv", "gcv", "fixed"],
        help="Automatic or fixed DRT regularization selection.",
    )
    parser.add_argument("--n-tau", type=int, help="Number of DRT tau grid points.")
    parser.add_argument(
        "--n-basis",
        type=int,
        help="Gaussian basis count. Default: one center per measured frequency; use 0 for automatic.",
    )
    parser.add_argument("--tau-min", type=float, help="Minimum tau in seconds.")
    parser.add_argument("--tau-max", type=float, help="Maximum tau in seconds.")
    parser.add_argument(
        "--regularization-order",
        type=int,
        choices=[0, 1, 2],
        help="DRT regularization order.",
    )
    parser.add_argument(
        "--basis-function",
        choices=["gaussian", "delta", "none"],
        help="DRT basis function.",
    )
    parser.add_argument(
        "--shape-factor", type=float,
        help="Gaussian FWHM coefficient; FWHM equals log-tau spacing divided by this value.",
    )
    parser.add_argument(
        "--tau-padding-decades", type=float,
        help="Plot/CSV padding outside the frequency-supported tau range.",
    )
    parser.add_argument(
        "--boundary-suppression-factor",
        type=float,
        help="Strength multiplier for ignore_polarization boundary suppression.",
    )
    parser.add_argument(
        "--polarization-removal",
        choices=[
            "ignore_polarization",
            "ignore",
            "none",
            "boundary_suppression",
        ],
        help="DRT polarization handling.",
    )
    parser.add_argument(
        "--allow-negative-drt",
        action="store_true",
        help="Allow negative DRT gamma values.",
    )
    parser.add_argument("--fit-inductance", action="store_true", help="Fit series inductance in DRT.")
    parser.add_argument(
        "--drt-weighting",
        choices=["modulus", "none"],
        help="DRT fitting weighting.",
    )
    parser.add_argument("--drt-max-nfev", type=int, help="Maximum DRT optimizer evaluations.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser


def _override_io_options(io_options, args):
    if args.freq_col:
        io_options["freq_col"] = args.freq_col
    if args.real_col:
        io_options["real_col"] = args.real_col
    if args.imag_col:
        io_options["imag_col"] = args.imag_col
    if args.imag_sign is not None:
        io_options["imag_sign"] = args.imag_sign
    if args.sheet_name is not None:
        io_options["sheet_name"] = _parse_sheet_name(args.sheet_name)


def _override_ecm_options(ecm_data, args):
    selected_models = []
    if args.ecm_model:
        selected_models.extend(parse_model_list(args.ecm_model))
    if args.ecm_models:
        selected_models.extend(parse_model_list(args.ecm_models))
    if selected_models:
        ecm_data["models"] = selected_models
        ecm_data.pop("model", None)
    if args.ecm_initial:
        ecm_data["initial"] = _load_json_value(args.ecm_initial)
    if args.ecm_bounds:
        ecm_data["bounds"] = _load_json_value(args.ecm_bounds)
    if args.weighting:
        ecm_data["weighting"] = args.weighting
    if args.ecm_max_nfev:
        ecm_data["max_nfev"] = args.ecm_max_nfev
    if args.ecm_acceptable_relative_rmse is not None:
        ecm_data["acceptable_relative_rmse"] = args.ecm_acceptable_relative_rmse


def _override_drt_options(drt_data, args):
    if args.drt_lambda is not None:
        drt_data["lambda_value"] = args.drt_lambda
        if not args.drt_lambda_selection:
            drt_data["lambda_selection"] = "fixed"
    if args.drt_lambda_selection:
        drt_data["lambda_selection"] = args.drt_lambda_selection
    if args.n_tau is not None:
        drt_data["n_tau"] = args.n_tau
    if args.n_basis is not None:
        drt_data["n_basis"] = args.n_basis
    if args.tau_min is not None:
        drt_data["tau_min"] = args.tau_min
    if args.tau_max is not None:
        drt_data["tau_max"] = args.tau_max
    if args.tau_padding_decades is not None:
        drt_data["tau_padding_decades"] = args.tau_padding_decades
    if args.regularization_order is not None:
        drt_data["regularization_order"] = args.regularization_order
    if args.basis_function:
        drt_data["basis_function"] = args.basis_function
    if args.shape_factor is not None:
        drt_data["shape_factor"] = args.shape_factor
    if args.boundary_suppression_factor is not None:
        drt_data["boundary_suppression_factor"] = args.boundary_suppression_factor
    if args.polarization_removal:
        drt_data["polarization_removal"] = args.polarization_removal
    if args.allow_negative_drt:
        drt_data["nonnegative"] = False
    if args.fit_inductance:
        drt_data["fit_inductance"] = True
    if args.drt_weighting:
        drt_data["weighting"] = args.drt_weighting
    if args.drt_max_nfev:
        drt_data["max_nfev"] = args.drt_max_nfev


def _load_json_value(value):
    if os.path.isfile(value):
        with open(value, "r") as handle:
            return json.load(handle)
    return json.loads(value)


def _parse_sheet_name(value):
    try:
        return int(value)
    except ValueError:
        return value


def _print_ecm_models():
    print("Built-in ECM models:")
    presets = {item["name"]: item for item in ecm_model_presets()}
    for name in PUBLIC_ECM_MODEL_IDS:
        item = presets[name]
        print("  {name:12s} {expression:35s} {description}".format(**item))
