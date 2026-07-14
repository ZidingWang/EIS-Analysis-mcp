import json
import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from .choices import drt_config_to_dict, identify_drt_preset
from .drt import DRTConfig, solve_drt
from .ecm import ECMConfig, ecm_configs_from_dict, fit_ecm
from .io import expand_inputs, read_eis_file
from .paths import ensure_output_dir
from .plotting import plot_drt


def analyze_file(
    input_path,
    output_dir=None,
    ecm_config=None,
    drt_config=None,
    io_options=None,
    make_plots=True,
    write_readme=True,
):
    io_options = dict(io_options or {})
    ecm_configs = _ensure_ecm_configs(ecm_config)
    drt_config = drt_config or DRTConfig()

    output_dir = ensure_output_dir(output_dir, prefix="single")

    data = read_eis_file(input_path, **io_options)
    ecm_results = _fit_ecm_candidates(data.freq_hz, data.z, ecm_configs)
    selected = _select_best_ecm(ecm_results)
    ecm_config = selected["config"]
    ecm_result = selected["result"]
    drt_result = solve_drt(data.freq_hz, data.z, drt_config)

    stem = _safe_stem(input_path)
    paths = _output_paths(output_dir, stem)

    ecm_frame = _selected_ecm_frame(stem, selected)
    ecm_frame.to_csv(paths["ecm"], index=False)

    drt_result.frame().to_csv(paths["drt"], index=False)

    summary = _summary_row(stem, input_path, data, ecm_config, ecm_result, drt_config, drt_result)

    if make_plots:
        plot_drt(drt_result.tau, drt_result.gamma, paths["drt_plot"])

    if write_readme:
        paths["readme"] = _write_output_readme(
            output_dir=output_dir,
            mode="single",
            input_files=[input_path],
            rows=[summary],
            ecm_configs=ecm_configs,
            drt_config=drt_config,
            make_plots=make_plots,
        )

    return {
        "input": input_path,
        "sample": stem,
        "data": data,
        "ecm": ecm_result,
        "ecm_config": ecm_config,
        "ecm_results": ecm_results,
        "drt": drt_result,
        "summary": summary,
        "paths": paths,
    }


def analyze_batch(
    inputs,
    output_dir=None,
    ecm_config=None,
    drt_config=None,
    io_options=None,
    make_plots=True,
    recursive=False,
    continue_on_error=True,
):
    paths = expand_inputs(inputs, recursive=recursive)
    if not paths:
        raise ValueError("No input files found")

    output_dir = ensure_output_dir(output_dir, prefix="batch")
    ecm_configs = _ensure_ecm_configs(ecm_config)

    rows = []
    results = []
    for path in paths:
        try:
            result = analyze_file(
                path,
                output_dir=output_dir,
                ecm_config=ecm_configs,
                drt_config=drt_config,
                io_options=io_options,
                make_plots=make_plots,
                write_readme=False,
            )
            rows.append(result["summary"])
            results.append(result)
        except Exception as exc:
            if not continue_on_error:
                raise
            stem = _safe_stem(path)
            rows.append(
                {
                    "sample": stem,
                    "input_file": os.path.abspath(path),
                    "success": False,
                    "error": str(exc),
                }
            )

    readme_path = _write_output_readme(
        output_dir=output_dir,
        mode="batch",
        input_files=paths,
        rows=rows,
        ecm_configs=ecm_configs,
        drt_config=drt_config or DRTConfig(),
        make_plots=make_plots,
    )
    return {
        "files": paths,
        "results": results,
        "rows": rows,
        "readme_path": readme_path,
        "output_dir": output_dir,
    }


def load_json_config(path):
    with open(path, "r") as handle:
        return json.load(handle)


def config_from_dict(data):
    data = data or {}
    return {
        "inputs": data.get("inputs", []),
        "output_dir": data.get("output_dir"),
        "io_options": data.get("io", {}),
        "ecm_config": ecm_configs_from_dict(data.get("ecm", {})),
        "drt_config": DRTConfig.from_dict(data.get("drt", {})),
        "make_plots": data.get("plots", True),
        "recursive": data.get("recursive", False),
    }


def _summary_row(sample, input_path, data, ecm_config, ecm_result, drt_config, drt_result):
    row = {
        "sample": sample,
        "input_file": os.path.abspath(input_path),
        "success": bool(ecm_result.success and drt_result.success),
        "freq_col": data.columns.get("freq_col"),
        "real_col": data.columns.get("real_col"),
        "imag_col": data.columns.get("imag_col"),
        "imag_multiplier_to_minus_zimag": data.imag_multiplier_to_minus,
        "frequency_points": int(len(data.freq_hz)),
        "frequency_min_hz": float(np.min(data.freq_hz)),
        "frequency_max_hz": float(np.max(data.freq_hz)),
        "selected_ecm_model_name": ecm_config.model_name,
        "ecm_model": ecm_config.model,
        "ecm_rmse_ohm": ecm_result.metrics["rmse_ohm"],
        "ecm_relative_rmse": ecm_result.metrics["relative_rmse"],
        "ecm_r2_complex": ecm_result.metrics["r2_complex"],
        "drt_lambda": drt_result.lambda_value,
        "drt_n_tau": drt_config.n_tau,
        "drt_tau_min_s": float(drt_result.tau[0]),
        "drt_tau_max_s": float(drt_result.tau[-1]),
        "drt_regularization_order": drt_config.regularization_order,
        "drt_basis_function": drt_config.basis_function,
        "drt_shape_factor": drt_config.shape_factor,
        "drt_requested_n_basis": drt_config.n_basis,
        "drt_n_basis": drt_result.n_basis,
        "drt_polarization_removal": drt_config.polarization_removal,
        "drt_boundary_suppression_factor": drt_config.boundary_suppression_factor,
        "drt_nonnegative": drt_config.nonnegative,
        "drt_fit_r_inf": drt_config.fit_r_inf,
        "drt_fit_inductance": drt_config.fit_inductance,
        "drt_weighting": drt_config.weighting,
        "drt_r_inf_ohm": drt_result.r_inf,
        "drt_inductance_h": drt_result.inductance,
        "drt_polarization_resistance_ohm": drt_result.total_polarization_resistance,
        "error": "",
    }
    for name in ecm_result.parameter_names:
        row["ecm_" + name] = ecm_result.parameters[name]
    return row


def _output_paths(output_dir, stem):
    return {
        "ecm": os.path.join(output_dir, stem + "_ecm.csv"),
        "drt": os.path.join(output_dir, stem + "_drt.csv"),
        "drt_plot": os.path.join(output_dir, stem + "_drt.png"),
    }


def _write_output_readme(
    output_dir,
    mode,
    input_files,
    rows,
    ecm_configs,
    drt_config,
    make_plots,
):
    path = os.path.join(output_dir, "README_输出说明.txt")
    rows = list(rows or [])
    statistics = _readme_statistics(rows)
    failed = [row for row in rows if not bool(row.get("success", False))]
    preset_id = identify_drt_preset(drt_config)
    preset_label = "balanced（推荐）" if preset_id == "balanced" else preset_id
    candidate_text = ", ".join("%s (%s)" % (config.model_name, config.model) for config in ecm_configs)

    lines = [
        "EIS 分析输出说明",
        "生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "输出目录: %s" % os.path.abspath(output_dir),
        "分析模式: %s" % ("批量处理" if mode == "batch" else "单文件处理"),
        "",
        "输出文件",
        "- <样本名>_ecm.csv: 最终选中的一组 ECM 模型参数和拟合指标，每个文件只有一行。",
        "- <样本名>_drt.csv: DRT 曲线数据，包含 tau_s、gamma_ohm 和 integration_weight。",
        "- <样本名>_drt.png: DRT 曲线图。" if make_plots else "- 本次已关闭图片生成，因此没有 DRT PNG。",
        "- README_输出说明.txt: 本说明文件。",
        "",
        "本次选择",
        "- ECM 候选模型: %s" % candidate_text,
        "- 多个 ECM 候选模型按 relative_rmse 选择最佳结果。",
        "- DRT 预设: %s" % preset_label,
        "- DRT 完整参数: %s" % json.dumps(drt_config_to_dict(drt_config), ensure_ascii=False, sort_keys=True),
        "",
        "数据统计",
        "- 输入 EIS 曲线数: %d" % statistics["curve_count"],
        "- 成功曲线数: %d" % statistics["success_count"],
        "- 失败曲线数: %d" % statistics["failure_count"],
        "- 总频率点数: %d" % statistics["total_points"],
    ]

    if statistics["points_min"] is not None:
        lines.extend(
            [
                "- 单条曲线频率点数: 最小 %s，中位数 %s，最大 %s"
                % (
                    _format_readme_number(statistics["points_min"]),
                    _format_readme_number(statistics["points_median"]),
                    _format_readme_number(statistics["points_max"]),
                ),
                "- 总频率范围: %s–%s Hz"
                % (
                    _format_readme_number(statistics["frequency_min_hz"]),
                    _format_readme_number(statistics["frequency_max_hz"]),
                ),
                "- DRT tau 总范围: %s–%s s"
                % (
                    _format_readme_number(statistics["tau_min_s"]),
                    _format_readme_number(statistics["tau_max_s"]),
                ),
                "- DRT 输出网格点数: %s"
                % ", ".join(_format_readme_number(value) for value in statistics["drt_n_tau"]),
            ]
        )

    lines.extend(["", "ECM 结果统计"])
    for model_name, count in sorted(statistics["model_counts"].items()):
        lines.append("- %s: %d 组" % (model_name, count))

    if statistics["rmse_mean"] is not None:
        lines.extend(
            [
                "- relative RMSE 平均值: %s" % _format_readme_number(statistics["rmse_mean"]),
                "- relative RMSE 中位数: %s" % _format_readme_number(statistics["rmse_median"]),
                "- relative RMSE 最大值: %s" % _format_readme_number(statistics["rmse_max"]),
                "- R² 平均值: %s" % _format_readme_number(statistics["r2_mean"]),
                "- R² 最小值: %s" % _format_readme_number(statistics["r2_min"]),
            ]
        )

    lines.extend(["", "样本结果"])

    for row in rows:
        if row.get("success"):
            lines.append(
                "- %s: ECM=%s, relative_rmse=%s, R²=%s"
                % (
                    row.get("sample", ""),
                    row.get("selected_ecm_model_name", ""),
                    _format_readme_number(row.get("ecm_relative_rmse")),
                    _format_readme_number(row.get("ecm_r2_complex")),
                )
            )

    if failed:
        lines.extend(["", "失败样本"])
        for row in failed:
            lines.append("- %s: %s" % (row.get("sample", ""), row.get("error", "")))

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _readme_statistics(rows):
    successful = [row for row in rows if bool(row.get("success", False))]

    def values(key):
        output = []
        for row in successful:
            value = row.get(key)
            if value is not None and not pd.isnull(value):
                output.append(float(value))
        return np.asarray(output, dtype=float)

    points = values("frequency_points")
    rmse = values("ecm_relative_rmse")
    r2 = values("ecm_r2_complex")
    frequency_min = values("frequency_min_hz")
    frequency_max = values("frequency_max_hz")
    tau_min = values("drt_tau_min_s")
    tau_max = values("drt_tau_max_s")
    drt_n_tau = values("drt_n_tau")

    return {
        "curve_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "total_points": int(np.sum(points)) if points.size else 0,
        "points_min": int(np.min(points)) if points.size else None,
        "points_median": float(np.median(points)) if points.size else None,
        "points_max": int(np.max(points)) if points.size else None,
        "frequency_min_hz": float(np.min(frequency_min)) if frequency_min.size else None,
        "frequency_max_hz": float(np.max(frequency_max)) if frequency_max.size else None,
        "model_counts": Counter(row.get("selected_ecm_model_name", "") for row in successful),
        "rmse_mean": float(np.mean(rmse)) if rmse.size else None,
        "rmse_median": float(np.median(rmse)) if rmse.size else None,
        "rmse_max": float(np.max(rmse)) if rmse.size else None,
        "r2_mean": float(np.mean(r2)) if r2.size else None,
        "r2_min": float(np.min(r2)) if r2.size else None,
        "tau_min_s": float(np.min(tau_min)) if tau_min.size else None,
        "tau_max_s": float(np.max(tau_max)) if tau_max.size else None,
        "drt_n_tau": sorted({int(value) for value in drt_n_tau}),
    }


def _format_readme_number(value):
    if value is None:
        return "None"
    return "%.6g" % float(value)


def _format_readme_value(value):
    if value is None:
        return "None"
    return str(value)


def _ensure_ecm_configs(ecm_config):
    if ecm_config is None:
        return [ECMConfig()]
    if isinstance(ecm_config, ECMConfig):
        return [ecm_config]
    if isinstance(ecm_config, (list, tuple)):
        configs = []
        for item in ecm_config:
            if isinstance(item, ECMConfig):
                configs.append(item)
            else:
                configs.append(ECMConfig(model=item))
        if not configs:
            return [ECMConfig()]
        return configs
    return [ECMConfig(model=ecm_config)]


def _fit_ecm_candidates(freq_hz, z, ecm_configs):
    results = []
    for config in ecm_configs:
        result = fit_ecm(freq_hz, z, config)
        results.append({"config": config, "result": result})
    return results


def _select_best_ecm(ecm_results):
    successful = [item for item in ecm_results if item["result"].success]
    candidates = successful or list(ecm_results)
    candidates.sort(key=lambda item: _metric_for_sort(item["result"]))
    return candidates[0]


def _metric_for_sort(result):
    value = result.metrics.get("relative_rmse")
    if value is None or not np.isfinite(value):
        return np.inf
    return value


def _selected_ecm_frame(sample, selected):
    config = selected["config"]
    result = selected["result"]
    row = {
        "sample": sample,
        "selected": True,
        "model_name": config.model_name,
        "model": config.model,
        "success": bool(result.success),
        "message": result.message,
        "cost": result.cost,
        "rmse_ohm": result.metrics["rmse_ohm"],
        "relative_rmse": result.metrics["relative_rmse"],
        "r2_complex": result.metrics["r2_complex"],
    }
    for name in result.parameter_names:
        row[name] = result.parameters[name]
    return pd.DataFrame([row])


def _safe_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = []
    for char in stem:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "sample"
