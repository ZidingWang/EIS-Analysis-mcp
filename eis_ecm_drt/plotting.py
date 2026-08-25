import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_nyquist(freq_hz, z, ecm_z=None, drt_z=None, output_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=150)
    ax.plot(np.real(z), -np.imag(z), "o", ms=4, label="EIS")
    if ecm_z is not None:
        ax.plot(np.real(ecm_z), -np.imag(ecm_z), "-", lw=1.8, label="ECM fit")
    if drt_z is not None:
        ax.plot(np.real(drt_z), -np.imag(drt_z), "--", lw=1.5, label="DRT fit")
    ax.set_xlabel("Z' / Ohm")
    ax.set_ylabel("-Z'' / Ohm")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_ecm_compare(freq_hz, z, ecm_results, output_path=None):
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=150)
    ax.plot(np.real(z), -np.imag(z), "o", ms=4, label="EIS")
    for item in ecm_results:
        result = item["result"]
        config = item["config"]
        if not result.success:
            continue
        label = "%s (%.3g)" % (
            config.model_name,
            result.metrics["relative_rmse"],
        )
        ax.plot(np.real(result.z_fit), -np.imag(result.z_fit), lw=1.4, label=label)
    ax.set_xlabel("Z' / Ohm")
    ax.set_ylabel("-Z'' / Ohm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_bode(freq_hz, z, ecm_z=None, drt_z=None, output_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), dpi=150, sharex=True)
    axes[0].loglog(freq_hz, np.abs(z), "o", ms=4, label="EIS")
    axes[1].semilogx(freq_hz, np.angle(z, deg=True), "o", ms=4, label="EIS")

    if ecm_z is not None:
        axes[0].loglog(freq_hz, np.abs(ecm_z), "-", lw=1.8, label="ECM fit")
        axes[1].semilogx(freq_hz, np.angle(ecm_z, deg=True), "-", lw=1.8, label="ECM fit")
    if drt_z is not None:
        axes[0].loglog(freq_hz, np.abs(drt_z), "--", lw=1.5, label="DRT fit")
        axes[1].semilogx(freq_hz, np.angle(drt_z, deg=True), "--", lw=1.5, label="DRT fit")

    axes[0].set_ylabel("|Z| / Ohm")
    axes[1].set_ylabel("Phase / deg")
    axes[1].set_xlabel("Frequency / Hz")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_drt(
    tau,
    gamma,
    output_path=None,
    supported_tau_min=None,
    supported_tau_max=None,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    ax.semilogx(tau, gamma, "-", lw=1.8)
    ax.fill_between(tau, 0.0, gamma, alpha=0.18)
    limit_label_added = False
    if supported_tau_min is not None and float(supported_tau_min) > float(np.min(tau)):
        ax.axvspan(float(np.min(tau)), float(supported_tau_min), color="0.5", alpha=0.08)
        ax.axvline(
            float(supported_tau_min),
            color="0.45",
            ls=":",
            lw=1.0,
            label="frequency-supported limit",
        )
        limit_label_added = True
    if supported_tau_max is not None and float(supported_tau_max) < float(np.max(tau)):
        ax.axvspan(float(supported_tau_max), float(np.max(tau)), color="0.5", alpha=0.08)
        ax.axvline(
            float(supported_tau_max),
            color="0.45",
            ls=":",
            lw=1.0,
            label=None if limit_label_added else "frequency-supported limit",
        )
        limit_label_added = True
    ax.set_xlabel("Relaxation time tau / s")
    ax.set_ylabel("gamma(tau) / Ohm")
    ax.grid(True, which="both", alpha=0.25)
    if limit_label_added:
        ax.legend(fontsize=8)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
