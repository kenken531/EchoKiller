"""
EchoKiller  —  Windows Edition
================================
Load an audio file with echo (or generate a synthetic one).
Apply an adaptive FIR filter to attenuate the echo.
Display before/after waveforms side-by-side.
Visualize the filter coefficients (impulse response).

Install dependencies:
    pip install scipy numpy soundfile matplotlib sounddevice

Usage:
    python echokiller.py                          # generate synthetic echo + filter it
    python echokiller.py --input music.mp3        # use your own WAV/MP3 file
    python echokiller.py --order 64 --delay 0.15  # tune filter parameters
    python echokiller.py --no-play                # skip audio playback
    python echokiller.py --save                   # save filtered output to WAV
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import signal
from pathlib import Path

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_ORDER  = 64        # FIR filter order (number of taps)
DEFAULT_DELAY  = 0.12      # simulated echo delay in seconds
DEFAULT_DECAY  = 0.45      # echo amplitude decay (0=no echo, 1=full echo)
SAMPLE_RATE    = 16000     # sample rate for synthetic audio

# LMS adaptive filter config
LMS_MU         = 0.002     # step size (learning rate) — reduce if unstable
LMS_ITERATIONS = 3         # passes over the signal (more = better convergence)

# ── Colour palette ────────────────────────────────────────────────────────────

BG        = "#0a0a10"
PANEL_BG  = "#0d0d1a"
C_ORIG    = "#00e5ff"      # original signal cyan
C_ECHO    = "#ff4081"      # echo-corrupted signal pink
C_FILT    = "#69ff47"      # filtered signal green
C_COEFF   = "#ffcc00"      # filter coefficients amber
C_TEXT    = "#ccccdd"
C_DIM     = "#555566"
C_GRID    = "#1a1a2e"

# ── Signal generation ─────────────────────────────────────────────────────────

def generate_speech_like(duration: float, sample_rate: int) -> np.ndarray:
    """
    Generate a synthetic speech-like signal:
    sum of sine waves at voice frequencies with amplitude envelope,
    plus band-limited noise for consonant-like bursts.
    This sounds vaguely like speech and works well for demo purposes.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Voice frequencies (fundamental + harmonics typical for speech)
    voice_freqs = [120, 240, 360, 480, 800, 1200, 2400]
    voice_amps  = [0.6,  0.4,  0.3,  0.2,  0.15,  0.1,  0.08]

    signal_out = np.zeros_like(t)
    for f, a in zip(voice_freqs, voice_amps):
        phase = np.random.uniform(0, 2 * np.pi)
        signal_out += a * np.sin(2 * np.pi * f * t + phase)

    # Amplitude envelope: syllable-like bursts
    envelope = np.zeros_like(t)
    syllable_rate = 4.0   # syllables per second
    for i in range(int(duration * syllable_rate)):
        center = i / syllable_rate + np.random.uniform(-0.05, 0.05)
        width  = np.random.uniform(0.06, 0.12)
        envelope += np.exp(-0.5 * ((t - center) / width) ** 2)

    envelope = np.clip(envelope, 0, 1)
    signal_out *= envelope

    # Add some noise bursts (consonants)
    noise = np.random.randn(len(t)) * 0.05
    burst_mask = np.random.rand(len(t)) < 0.05
    noise *= burst_mask.astype(float)
    signal_out += np.convolve(noise, np.ones(50) / 50, mode='same')

    # Normalise to [-0.8, 0.8]
    peak = np.max(np.abs(signal_out))
    if peak > 0:
        signal_out = signal_out / peak * 0.8

    return signal_out.astype(np.float32)


def add_echo(signal_in: np.ndarray, sample_rate: int,
             delay_sec: float, decay: float,
             num_echoes: int = 3) -> np.ndarray:
    """
    Add multiple echoes to a signal.
    Each successive echo is delayed and attenuated further.
    This simulates a reverberant room.

    signal_out[n] = signal_in[n]
                  + decay   * signal_in[n - delay_samples]
                  + decay^2 * signal_in[n - 2*delay_samples]
                  + ...
    """
    delay_samples = int(delay_sec * sample_rate)
    result = signal_in.copy()

    for i in range(1, num_echoes + 1):
        shift   = delay_samples * i
        amp     = decay ** i
        padding = np.zeros(shift, dtype=np.float32)
        delayed = np.concatenate([padding, signal_in])[: len(signal_in)]
        result  = result + amp * delayed

    # Normalise to prevent clipping
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.8

    return result.astype(np.float32)

# ── FIR filter design ─────────────────────────────────────────────────────────

def design_wiener_fir(echo_signal: np.ndarray, clean_reference: np.ndarray,
                      order: int) -> np.ndarray:
    """
    Design a Wiener FIR filter using the Wiener-Hopf equations.

    The Wiener filter minimises the mean-square error between the filter
    output and the desired signal. It solves:

        R * h = p

    where:
        R = autocorrelation matrix of the echo signal (Toeplitz)
        p = cross-correlation vector between echo signal and clean reference
        h = optimal filter coefficients

    This is the closed-form optimal linear filter — the same math
    inside every AEC chip, just solved analytically instead of adaptively.
    """
    n = len(echo_signal)

    # Autocorrelation of the echo signal
    # R[k] = E[x[n] * x[n-k]]  for k = 0, 1, ..., order
    autocorr = np.correlate(echo_signal, echo_signal, mode='full')
    mid       = len(autocorr) // 2
    r_vec     = autocorr[mid : mid + order + 1] / n

    # Build Toeplitz autocorrelation matrix R (symmetric)
    R = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            R[i, j] = r_vec[abs(i - j)]

    # Cross-correlation between echo signal and clean reference
    # p[k] = E[d[n] * x[n-k]]
    crosscorr = np.correlate(clean_reference, echo_signal, mode='full')
    mid_c     = len(crosscorr) // 2
    p_vec     = crosscorr[mid_c : mid_c + order + 1] / n

    # Solve the Wiener-Hopf equation: R * h = p
    # Add small diagonal regularisation (Tikhonov) to ensure invertibility
    reg = 1e-6 * np.eye(order + 1)
    try:
        h = np.linalg.solve(R + reg, p_vec)
    except np.linalg.LinAlgError:
        # Fallback: use least-squares if matrix is singular
        h, _, _, _ = np.linalg.lstsq(R + reg, p_vec, rcond=None)

    return h.astype(np.float64)


def lms_adaptive_filter(echo_signal: np.ndarray, clean_reference: np.ndarray,
                        order: int, mu: float,
                        iterations: int = 1) -> tuple:
    """
    LMS (Least Mean Squares) adaptive FIR filter.

    The LMS algorithm iteratively adjusts filter weights to minimise
    the error between the filter output and the desired signal.

    Update rule:
        e[n]  = d[n] - h^T * x[n]       (error signal)
        h     = h + mu * e[n] * x[n]    (weight update)

    This is the same algorithm in every real-time AEC system — the
    closed-form Wiener solution can't be used in real-time because
    you don't have the full signal yet.

    Returns (filtered_signal, final_weights, error_history)
    """
    n          = len(echo_signal)
    h          = np.zeros(order + 1, dtype=np.float64)
    x_buf      = np.zeros(order + 1, dtype=np.float64)
    filtered   = np.zeros(n, dtype=np.float32)
    errors     = []

    for iteration in range(iterations):
        h_buf  = np.zeros(order + 1, dtype=np.float64)
        x_buf  = np.zeros(order + 1, dtype=np.float64)

        for i in range(n):
            # Shift input buffer
            x_buf[1:] = x_buf[:-1]
            x_buf[0]  = echo_signal[i]

            # Filter output: y[n] = h^T * x[n]
            y = np.dot(h, x_buf)

            # Error: e[n] = d[n] - y[n]
            e = float(clean_reference[i]) - y

            # LMS weight update
            h = h + mu * e * x_buf

            if iteration == iterations - 1:
                filtered[i] = float(y)
                errors.append(e ** 2)

    return filtered, h, np.array(errors)

# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple:
    """Load a WAV file and return (samples, sample_rate)."""
    if not HAS_SOUNDFILE:
        print("ERROR: soundfile not installed. Run: pip install soundfile")
        sys.exit(1)
    try:
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]   # take left channel if stereo
        return data, sr
    except Exception as e:
        print(f"ERROR: Cannot load '{path}': {e}")
        print("Note: soundfile supports WAV, FLAC, OGG. For MP3, convert first:")
        print("  ffmpeg -i input.mp3 output.wav")
        sys.exit(1)


def save_audio(path: str, data: np.ndarray, sample_rate: int):
    """Save audio to WAV."""
    if not HAS_SOUNDFILE:
        return
    try:
        sf.write(path, data.astype(np.float32), sample_rate)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Warning: Could not save audio: {e}")


def play_audio(data: np.ndarray, sample_rate: int, label: str = ""):
    """Play audio through speakers."""
    if not HAS_SOUNDDEVICE:
        return
    try:
        print(f"  Playing {label}...", end="", flush=True)
        sd.play(data.astype(np.float32), sample_rate)
        sd.wait()
        print(" done.")
    except Exception as e:
        print(f"\n  Warning: Playback failed: {e}")

# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(clean: np.ndarray, echo: np.ndarray, filtered: np.ndarray,
                 coeffs: np.ndarray, errors: np.ndarray,
                 sample_rate: int, title: str = "EchoKiller"):
    """
    Create a 2×3 figure:
      Row 1: clean waveform | echo waveform | filtered waveform
      Row 2: filter coefficients | convergence curve | frequency response
    """
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(
        "EchoKiller  —  Acoustic Echo Cancellation  |  BUILDCORED ORCAS"
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97,
        top=0.90,  bottom=0.08,
        wspace=0.35, hspace=0.55,
    )

    def setup_ax(ax, title_text, xlabel="", ylabel="", color=C_TEXT):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title_text, color=color, fontsize=9, pad=6, loc="left")
        ax.set_xlabel(xlabel, color=C_DIM, fontsize=8)
        ax.set_ylabel(ylabel, color=C_DIM, fontsize=8)
        ax.tick_params(colors=C_DIM, labelsize=7)
        ax.spines[:].set_color(C_DIM)
        ax.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")

    # Time axis
    t_clean = np.linspace(0, len(clean)  / sample_rate, len(clean))
    t_echo  = np.linspace(0, len(echo)   / sample_rate, len(echo))
    t_filt  = np.linspace(0, len(filtered)/ sample_rate, len(filtered))

    # ── Row 1: Waveforms ──────────────────────────────────────────────────────

    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, "① Original / Reference Signal", "Time (s)", "Amplitude", C_ORIG)
    ax1.plot(t_clean, clean, color=C_ORIG, linewidth=0.8, alpha=0.9)
    ax1.fill_between(t_clean, clean, alpha=0.1, color=C_ORIG)
    ax1.set_ylim(-1.05, 1.05)

    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "② Echo-Corrupted Signal (Input)", "Time (s)", "Amplitude", C_ECHO)
    ax2.plot(t_echo, echo, color=C_ECHO, linewidth=0.8, alpha=0.9)
    ax2.fill_between(t_echo, echo, alpha=0.1, color=C_ECHO)
    ax2.set_ylim(-1.05, 1.05)

    ax3 = fig.add_subplot(gs[0, 2])
    setup_ax(ax3, "③ Echo-Cancelled Output (Filtered)", "Time (s)", "Amplitude", C_FILT)
    ax3.plot(t_filt, filtered, color=C_FILT, linewidth=0.8, alpha=0.9)
    ax3.fill_between(t_filt, filtered, alpha=0.1, color=C_FILT)
    ax3.set_ylim(-1.05, 1.05)

    # ── Row 2: Filter analysis ────────────────────────────────────────────────

    # Filter coefficients (impulse response)
    ax4 = fig.add_subplot(gs[1, 0])
    setup_ax(ax4, "④ FIR Impulse Response (Coefficients)", "Tap index", "Weight", C_COEFF)
    tap_idx = np.arange(len(coeffs))
    colors  = [C_COEFF if c >= 0 else C_ECHO for c in coeffs]
    ax4.bar(tap_idx, coeffs, color=colors, alpha=0.8, width=0.8)
    ax4.axhline(0, color=C_DIM, linewidth=0.8)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # LMS convergence curve
    ax5 = fig.add_subplot(gs[1, 1])
    setup_ax(ax5, "⑤ LMS Convergence (Error²)", "Sample", "MSE", C_TEXT)
    if len(errors) > 0:
        # Smooth for display
        window_size = max(1, len(errors) // 200)
        smoothed    = np.convolve(errors,
                                  np.ones(window_size) / window_size,
                                  mode='valid')
        t_err = np.linspace(0, len(errors), len(smoothed))
        ax5.semilogy(t_err, smoothed + 1e-12, color=C_COEFF, linewidth=1.0)
        ax5.set_ylabel("MSE (log scale)", color=C_DIM, fontsize=8)

    # Frequency response of the filter
    ax6 = fig.add_subplot(gs[1, 2])
    setup_ax(ax6, "⑥ Filter Frequency Response", "Frequency (Hz)", "|H(f)| dB", C_TEXT)
    w, h_resp = signal.freqz(coeffs, worN=1024, fs=sample_rate)
    h_db      = 20 * np.log10(np.maximum(np.abs(h_resp), 1e-10))
    ax6.plot(w, h_db, color=C_FILT, linewidth=1.0)
    ax6.axhline(0,   color=C_DIM, linewidth=0.6, linestyle=":")
    ax6.axhline(-6,  color=C_DIM, linewidth=0.5, linestyle=":")
    ax6.axhline(-20, color=C_DIM, linewidth=0.5, linestyle=":")
    ax6.set_ylim(-60, 10)

    # ── Stats overlay ─────────────────────────────────────────────────────────
    echo_power   = 10 * np.log10(np.mean(echo ** 2)     + 1e-12)
    filt_power   = 10 * np.log10(np.mean(filtered ** 2) + 1e-12)
    reduction_db = echo_power - filt_power
    snr_before   = _snr(clean[:len(echo)],     echo)
    snr_after    = _snr(clean[:len(filtered)], filtered)

    stats = (
        f"Echo power:  {echo_power:.1f} dBFS  →  {filt_power:.1f} dBFS  "
        f"({reduction_db:.1f} dB reduction)   "
        f"|   SNR:  {snr_before:.1f} dB  →  {snr_after:.1f} dB"
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.96, "EchoKiller", ha="center", va="top",
             fontsize=17, fontweight="bold", color=C_FILT,
             fontfamily="monospace")
    fig.text(0.5, 0.93, "Acoustic Echo Cancellation  ·  FIR Wiener Filter + LMS  ·  BUILDCORED ORCAS",
             ha="center", va="top", fontsize=8, color=C_DIM)
    fig.text(0.5, 0.905, stats, ha="center", va="top",
             fontsize=8, color=C_TEXT)

    plt.show()


def _snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-noise ratio in dB. Clips to a reasonable range."""
    n = min(len(clean), len(noisy))
    if n == 0:
        return 0.0
    s = clean[:n]
    n_sig = noisy[:n] - s
    signal_power = np.mean(s ** 2)
    noise_power  = np.mean(n_sig ** 2)
    if noise_power < 1e-12:
        return 60.0
    return float(np.clip(10 * np.log10(signal_power / noise_power), -60, 60))

# ── Console banner ────────────────────────────────────────────────────────────

def print_banner(args):
    print("\n" + "─" * 58)
    print("  EchoKiller  ·  Acoustic Echo Cancellation")
    print("  Day 16 — BUILDCORED ORCAS")
    print("─" * 58)
    if args.input:
        print(f"  Input file  : {args.input}")
    else:
        print(f"  Mode        : synthetic echo generation")
        print(f"  Echo delay  : {args.delay * 1000:.0f} ms")
        print(f"  Echo decay  : {args.decay:.2f}")
    print(f"  Filter order: {args.order} taps")
    print(f"  LMS mu      : {LMS_MU}")
    print(f"  LMS passes  : {LMS_ITERATIONS}")
    print("─" * 58 + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EchoKiller — adaptive FIR acoustic echo cancellation"
    )
    parser.add_argument("--input",  "-i", default=None,
                        help="Input WAV file (default: generate synthetic echo)")
    parser.add_argument("--order",  "-o", type=int, default=DEFAULT_ORDER,
                        help=f"FIR filter order/taps (default: {DEFAULT_ORDER})")
    parser.add_argument("--delay",  "-d", type=float, default=DEFAULT_DELAY,
                        help=f"Echo delay in seconds for synthesis (default: {DEFAULT_DELAY})")
    parser.add_argument("--decay",  "-c", type=float, default=DEFAULT_DECAY,
                        help=f"Echo decay factor for synthesis (default: {DEFAULT_DECAY})")
    parser.add_argument("--no-play", action="store_true",
                        help="Skip audio playback")
    parser.add_argument("--save",   "-s", action="store_true",
                        help="Save filtered output to echokiller_output.wav")
    args = parser.parse_args()

    print_banner(args)

    # ── Load or generate audio ────────────────────────────────────────────────

    if args.input:
        print(f"  Loading '{args.input}'...")
        clean, sr = load_audio(args.input)
        # Trim to max 10 seconds for performance
        max_samples = 10 * sr
        if len(clean) > max_samples:
            print(f"  Trimming to 10 seconds ({max_samples} samples)")
            clean = clean[:max_samples]
        echo = add_echo(clean, sr, args.delay, args.decay)
        print(f"  Loaded: {len(clean)/sr:.2f}s at {sr} Hz")
        print(f"  Echo added: delay={args.delay*1000:.0f}ms, decay={args.decay:.2f}")
    else:
        print("  Generating synthetic speech-like signal...")
        sr    = SAMPLE_RATE
        clean = generate_speech_like(duration=4.0, sample_rate=sr)
        echo  = add_echo(clean, sr, args.delay, args.decay)
        print(f"  Generated: {len(clean)/sr:.2f}s at {sr} Hz")

    print(f"\n  Signal: {len(clean)} samples  |  Order: {args.order} taps")

    # ── Design Wiener filter ──────────────────────────────────────────────────

    print("\n  [1/3] Designing Wiener FIR filter (closed-form)...")
    wiener_coeffs = design_wiener_fir(echo, clean, args.order)
    print(f"  Filter designed: {len(wiener_coeffs)} taps")
    print(f"  Coefficient range: [{wiener_coeffs.min():.4f}, {wiener_coeffs.max():.4f}]")

    # ── Apply LMS adaptive filter ─────────────────────────────────────────────

    print("\n  [2/3] Running LMS adaptive filter...")
    filtered, lms_coeffs, errors = lms_adaptive_filter(
        echo, clean, args.order, LMS_MU, LMS_ITERATIONS
    )
    final_mse = float(np.mean(errors[-1000:])) if len(errors) >= 1000 else float(np.mean(errors))
    print(f"  LMS converged: final MSE = {final_mse:.6f}")

    # ── Compute metrics ───────────────────────────────────────────────────────

    snr_before = _snr(clean[:len(echo)],     echo)
    snr_after  = _snr(clean[:len(filtered)], filtered)
    echo_power = 10 * np.log10(np.mean(echo ** 2)     + 1e-12)
    filt_power = 10 * np.log10(np.mean(filtered ** 2) + 1e-12)

    print(f"\n  ── Results ──────────────────────────────────────────")
    print(f"  SNR before filtering : {snr_before:.1f} dB")
    print(f"  SNR after filtering  : {snr_after:.1f} dB  (+{snr_after - snr_before:.1f} dB)")
    print(f"  Echo power reduction : {echo_power:.1f} → {filt_power:.1f} dBFS")
    print(f"  Echo attenuation     : {echo_power - filt_power:.1f} dB")

    # ── Playback ──────────────────────────────────────────────────────────────

    if not args.no_play and HAS_SOUNDDEVICE:
        print(f"\n  [Playback] You will hear: echo → filtered → original")
        input("  Press ENTER to play echo-corrupted version...")
        play_audio(echo, sr, "echo-corrupted")
        input("  Press ENTER to play filtered (echo-cancelled) version...")
        play_audio(filtered, sr, "filtered")
        input("  Press ENTER to play clean reference...")
        play_audio(clean, sr, "clean reference")
    elif not HAS_SOUNDDEVICE:
        print("\n  (Install sounddevice for audio playback: pip install sounddevice)")

    # ── Save output ───────────────────────────────────────────────────────────

    if args.save:
        out_path = "echokiller_output.wav"
        print(f"\n  Saving filtered output to {out_path}...")
        save_audio(out_path, filtered, sr)

    # ── Plot ──────────────────────────────────────────────────────────────────

    print("\n  [3/3] Rendering plots...")
    plot_results(clean, echo, filtered, lms_coeffs, errors, sr)


if __name__ == "__main__":
    main()