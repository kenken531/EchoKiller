# EchoKiller 🔇

EchoKiller is an **acoustic echo cancellation pipeline**: it loads an audio file (or generates a synthetic echo), applies an adaptive FIR Wiener filter to attenuate the echo, and displays before/after waveforms side-by-side with a visualization of the filter coefficients (impulse response) and the LMS convergence curve. It's built for the **BUILDCORED ORCAS — Day 16** challenge.

## How it works

- Loads a WAV file or generates a **synthetic speech-like signal** with multiple echoes at a configurable delay and decay.
- Designs a **Wiener FIR filter** by solving the Wiener-Hopf equations: builds the autocorrelation matrix of the echo signal and the cross-correlation with the clean reference, then solves `R·h = p` for the optimal filter weights.
- Runs an **LMS (Least Mean Squares) adaptive filter** for real-time-style convergence, making multiple passes over the signal and updating weights on each sample.
- Displays a **6-panel figure**: original signal, echo-corrupted signal, filtered output, filter impulse response (bar chart of tap weights), LMS convergence curve (MSE over time), and the filter's frequency response.
- Computes and prints **SNR improvement** and **echo attenuation in dB** before and after filtering.
- Optionally plays back the echo and filtered versions through your speakers so you can hear the difference.

## Requirements

- Python 3.10.x
- A WAV audio file (or use the built-in synthetic generator)

## Python packages:

```bash
pip install scipy numpy soundfile matplotlib sounddevice
```

(`sounddevice` is optional — only needed for audio playback)

## Setup

1. Install the required Python packages (see above or run:
```
pip install -r requirements.txt
```
after downloading `requirements.txt`)
2. Optionally prepare a WAV file with noticeable echo or reverb (a recording in a bathroom or stairwell works well).

## Usage

```bash
python echokiller.py                          # generate synthetic echo + filter it
python echokiller.py --input myfile.wav       # use your own WAV file
python echokiller.py --order 32               # fewer taps = faster, less aggressive
python echokiller.py --order 128              # more taps = slower, more thorough
python echokiller.py --delay 0.2 --decay 0.6  # tune synthetic echo parameters
python echokiller.py --no-play                # skip audio playback
python echokiller.py --save                   # save filtered output as WAV
```

The 6-panel figure shows:

| Panel | What it shows |
|---|---|
| ① Original | The clean reference signal |
| ② Echo-corrupted | The signal with echo added — the input to the filter |
| ③ Filtered output | The echo-cancelled result |
| ④ Impulse response | FIR tap weights — positive taps pass signal, negative taps cancel echo |
| ⑤ Convergence | LMS mean-squared error over time — should decrease as filter adapts |
| ⑥ Frequency response | How the filter affects each frequency band |

## Common fixes

**soundfile can't read .mp3** — convert to WAV first:
```bash
ffmpeg -i input.mp3 output.wav
```
Or use any online MP3-to-WAV converter.

**Filter makes audio worse** — reduce the filter order: `--order 32`. A very high order with a short signal can overfit.

**No visible difference in waveforms** — both signals are normalised to the same peak. The difference is in the fine structure — zoom in on the waveform or listen to the playback to hear the echo reduction.

**Audio playback fails** — install sounddevice: `pip install sounddevice`. If it still fails, use `--no-play` and compare by saving: `--save`.

**LMS convergence is slow** — the step size `LMS_MU` controls convergence speed. Increase it carefully (e.g. `0.005`) for faster adaptation, but values above `0.01` may cause instability.

## Hardware concept

EchoKiller implements **Acoustic Echo Cancellation (AEC)** — the exact algorithm running inside every speakerphone DSP chip, hearing aid, smart speaker, and video conferencing system. The Wiener filter solves the same linear algebra that Cirrus Logic, Texas Instruments, and Qualcomm implement in silicon. The LMS algorithm is the real-time version: instead of solving the whole system at once, it updates one sample at a time — exactly what an embedded DSP running at audio sample rate must do. The filter coefficients visualized as a bar chart are the **impulse response** of the learned echo path: the shape you'd see on an oscilloscope if you connected a hardware AEC chip's coefficient readout pins.

## Credits

- Signal processing: [SciPy](https://scipy.org/) + [NumPy](https://numpy.org/)
- Audio I/O: [soundfile](https://python-soundfile.readthedocs.io/)
- Visualization: [Matplotlib](https://matplotlib.org/)
- Audio playback: [sounddevice](https://python-sounddevice.readthedocs.io/)

Built as part of the **BUILDCORED ORCAS — Day 16: EchoKiller** challenge.
