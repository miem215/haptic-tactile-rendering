import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def plot_tactile_signals(logs, surface_wall_x=0.3, target_freq=80):
    """
    Renders the 3-panel Cutaneous & Kinesthetic Signal Dashboard.
    Uses Ensemble Averaging across individual passes to eliminate boundary phase jumps.
    """
    # ---------------------------------------------------------
    # 1. Unpack logs from simulation
    # ---------------------------------------------------------
    time = np.array(logs['time'])
    pos_x = np.array(logs['pos_x'])
    pos_y = np.array(logs['pos_y'])
    vel_y = np.array(logs['vel_y'])
    Fx = np.array(logs['force_x'])
    Fy = np.array(logs['force_y'])

    # ---------------------------------------------------------
    # 2. Extract Individual Forward Passes (v_y > 0)
    # ---------------------------------------------------------
    is_fwd = (vel_y > 0.0)
    padded = np.pad(is_fwd.astype(int), (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # Target exact physical aperture [-0.025m, +0.025m] (L = 0.050m)
    y_start, y_end = -0.025, 0.025
    L_pass = y_end - y_start
    pts_per_pass = 1000
    y_grid = np.linspace(y_start, y_end, pts_per_pass)
    spatial_fs = pts_per_pass / L_pass  # 20,000 samples/m

    n_fft = 4096
    pass_spectra = []

    for s_idx, e_idx in zip(starts, ends):
        if (e_idx - s_idx) > 10:
            y_pass = pos_y[s_idx:e_idx]
            Fy_pass = Fy[s_idx:e_idx]

            # Ensure pass spans the physical window
            if np.min(y_pass) <= y_start and np.max(y_pass) >= y_end:
                mask = (y_pass >= y_start) & (y_pass <= y_end)
                y_sub = y_pass[mask]
                Fy_sub = Fy_pass[mask]

                # Resample onto uniform grid
                Fy_interp = np.interp(y_grid, y_sub, Fy_sub)

                # High-pass filter (> 20 cycles/m) to cut friction baseline
                Fy_detrend = Fy_interp - np.mean(Fy_interp)
                b, a = signal.butter(4, 20.0 / (0.5 * spatial_fs), btype='high')
                Fy_filtered = signal.filtfilt(b, a, Fy_detrend)

                # Window function
                window = np.hanning(len(Fy_filtered))
                Fy_windowed = Fy_filtered * window

                # Pass FFT
                fft_mag = np.abs(np.fft.rfft(Fy_windowed, n=n_fft))
                pass_spectra.append(fft_mag)

    # ---------------------------------------------------------
    # 3. Ensemble Average
    # ---------------------------------------------------------
    if len(pass_spectra) > 0:
        mean_spectrum = np.mean(pass_spectra, axis=0)
        spatial_freqs = np.fft.rfftfreq(n_fft, d=1.0/spatial_fs)
        fft_mag_scaled = (2.0 / pts_per_pass) * mean_spectrum
    else:
        spatial_freqs = np.linspace(0, 200, 100)
        fft_mag_scaled = np.zeros_like(spatial_freqs)

    # ---------------------------------------------------------
    # 4. Dashboard Rendering
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=False, constrained_layout=True)
    fig.suptitle("Cutaneous & Kinesthetic Signal Analysis Dashboard", fontsize=14, fontweight='bold')

    # --- Panel 1: Trajectory ---
    axs[0].set_title("End-Effector Trajectory", fontsize=11, fontweight='semibold')
    axs[0].plot(time, pos_x, label="Pos X (Depth)", color="tab:blue")
    axs[0].plot(time, pos_y, label="Pos Y (Sliding)", color="tab:orange")
    axs[0].axhline(y=surface_wall_x, color="tab:red", linestyle="--", label=f"Surface Wall ({surface_wall_x}m)")
    axs[0].set_ylabel("Position (m)")
    axs[0].set_xlim([time[0], time[-1]])
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    # --- Panel 2: Force Signals ---
    axs[1].set_title("Force Signals (Cutaneous Friction Modulation)", fontsize=11, fontweight='semibold')
    axs[1].plot(time, Fx, label="Fx: Normal Force (Stiffness)", color="tab:blue", alpha=0.8)
    axs[1].plot(time, Fy, label="Fy: Tangential Force (Texture + Friction)", color="tab:green", alpha=0.8)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_xlim([time[0], time[-1]])
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # --- Panel 3: Ensemble Averaged Spatial Spectrum ---
    axs[2].set_title("Spatial Domain Spectrum (Pass-Ensemble Averaged Recovery)", fontsize=11, fontweight='semibold')
    axs[2].plot(spatial_freqs, fft_mag_scaled, color="tab:red", label="Ensemble Power Spectrum")
    axs[2].axvline(x=target_freq, color="black", linestyle=":", linewidth=2, label=f"True Texture Frequency ({target_freq} cycles/m)")
    axs[2].set_xlabel("Spatial Frequency (cycles / meter)")
    axs[2].set_ylabel("Magnitude")
    axs[2].set_xlim([0, 200])
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right")

    plt.show()