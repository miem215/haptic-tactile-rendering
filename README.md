# Tactile Haptics Signal Processing & Analysis

A MuJoCo-based haptic simulation framework and signal processing pipeline for rendering and recovering spatial micro-textures from end-effector tactile force signals.This project isolates position-based surface micro-textures ($80\text{ cycles/meter}$) from non-linear friction and macro-sliding dynamics using Pass-Ensemble Spectral Averaging.

## Research Question
> **How can velocity-invariant spatial micro-texture attributes ($80\text{ cycles/m}$) be deterministically rendered in simulation and cleanly isolated from non-linear contact friction and directional boundary phase jumps?**

##  Multimodal Tactile Signal Decomposition

Human tactile perception relies on two distinct sensory pathways: **kinesthetic perception** (sensing bulk forces, stiffness, and limb position via mechanoreceptors in muscles, tendons, and joints) and **cutaneous perception** (sensing fine skin deformation, friction drag, and micro-vibrations via tactile corpuscles in the dermal layers).

This framework decouples and models both sensory channels within a unified real-time control loop.

---

### 1. Mathematical Formulation

#### A. Kinesthetic Normal Force ($F_x$)
Simulates bulk surface compliance and structural boundary penetration:

$$F_x = -K_{\text{wall}} \cdot (x - x_{\text{wall}}) - B_{\text{wall}} \cdot v_x$$

* **Physical Role:** Provides rigid wall resistance during deep contact pressing.
* **Biological Target:** **Slow-Adapting Type I (SA-I / Merkel)** mechanoreceptors, which respond to static pressure and gross structural contours ($0\text{--}5\text{ Hz}$).

#### B. Cutaneous Tangential Force ($F_y$)
Combines non-linear velocity-dependent friction with velocity-invariant spatial micro-textures:

$$F_y = \underbrace{-\mu \vert{}F_x\vert{} \tanh(10 v_y)}_{\text{Friction Drag Profile}} + \underbrace{A \cdot \sin(2\pi \cdot f_{\text{spatial}} \cdot y)}_{\text{Spatial Micro-Texture}}$$

* **Physical Role:** Generates tactile surface drag along with high-frequency spatial vibrations ($80\text{ cycles/m}$) as the end-effector glides along the $Y$-axis.
* **Biological Target:** **Fast-Adapting Type II (FA-II / Pacinian)** mechanoreceptors, specialized in detecting high-frequency skin vibrations and micro-textures ($50\text{--}400\text{ Hz}$).

### 2. Physical Hardware & Mechatronic Mapping

To transition this signal model from simulation to a physical haptic interface, the decoupled kinesthetic and cutaneous forces map to distinct physical actuators on a single stylus or knob:

```text
┌─────────────────────────────────────────────────────────────┐
│                       USER'S HAND                           │
└──────────────────────────────┬──────────────────────────────┘
                               │
             ┌─────────────────┴──────────────────┐
             │         Physical Stylus / Knob     │
             └─────────┬─────────────────┬────────┘
                       │                 │
      Kinesthetic Force│                 │Cutaneous Vibration
                       ▼                 ▼
 ┌───────────────────────────┐     ┌───────────────────────────┐
 │ Kinesthetic Actuator      │     │ Cutaneous Transducer      │
 │ (DC / BLDC / Linear Motor)│     │ (Voice Coil / LRA / Piezo)│
 └─────────────┬─────────────┘     └─────────────┬─────────────┘
               │                                 │
               │  Position (x, y)                │ Voltage Command
               ▼                                 │
 ┌───────────────────────────────────────────────┴─────────────┐
 │ 1 kHz Real-Time Microcontroller (Teensy / STM32 / RT-Linux) │
 │ Enforces: Fx = -K*Δx  |  Fy = -μ*Fx*tanh(v) + A*sin(2π*80*y) │
 └─────────────────────────────────────────────────────────────┘
```

## Spatial Signal Processing Pipeline

Standard time-domain Fourier transforms fail during manual or dynamic robotic tactile exploration because sliding velocity changes continuously. This causes temporal frequency smearing ($f_{\text{temporal}} = f_{\text{spatial}} \cdot \vert{}v_y\vert{}$). 

This pipeline transforms time-series force logs into a velocity-invariant spatial domain, using **Exact-Aperture Resampling** and **Pass-Ensemble Averaging** to recover true surface spatial frequencies.

---

### 1. DSP Pipeline Architecture

```text
 ┌───────────────────────────────────────────────────────────────┐
 │               Raw Simulation Force & Kinematic Logs           │
 │               [ time, pos_x, pos_y, vel_y, Fx, Fy ]           │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 1. Motion Segmentation & Direction Filtering                  │
 │    • Isolate forward passes (v_y > 0 m/s)                     │
 │    • Reject direction reversals to eliminate phase modulation │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 2. Exact Physical Aperture Resampling                         │
 │    • Slice pass to exact window: y ∈ [-0.025m, +0.025m]       │
 │    • Aperture L = 0.050m (exactly 4.0 wave cycles at 80 c/m)   │
 │    • Interpolate onto uniform grid (N = 1000, fs = 20 kHz/m)   │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 3. Detrending & Butterworth High-Pass Filtering               │
 │    • Subtract mean friction DC offset                         │
 │    • 4th-order zero-phase Butterworth filter (f_cutoff = 20 c/m)│
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 4. Windowing & Zero-Padded Spatial FFT                        │
 │    • Apply Hann window: w[n] = 0.5 * (1 - cos(2πn / N))        │
 │    • Zero-padded FFT (n_fft = 4096) for high spectral density  │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 5. Pass-Ensemble Spectral Averaging                           │
 │    • Compute magnitude spectrum per individual forward pass    │
 │    • Average spectra across passes: S_avg(f) = (1/M) Σ |X_m(f)|│
 └───────────────────────────────────────────────────────────────┘
```

### 2. Theoretical Solutions to Key DSP Challenges

#### A. Velocity Decoupling & Exact-Aperture Resampling
To eliminate time-domain frequency smearing caused by non-constant sliding speeds, the tangential force $F_y(t)$ is re-indexed against physical probe position $y(t)$:

$$F_y(y) = \text{Interpolate}\Big(y_{\text{grid}}, \, y(t), \, F_y(t)\Big)$$

To eliminate spectral leakage without heavy window distortion, each pass is cropped to an **exact integer wavelength window**:

* **Physical Aperture Length ($L$):** $0.050\text{ m}$ ($5\text{ cm}$)
* **Target Wavelength ($\lambda$):** $\lambda = \frac{1}{80\text{ cycles/m}} = 0.0125\text{ m}$ ($1.25\text{ cm}$)
* **Integer Wavelength Condition:** 

$$\frac{L}{\lambda} = \frac{0.050\text{ m}}{0.0125\text{ m}} = 4.0 \text{ integer cycles}$$

Because the observation window contains exactly $4.0$ complete wave cycles, edge truncation discontinuities are structurally minimized prior to windowing.

---

#### B. Pass-Ensemble Averaging vs. Spatial Stitching
Gluing multiple forward passes end-to-end to create a synthetic continuous spatial path introduces micro-discontinuities at sweep boundaries. These phase jumps act as low-frequency spatial phase modulation, splitting the true spectrum into artificial sidebands ($73$ and $86\text{ cycles/m}$).

Instead, **Pass-Ensemble Averaging** processes each sweep independently and averages their magnitude spectra:

$$S_{\text{ensemble}}(f_{\text{spatial}}) = \frac{1}{M} \sum_{m=1}^{M} \left\vert{} \mathcal{F} \left\{ F_{y, m}(y) \cdot w(y) \right\} \right\vert{}$$

Where:
* $M$ is the total number of valid forward passes.
* $F_{y, m}(y)$ is the position-resampled force profile for pass $m$.
* $w(y)$ is the spatial Hann window function.

This cancels stochastic stick-slip contact noise and suppresses uncorrelated measurement variations while preserving an uncorrupted, sharp fundamental peak at $80.0\text{ cycles/m}$.

---

### 3. Spectral Peak Interpretation

```text
  Normalized Power
        ▲
        │                 ★ Fundamental Texture Peak (80 cycles/m)
   1.00 ┼                    ┌─┐
        │                    │ │
   0.60 ┼   Aperture / HPF   │ │               ★ 2nd Contact Harmonic (160 cycles/m)
        │     Bump (20 c/m)  │ │                  ┌─┐
   0.30 ┼        ┌─┐         │ │                  │ │
        │        │ │         │ │                  │ │
   0.00 ┴────────┴─┴─────────┴─┴──────────────────┴─┴───────────► Spatial Frequency
        0        20          80                   160      200    (cycles / meter)