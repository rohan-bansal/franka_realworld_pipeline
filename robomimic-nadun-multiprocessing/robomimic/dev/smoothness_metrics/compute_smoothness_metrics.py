import numpy as np
import math
import torch


def compute_sparc(positions, dt):
    """
    Compute SPARC (Spectral Arc Length) for a given trajectory.
    This implementation:
      - Extracts 3D end-effector positions from the trajectory.
      - Computes speed (magnitude of velocity).
      - Computes the real FFT of the speed.
      - Calculates the arc length in frequency domain.
      - Returns SPARC = -log(arc_length + epsilon).

    Args:
        positions: obs robot0_eef_pos from a rollout/demo
        dt:
    """

    if len(positions) < 2:
        return 0.0

    # Compute velocity and speed
    vel = (positions[1:] - positions[:-1]) / dt  # shape (N-1, 3)
    speed = np.linalg.norm(vel, axis=1)          # shape (N-1,)

    # If there's not enough data, return 0.0
    if len(speed) < 2:
        return 0.0

    # Compute FFT of the speed
    # rfft gives us the positive frequencies only
    speed_fft = np.fft.rfft(speed)
    freqs = np.fft.rfftfreq(len(speed), d=dt)

    # Compute arc length in the frequency domain
    # arc_length = sum over i of sqrt((Δfreq)^2 + (Δ|FFT|)^2)
    arc_length = 0.0
    for i in range(len(freqs) - 1):
        df = freqs[i+1] - freqs[i]
        dmag = np.abs(speed_fft[i+1]) - np.abs(speed_fft[i])
        arc_length += np.sqrt(df**2 + dmag**2)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    sparc = -math.log(arc_length + epsilon)

    return sparc

def compute_sparc_torch(positions, dt):
    """
    Compute SPARC (Spectral Arc Length) for a given trajectory using PyTorch.
    """
    positions = torch.tensor(positions, requires_grad=True, dtype=torch.float32)
    dt = torch.tensor(dt, dtype=torch.float32)

    if positions.shape[0] < 2:
        return torch.tensor(0.0)

    # Compute velocity and speed
    vel = (positions[1:] - positions[:-1]) / dt  # shape (N-1, 3)
    speed = torch.linalg.norm(vel, dim=1)       # shape (N-1,)

    if speed.shape[0] < 2:
        return torch.tensor(0.0)

    # Compute FFT of the speed
    speed_fft = torch.fft.rfft(speed)           # Real FFT of speed
    freqs = torch.fft.rfftfreq(len(speed), d=dt)
    freqs = freqs.to(device=speed_fft.device)

    # Compute arc length in the frequency domain
    df = freqs[1:] - freqs[:-1]                 # Frequency differences
    dmag = torch.abs(speed_fft[1:]) - torch.abs(speed_fft[:-1])  # Magnitude differences
    arc_length = torch.sum(torch.sqrt(df**2 + dmag**2))

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    sparc = -torch.log(arc_length + epsilon)

    return sparc, positions


def compute_LDLJ(positions=None, vel=None, print_results=False):
    assert not (positions is None and vel is None), "Pass in either positions or velocities"

    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    # 1. Compute velocity
    if vel is None:
        vel = np.diff(positions, axis=0) / dt

    # 2. Compute speed
    speed = np.linalg.norm(vel, axis=1)  # shape (N-1,)

    # 3. Compute first derivative of speed (dv/dt)
    dvdt = (speed[1:] - speed[:-1]) / dt  # shape (N-2,)

    # 4. Compute second derivative of speed (d2v/dt2)
    d2vdt2 = (dvdt[1:] - dvdt[:-1]) / dt  # shape (N-3,)

    # Now we have d2v/dt2 defined on a shorter time interval.
    # Let's define our effective start and end times:
    # Original time array for position: t = 0, dt, 2*dt, ..., (N-1)*dt
    N = vel.shape[0] + 1
    t1 = 0.0
    t2 = (N - 1) * dt

    # 5. Find v_peak
    v_peak = np.max(speed) if len(speed) > 0 else 0.0
    if v_peak == 0:
        raise ValueError("Peak speed is zero, cannot compute dimensionless jerk.")

    # 6. Compute the integral of |d2v/dt2|^2 from t1 to t2
    # Align indices carefully. d2vdt2 corresponds to times:
    # For positions: indices: 0 to N-1
    # For speed: indices: 0 to N-2
    # for dvdt: indices: 0 to N-3
    # for d2vdt2: indices: 0 to N-4
    #
    # We can approximate the integral as sum(|d2v/dt2|^2)*dt.
    integrand = np.abs(d2vdt2) ** 2
    integral = np.sum(integrand) * dt

    # 7. Compute DLJ
    movement_time = t2 - t1
    DLJ = - (movement_time ** 5 / (v_peak ** 2)) * integral

    # 8. Compute LDLJ
    LDLJ = -math.log(abs(DLJ))

    if print_results:
        print("DLJ:", DLJ)
        print("LDLJ:", LDLJ)

    return LDLJ