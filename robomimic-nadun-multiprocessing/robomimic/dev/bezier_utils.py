import torch


def compute_bezier_points_batch(control_points, tension=0.5):
    """
    Compute control points for piecewise cubic Bézier curve with batch support and arbitrary dimensions.

    Args:
        control_points (torch.Tensor): Tensor of shape (batch_size, n, d), where d is the dimension of the points.
        tension (float): Determines the "tightness" of the curve (0-1).

    Returns:
        torch.Tensor: Tensor of shape (batch_size, n-1, 4, d), representing the Bézier control points for each segment.
    """
    bezier_segments = []
    batch_size, n, d = control_points.shape
    for i in range(n - 1):
        # Start and end points of the segment
        P0 = control_points[:, i]  # Shape: (batch_size, d)
        P3 = control_points[:, i + 1]  # Shape: (batch_size, d)

        # Compute adjacent points for tangent calculation
        if i == 0:
            P_prev = P0
        else:
            P_prev = control_points[:, i - 1]

        if i == n - 2:
            P_next = P3
        else:
            P_next = control_points[:, i + 2]

        # Compute tangents
        T1 = tension * (P3 - P_prev)  # Shape: (batch_size, d)
        T2 = tension * (P_next - P0)  # Shape: (batch_size, d)

        # Compute control points
        C1 = P0 + T1 / 3  # Shape: (batch_size, d)
        C2 = P3 - T2 / 3  # Shape: (batch_size, d)

        # Combine into Bézier segment
        bezier_segment = torch.stack((P0, C1, C2, P3), dim=1)  # Shape: (batch_size, 4, d)
        bezier_segments.append(bezier_segment)

    return torch.stack(bezier_segments, dim=1)  # Shape: (batch_size, n-1, 4, d)


def bezier_curve_batch(bezier_points, num_samples=50):
    """
    Evaluate cubic Bézier curve segments at uniformly spaced intervals in batch with arbitrary dimensions.

    Args:
        bezier_points (torch.Tensor): Tensor of shape (batch_size, n-1, 4, d), where d is the dimension of the points.
        num_samples (int): Number of samples to generate per segment.

    Returns:
        torch.Tensor: Sampled points of shape (batch_size, (n-1)*num_samples, d).
    """
    device = bezier_points.device
    batch_size, n_segments, _, d = bezier_points.shape
    t = torch.linspace(0, 1, num_samples, device=device).view(1, 1, -1)  # Shape: (1, 1, num_samples)

    # Compute basis functions
    B0 = (1 - t) ** 3  # Shape: (1, 1, num_samples)
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3

    basis = torch.stack((B0, B1, B2, B3), dim=-1)  # Shape: (1, 1, num_samples, 4)

    # Evaluate Bézier curve
    bezier_points = bezier_points.unsqueeze(2)  # Shape: (batch_size, n-1, 1, 4, d)
    curve = (basis.unsqueeze(-1) * bezier_points).sum(dim=-2)  # Shape: (batch_size, n-1, num_samples, d)

    return curve.view(batch_size, -1, d)  # Shape: (batch_size, (n-1)*num_samples, d)


def generate_smooth_curve_batch(control_points, tension=0.5, num_samples=50):
    """
    Generate smooth curves using piecewise cubic Bézier segments in batch with arbitrary dimensions.

    Args:
        control_points (torch.Tensor): Tensor of shape (batch_size, n, d), where d is the dimension of the points.
        tension (float): Determines the "tightness" of the curve (0-1).
        num_samples (int): Number of samples per segment.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, (n-1)*num_samples, d).
    """
    bezier_segments = compute_bezier_points_batch(control_points, tension)
    smooth_curve = bezier_curve_batch(bezier_segments, num_samples)
    return smooth_curve


def random_sample_from_bezier(B=2, N=16, D=7, S=8, T=64, device='cpu'):
    """
    Sample "noise" from smoothed gaussian instead of raw gaussian
    Args:
        B:
        N:
        D:
        T:
        device:

    Returns:

    """

    # TODO maybe subsample noise from a larger noise vector instead of creating directly


    # tc = torch.linspace(0, 1, N)
    # tc_n = torch.linspace(0, 1, T + 1)
    # noise = torch.randn(T + 1, 1)
    # control_points = torch.cat([tc.unsqueeze(1), noise[::4, :]], 1).unsqueeze(0).repeat(2, 1, 1)

    # Create control points from noise
    control_points = torch.randn((B, S + 1, D), device=device)
    # Sample control points from bezier curve
    smooth_control_points = generate_smooth_curve_batch(control_points, tension=0.5, num_samples=N//S)

    return smooth_control_points