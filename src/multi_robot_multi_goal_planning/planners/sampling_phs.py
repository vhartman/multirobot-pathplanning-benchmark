import numpy as np

# # taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# # needed adaption to work.
# def sample_unit_ball(dim) -> np.array:
#     """Samples a point uniformly from the unit ball. This is used to sample points from the Prolate HyperSpheroid (PHS).

#     Returns:
#         Sampled Point (np.array): The sampled point from the unit ball.
#     """
#     # u = np.random.uniform(-1, 1, dim)
#     # norm = np.linalg.norm(u)
#     # r = np.random.random() ** (1.0 / dim)
#     # return r * u / norm
#     u = np.random.normal(0, 1, dim)
#     norm = np.linalg.norm(u)
#     # Generate radius with correct distribution
#     r = np.random.random() ** (1.0 / dim)
#     return (r / norm) * u


# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.
def sample_unit_ball(dim, n=1) -> np.ndarray:
    """Samples n points uniformly from the unit ball. This is used to sample points from the Prolate HyperSpheroid (PHS).

    Args:
        dim (int): Dimension of the unit ball.
        n (int): Number of points to sample. Default is 1.

    Returns:
        np.ndarray: An array of shape (n, dim) containing the sampled points.
    """
    u = np.random.normal(0, 1, (dim, n))
    norms = np.linalg.norm(u, axis=0, keepdims=True)
    # Generate radii with correct distribution
    r = np.random.random(n) ** (1.0 / dim)
    return (r[None, :] / norms) * u


def compute_PHS_matrices(a, b, c):
    dim = len(a)
    diff = b - a

    # Calculate the center of the PHS.
    center = (a + b) / 2
    # The transverse axis in the world frame.
    c_min = np.linalg.norm(diff)

    # The first column of the identity matrix.
    # one_1 = np.eye(a1.shape[0])[:, 0]
    a1 = diff / c_min
    e1 = np.zeros(dim)
    e1[0] = 1.0

    # Optimized rotation matrix calculation
    U, S, Vt = np.linalg.svd(np.outer(a1, e1))
    # Sigma = np.diag(S)
    # lam = np.eye(Sigma.shape[0])
    lam = np.eye(dim)
    lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
    # Calculate the rotation matrix.
    # cwe = np.matmul(U, np.matmul(lam, Vt))
    cwe = U @ lam @ Vt
    # Get the radius of the first axis of the PHS.
    # r1 = c / 2
    # Get the radius of the other axes of the PHS.
    # rn = [np.sqrt(c**2 - c_min**2) / 2] * (dim - 1)
    # Create a vector of the radii of the PHS.
    # r = np.diag(np.array([r1] + rn))
    r = np.diag([c * 0.5] + [np.sqrt(c**2 - c_min**2) * 0.5] * (dim - 1))

    return cwe @ r, center


def sample_phs_with_given_matrices(rot, center, n=1):
    dim = len(center)
    x_ball = sample_unit_ball(dim, n)
    # Transform the point from the unit ball to the PHS.
    # op = np.matmul(np.matmul(cwe, r), x_ball) + center
    return rot @ x_ball + center[:, None]
