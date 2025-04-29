import numpy as np
from scipy.linalg import svd, rq


def normalize_points(points):
    """
    Normalize 2D or 3D points to have zero mean and unit variance.

    Parameters:
    -----------
    points : ndarray of shape (N, D)
        Points to normalize, where D is 2 (for 2D points) or 3 (for 3D points).

    Returns:
    --------
    points_norm : ndarray of shape (N, D)
        Normalized points.
    T : ndarray of shape (D+1, D+1)
        Transformation matrix for normalization.
    """
    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)
    dim = points.shape[1]

    # Construct normalization matrix
    T = np.eye(dim + 1)
    T[:dim, :dim] /= std_dev
    T[:dim, -1] = -mean / std_dev

    # Apply normalization
    points_homog = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    points_norm = (T @ points_homog.T).T[:, :dim]  # Normalize and return to non-homogeneous

    return points_norm, T


def dlt(points_3d, points_2d):
    """
    Perform Direct Linear Transformation (DLT) with point normalization.

    Parameters:
    -----------
    points_3d : ndarray of shape (N, 3)
        Array of 3D world points with coordinates (X, Y, Z).
    points_2d : ndarray of shape (N, 2)
        Array of corresponding 2D image points with coordinates (x, y).

    Returns:
    --------
    P : ndarray of shape (3, 4)
        Camera projection matrix mapping 3D points to 2D points.
    """
    # Validate input dimensions
    assert points_3d.shape[0] == points_2d.shape[0], "Number of 3D and 2D points must match."
    assert points_3d.shape[1] == 3, "3D points must have shape (N, 3)."
    assert points_2d.shape[1] == 2, "2D points must have shape (N, 2)."

    # Normalize the 3D and 2D points
    points_3d_norm, T_3d = normalize_points(points_3d)
    points_2d_norm, T_2d = normalize_points(points_2d)

    # Construct the design matrix
    num_points = points_3d.shape[0]
    A = []

    for i in range(num_points):
        X, Y, Z = points_3d_norm[i]
        x, y = points_2d_norm[i]

        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

    A = np.array(A)

    # Perform Singular Value Decomposition (SVD) on A
    U, S, Vt = svd(A)
    V = Vt.T
    P_norm = V[:, -1].reshape(3, 4)

    # Denormalize the projection matrix
    P = np.linalg.inv(T_2d) @ P_norm @ T_3d

    # Normalize P such that P[2, 3] = 1
    P /= P[2, 3]

    return P


def decompose_projection_matrix(P):
    """
    Decomposes the projection matrix P into intrinsic matrix K, rotation matrix R, and translation vector t.

    Parameters:
    -----------
    P : ndarray of shape (3, 4)
        Camera projection matrix.

    Returns:
    --------
    K : ndarray of shape (3, 3)
        Intrinsic matrix (upper triangular).
    R : ndarray of shape (3, 3)
        Rotation matrix (orthogonal).
    t : ndarray of shape (3,)
        Translation vector.
    """
    # Extract the 3x3 matrix M (first 3 columns of P) and last column p4
    M = P[:, :3]
    p4 = P[:, 3]

    # Perform RQ decomposition of M to get K and R
    K, R = rq(M)

    # Ensure K has positive diagonal elements
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Normalize K such that K[2, 2] = 1
    K /= K[2, 2]

    # Compute the translation vector t
    # t = np.linalg.inv(K) @ p4
    t = np.linalg.solve(K, p4)

    return K, R, t


def dlt_calib(points_3d, points_2d):
    """First apply dlt(), then decompose_projection_matrix().

    Parameters:
    -----------
    points_3d : ndarray of shape (N, 3)
        Array of 3D world points with coordinates (X, Y, Z).
    points_2d : ndarray of shape (N, 2)
        Array of corresponding 2D image points with coordinates (x, y).

    Returns:
    --------
    Mint : ndarray of shape (3, 3)
        Intrinsic matrix.
    Mext : ndarray of shape (3, 4)
        Extrinsic matrix.
    """
    P = dlt(points_3d, points_2d)
    K, R, t = decompose_projection_matrix(P)
    Mint = K
    Mext = np.hstack((R, t.reshape(3, 1)))
    return Mint, Mext


# Example usage
if __name__ == "__main__":
    # Example 3D world points (N=6)
    points_3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])

    # Corresponding 2D image points
    points_2d = np.array([
        [100, 200],
        [200, 200],
        [200, 300],
        [100, 300],
        [120, 220],
        [220, 320]
    ])

    # Compute the projection matrix
    projection_matrix = dlt(points_3d, points_2d)
    print("Projection Matrix:")
    print(projection_matrix)

    # Decompose the projection matrix
    intrinsic_matrix, extrinsic_matrix = decompose_projection_matrix(projection_matrix)
    print("\nIntrinsic Matrix:")
    print(intrinsic_matrix)
    print("\nExtrinsic Matrix:")
    print(extrinsic_matrix)