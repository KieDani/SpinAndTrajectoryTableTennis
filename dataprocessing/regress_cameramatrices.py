'''
Find the extrinsic and instrinsic camera matrices given many measurements in image coordinates and their corresponding
3D coordinates.
'''
import random
import numpy as np
from scipy.optimize import minimize, least_squares
import math
from scipy.spatial.transform import Rotation as R

from dataprocessing.my_dlt import dlt_calib
from helper import table_points, WIDTH, HEIGHT
from helper import get_Mext, get_cameralocations, get_forwards, get_rights, get_ups
from helper import world2cam, cam2img

np.random.seed(42)
random.seed(42)


print(f'using resolution {HEIGHT}x{WIDTH}. Check if this is correct!')

points3d = {
    1: np.array(table_points[0], dtype=np.float64),
    2: np.array(table_points[1], dtype=np.float64),
    3: np.array(table_points[2], dtype=np.float64),
    4: np.array(table_points[3], dtype=np.float64),
    5: np.array(table_points[4], dtype=np.float64),
    6: np.array(table_points[5], dtype=np.float64),
    7: np.array(table_points[6], dtype=np.float64),
    8: np.array(table_points[7], dtype=np.float64),
    9: np.array(table_points[8], dtype=np.float64),
    10: np.array(table_points[9], dtype=np.float64),
    11: np.array(table_points[10], dtype=np.float64),
    12: np.array(table_points[11], dtype=np.float64),
    13: np.array(table_points[12], dtype=np.float64),
}


# regression to find the camera matrices
def regress_cameramatrices(points2d, points3d=points3d, startmatrices=None, use_lm=False, use_prints=True):
    '''
    points2d: [(key, point2d), (key, point2d), ...]
    points3d: {key: point3d, key: point3d, ...}
    startmatrices: (Mint, Mext)
    use_lm: Use Levenberg-Marquardt optimization if True.
    return: (Mint, Mext)
    '''

    # residuals for Levenberg-Marquardt optimization
    def residuals(x):
        px = WIDTH // 2
        py = HEIGHT // 2
        fx, fy, tx, ty, tz, a, b, c = x
        Mint = np.array([[fx, 0, px, 0],
                         [0, fy, py, 0],
                         [0, 0, 1, 0], ])

        rot = R.from_euler('xyz', [a, b, c], degrees=False).as_matrix()
        Mext = np.array([
            [rot[0, 0], rot[0, 1], rot[0, 2], tx],
            [rot[1, 0], rot[1, 1], rot[1, 2], ty],
            [rot[2, 0], rot[2, 1], rot[2, 2], tz],
            [0, 0, 0, 1]
        ])
        projected_cam = world2cam(points3d_lst, Mext)
        projected_img = cam2img(projected_cam, Mint)
        return np.sqrt(np.sum(np.square(projected_img - points2d_lst), axis=1))

    # function to be optimized
    def opt_func(x):
        return np.sum(residuals(x))

    points2d_lst, points3d_lst = [], []
    for key, point in points2d:
        points2d_lst.append(point)
        points3d_lst.append(points3d[key])
    points2d_lst = np.array(points2d_lst)
    points3d_lst = np.array(points3d_lst)

    assert startmatrices is not None, 'startmatrices must be provided'
    Mint, Mext = startmatrices
    angles = R.from_matrix(Mext[:3, :3]).as_euler('xyz', degrees=False)
    x0 = np.array([Mint[0, 0], Mint[1, 1], Mext[0, 3], Mext[1, 3], Mext[2, 3], angles[0], angles[1], angles[2]])
    # map angles to [-pi, pi]
    x0[5:] = np.mod((x0[5:] + np.pi), (2 * np.pi)) - np.pi

    if use_prints: print('initial error: ', opt_func(x0) / len(points2d_lst))

    if use_lm:
        # Levenberg-Marquardt optimization
        res = least_squares(residuals, x0, method='lm')
    else:
        # BFGS optimization
        res = minimize(opt_func, x0, method='BFGS')

    fx, fy, tx, ty, tz, a, b, c = res.x

    px = WIDTH // 2
    py = HEIGHT // 2

    Mint = np.array([[fx, 0, px, 0],
                     [0, fy, py, 0],
                     [0, 0, 1, 0], ])
    rot = R.from_euler('xyz', [a, b, c], degrees=False).as_matrix()
    Mext = np.array([
        [rot[0, 0], rot[0, 1], rot[0, 2], tx],
        [rot[1, 0], rot[1, 1], rot[1, 2], ty],
        [rot[2, 0], rot[2, 1], rot[2, 2], tz],
        [0, 0, 0, 1]
    ])

    if use_prints: print('final error: ', opt_func(res.x) / len(points2d_lst))

    return Mint, Mext


def regress_cameramatrices_ransac(points2d, startmatrices=None, use_prints=True, use_lm=False):
    '''
    RANSAC-based regression for camera matrices. We always ensure that not all points are in the plane

    points2d: [(key, point2d), (key, point2d), ...]
    startmatrices: (Mint, Mext)
    use_prints: Print intermediate results if True.
    return: (Mint, Mext)
    '''
    max_iterations = 100  # Maximum number of iterations
    num_points = 6 # Number of points used for ransac
    inlier_threshold = 3.0  # Distance threshold to count as an inlier
    best_inliers = None
    best_matrices = None

    # Ensure points 10 and 11 are always included
    fixed_keys = [10, 11]
    fixed_points2d = [(int(key), point) for key, point in points2d if key in fixed_keys]

    rnd = np.random.default_rng(seed=42)
    for iteration in range(max_iterations):
        # Randomly sample other points (excluding fixed points)
        sampled_points = rnd.choice([int(key) for key, _ in points2d if key not in fixed_keys], size=num_points-len(fixed_keys), replace=False)
        sampled_points = [int(s) for s in sampled_points]
        sampled_points2d = [(int(key), point) for key, point in points2d if key in sampled_points]

        # Combine fixed points and sampled points
        subset_points2d = [*fixed_points2d, *sampled_points2d]
        subset_points3d = {key: points3d[key] for key in fixed_keys + sampled_points}

        # Run camera regression on the subset
        Mint, Mext = regress_cameramatrices(subset_points2d, subset_points3d, startmatrices=startmatrices, use_prints=False, use_lm=use_lm)

        # Evaluate inliers
        inliers = list()
        for key, point in points2d:
            projected_cam = world2cam(points3d[key], Mext)
            projected_img = cam2img(projected_cam, Mint)
            error = np.linalg.norm(projected_img - point)
            if error < inlier_threshold:
                inliers.append((key, point))

        # Update the best model if we have more inliers
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_matrices = (Mint, Mext)

    if best_matrices is None:
        raise ValueError("RANSAC failed to find a valid model.")

    if use_prints:
        print(f"Best model found with {len(best_inliers)} inliers.")

    # Refine the model using all inliers
    Mint, Mext = best_matrices
    subset_points2d = best_inliers
    subset_points3d = {key: points3d[key] for key, _ in subset_points2d}
    Mint, Mext = regress_cameramatrices(subset_points2d, subset_points3d, startmatrices=(Mint, Mext), use_prints=use_prints, use_lm=use_lm)

    return Mint, Mext


def DLT(points2d, points3d):
    #Direct linear transformation
    points2d_dlt, points3d_dlt = [], []
    for key, point in points2d:
        points2d_dlt.append(point)
        points3d_dlt.append(points3d[key])
    points2d_dlt = np.array(points2d_dlt)
    points3d_dlt = np.array(points3d_dlt)
    # P, K, R, t, err = dlt_calib(3, points3d_dlt, points2d_dlt)
    # Mext = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    # Mint = np.concatenate((K, np.array([[0, 0, 0]]).T), axis=1)
    Mint, Mext = dlt_calib(points3d_dlt, points2d_dlt)
    return Mint, Mext


def calc_cameramatrices(keypoints_dict, use_prints, use_lm=False, use_ransac=False):
    '''First do a DLT to get a rough estimate of the camera matrices.
    Then use the regression to get a better estimate of the camera matrices.
    -----
    keypoints_dict: {1:[(x, y), (x, y)], 2:[(x, y), (x, y)], ...}
    '''
    #Direct linear transformation
    keys_dlt = keypoints_dict.keys()
    assert len(keys_dlt) >= 6, 'not enough points for DLT'
    # points2d_lst: [(key1, point1), (key2, point2), ...]
    points2d_lst = []
    for key, points in keypoints_dict.items():
        for point in points:
            points2d_lst.append((key, point))

    # initial guess using DLT and a subset of the points
    Mint, Mext = DLT(points2d_lst, points3d)

    # intrinsic matirx is calculated
    if use_ransac:
        Mint, Mext = regress_cameramatrices_ransac(points2d_lst, startmatrices=(Mint, Mext), use_prints=use_prints, use_lm=use_lm)
    else: # regression using all points
        Mint, Mext = regress_cameramatrices(points2d_lst, points3d, startmatrices=(Mint, Mext), use_prints=use_prints, use_lm=use_lm)

    # with np.printoptions(precision=1, suppress=True):
    #     print('estimated Mint: \n ', Mint)
    #     print('estimated Mext: \n', Mext)
    #     print('estimated camera matrix: \n', Mint @ Mext)
    # print('-' * 50)

    return Mint, Mext


def do_regression(keypoint_path, use_prints=True, use_lm=False, use_ransac=False):
    '''
    keypoint_path: path to csv containing the keypoints
    params_path: path to csv containing the intrinsic parameters; if None, the intrinsic parameters are regressed too,
                otherwise the intrinsic parameters are read from the csv
    '''
    import pandas as pd
    keypoint_path = keypoint_path
    keypoints_csv = pd.read_csv(keypoint_path, sep=';', skiprows=[0])

    keypoints_dict = {}
    for k in keypoints_csv.iterrows():
        for keypoint in k[1].keys():
            if 'flag' in keypoint and 'ball' not in keypoint and 2.01 > k[1][keypoint] > 1.99:
                key = int(keypoint.replace('_flag', ''))
                if key not in keypoints_dict:
                    keypoints_dict[key] = []
                coords = (k[1][keypoint.replace('flag', 'x')], k[1][keypoint.replace('flag', 'y')])
                keypoints_dict[key].append(coords)

    #params = None
    Mint, Mext = calc_cameramatrices(keypoints_dict, use_prints=use_prints, use_lm=use_lm, use_ransac=use_ransac)
    # print('intrinsics: \n', Mint)
    if use_prints:
        print(f'fx: {Mint[0, 0]}, fy: {Mint[1, 1]}')
        print('extrinsics: \n', Mext)
        print('-' * 50)
        # camera position
        R = Mext[:3, :3]
        t = Mext[:3, 3]
        camera_position = get_cameralocations(Mext)
        print('camera position: ', camera_position)
        forward = get_forwards(Mext)
        print('forward: ', forward)
        up = get_ups(Mext)
        print('up: ', up)
        right = get_rights(Mext)
        print('right: ', right)

    return Mint, Mext


def visualize_results(regress_intrinsics=False):
    import cv2
    import matplotlib.pyplot as plt
    video_path = '../hdready_1_leftVid.mp4'
    keypoint_path = video_path.replace('.mp4', '_keypoints.csv')
    params_path = video_path.replace('leftVid.mp4', 'params.csv') if not regress_intrinsics else None
    cap = cv2.VideoCapture(video_path)

    Mint, Mext = do_regression(keypoint_path=keypoint_path, params_path=params_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError('video not found')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for key, point3d in points3d.items():
        point_cam = world2cam(point3d, Mext)
        point_img = cam2img(point_cam, Mint)
        point_img = point_img.astype(np.int32)
        cv2.circle(frame, (point_img[0], point_img[1]), 5, (255, 0, 0), -1)
        cv2.putText(frame, str(key), (point_img[0], point_img[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    frame = cv2.resize(frame, (4*1280, 4*720))

    plt.imshow(frame)
    plt.show()


def regress_events(keypoint_path, use_prints=True, use_lm=False, use_ransac=False):
    '''
    keypoint_path: path to csv containing the keypoints
    '''
    import pandas as pd
    keypoint_path = keypoint_path
    keypoints_csv = pd.read_csv(keypoint_path, sep=';', skiprows=[0])


    for k in keypoints_csv.iterrows():
        keypoints_dict = {}
        flag = False
        for keypoint in k[1].keys():
            if 'flag' in keypoint and 'ball' not in keypoint and 2.01 > k[1][keypoint] > 1.99:
                key = int(keypoint.replace('_flag', ''))
                if key not in keypoints_dict:
                    keypoints_dict[key] = []
                coords = (k[1][keypoint.replace('flag', 'x')], k[1][keypoint.replace('flag', 'y')])
                keypoints_dict[key].append(coords)
                flag = True

        if flag:
            # params = None
            Mint, Mext = calc_cameramatrices(keypoints_dict, use_prints=use_prints, use_lm=use_lm, use_ransac=use_ransac)
            # print('intrinsics: \n', Mint)
            if use_prints:
                print(f'fx: {Mint[0, 0]}, fy: {Mint[1, 1]}, cx: {Mint[0, 2]}, cy: {Mint[1, 2]}')
                print('extrinsics: \n', Mext)
                print('-' * 50)
                # camera position
                R = Mext[:3, :3]
                t = Mext[:3, 3]
                camera_position = -np.linalg.inv(R) @ t
                print('camera position: ', camera_position)

                print('---------------------------------------')
                print('---------------------------------------')
                print('---------------------------------------')


if __name__ == '__main__':
    pass

