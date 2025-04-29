import os
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate inference dataset')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_path', type=str, help='Path to the model')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
from functools import partial
from sklearn.metrics import roc_auc_score
import pandas as pd
import shutil

from helper import mujoco_metric, binary_metrics, create_confusion_matrix
from helper import seed_worker, get_logs_path
from helper import world2cam, cam2img
from helper import _mujoco_simulation, transform_rotationaxes, inversetransform_rotationaxes
from helper import table_connections, WIDTH, HEIGHT, TABLE_HEIGHT
from helper import plot_roc_curve, count_missortings
from synthetictabletennis.data import TableTennisDataset, RealInferenceDataset
from synthetictabletennis.transformations import get_transforms, UnNormalizeImgCoords, Compose, RandomizeDetections, MotionBlur, NormalizeImgCoords
from synthetictabletennis.model import get_model
from synthetictabletennis.config import EvalConfig
import paths


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
use_noise = False


def get_save_path(folder=None):
    save_path = os.path.join(get_logs_path(), 'inference')
    if folder is not None:
        save_path = os.path.join(save_path, folder)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def synthetic_inference(config, model, save=False):
    '''Performs inference on the synthetic dataset.
    Args:
        config (EvalConfig): Configuration object
        model (nn.Module): Model to perform inference
        save (bool): If True, the results are saved
    '''
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    # set model to evaluation mode
    model = model.to(device)
    model.eval()

    # load dataset
    if use_noise:
        transforms = Compose([MotionBlur(0.5), RandomizeDetections(3), NormalizeImgCoords()])
    else:
        transforms = get_transforms(config, mode='test')
    dataset = TableTennisDataset(mode='test', transforms=transforms)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(config.BATCH_SIZE, 16),
                                             worker_init_fn=seed_worker, generator=g)

    # initialize metrics functions
    metric_fn = lambda x, y: torch.sum(torch.sqrt(torch.sum((x - y) ** 2, dim=1)))
    metricabs_fn = lambda x, y: torch.sum(torch.abs(torch.norm(x, dim=1) - torch.norm(y, dim=1)))
    metricangle_fn = lambda x, y: torch.sum(torch.rad2deg(torch.acos(torch.einsum('bi, bi -> b', x, y) / (torch.norm(x, dim=1) * torch.norm(y, dim=1)))))
    metric_pos_fn = lambda x, y, mask: torch.sum(torch.sum(torch.sqrt(torch.sum((x - y) ** 2, dim=-1)) * mask, dim=1) / torch.sum(mask, dim=1))

    # iterate over the dataset
    with torch.no_grad():
        # initialize metrics
        metrics, relmetrics, metrics_abs, metrics_angle, metrics_pos = 0, 0, 0, 0, 0
        metricsx, metricsy, metricsz = 0, 0, 0
        # metric_trajectory, metric_trajectory_before, metric_trajectory_after = 0, 0, 0
        TPs, TNs, FPs, FNs = 0, 0, 0, 0
        number = 0
        #for cam_num in range(testloader.dataset.num_cameras):
        for cam_num in range(1):
            print('--------------')
            print(f'Camera {cam_num + 1}')
            testloader.dataset.cam_num = cam_num
            for i, data in enumerate(tqdm(testloader)):
                r_img, table_img, mask, r_world, rotation, times, hits, __, __ = data
                r_img, table_img, mask = r_img.to(device), table_img.to(device), mask.to(device) # inputs
                r_world, rotation = r_world.to(device), rotation.to(device) # targets

                # forward pass
                B, T, D = r_img.shape
                pred_rotation, pred_position = model(r_img, table_img, mask)

                # All metrics are calculated in the ball's coordinate system -> transform (predicted and) gt rotations accordingly
                rotation = transform_rotationaxes(rotation, r_world)
                if config.transform_mode == 'global':
                    pred_rotation = transform_rotationaxes(pred_rotation, r_world)

                # calculate metrics
                metrics += metric_fn(pred_rotation, rotation)
                relmetrics += (torch.sqrt(torch.sum((pred_rotation - rotation) ** 2, dim=1)) / torch.linalg.norm(rotation, dim=1)).sum()
                metricsx += torch.sum(torch.abs(pred_rotation[:, 0] - rotation[:, 0]))
                metricsy += torch.sum(torch.abs(pred_rotation[:, 1] - rotation[:, 1]))
                metricsz += torch.sum(torch.abs(pred_rotation[:, 2] - rotation[:, 2]))
                metrics_abs += metricabs_fn(pred_rotation, rotation)
                metrics_angle += metricangle_fn(pred_rotation, rotation)
                metrics_pos += metric_pos_fn(pred_position, r_world, mask)
                TP, TN, FP, FN = binary_metrics(rotation, pred_rotation)
                TPs, TNs, FPs, FNs = TPs + TP, TNs + TN, FPs + FP, FNs + FN
                number += B
                # tmp = mujoco_metric(r_world.cpu().numpy(), times.cpu().numpy(), rotation.cpu().numpy(), pred_rotation.cpu().numpy(), mask.cpu().numpy(),
                #                     hits.cpu().numpy(), num_workers=min(config.BATCH_SIZE, 16))
                # metric_trajectory += tmp[0]
                # metric_trajectory_before += tmp[1]
                # metric_trajectory_after += tmp[2]

            # calculate metrics
            metrics /= number
            relmetrics /= number
            metrics_abs /= number
            metrics_angle /= number
            metrics_pos /= number
            # metric_trajectory /= number
            # metric_trajectory_before /= number
            # metric_trajectory_after /= number
            metricsx /= number
            metricsy /= number
            metricsz /= number
            accuracy = (TPs + TNs) / (TPs + TNs + FPs + FNs)
            TPs, TNs, FPs, FNs = TPs.cpu().numpy(), TNs.cpu().numpy(), FPs.cpu().numpy(), FNs.cpu().numpy()
            confusion_matrix_x = create_confusion_matrix(TPs[0], TNs[0], FPs[0], FNs[0])
            confusion_matrix_y = create_confusion_matrix(TPs[1], TNs[1], FPs[1], FNs[1])
            confusion_matrix_z = create_confusion_matrix(TPs[2], TNs[2], FPs[2], FNs[2])

            print('Synthetic Evaluation')
            print(f'3D Trajectory Error: {metrics_pos.item()*100:.4f}cm, Rotation Error: {metrics.item():.4f}Hz')
            # print(f'Metrics: {metrics.item():.1f}, Metrics_abs: {metrics_abs.item():.4f}, Metrics_angle: {metrics_angle.item():.4f}, '
            #       f'Metrics_pos: {metrics_pos.item()*100:.1f}cm')
            #print(f'Metrics_x: {metricsx.item():.1f}, Metrics_y: {metricsy.item():.1f}, Metrics_z: {metricsz.item():.1f}')
            # print(f'Accuracy x: {accuracy[0]:.4f}, Accuracy y: {accuracy[1]:.4f}, Accuracy z: {accuracy[2]:.4f}')

            if save:
                save_path = get_save_path(config.folder)
                metrics_dict = {
                    'rotation_error': metrics.item(),
                    '3D_trajectory_error': metrics_pos.item(),
                    'TPs': list(TPs),
                    'TNs': list(TNs),
                    'FPs': list(FPs),
                    'FNs': list(FNs),
                }
                dir = os.path.join(save_path, config.ident)
                os.makedirs(dir, exist_ok=True)
                pd.DataFrame([metrics_dict]).to_json(os.path.join(dir, f'synthetic_metrics_{cam_num}.json'))

            # plot with three confusion matrix images next to each other
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(confusion_matrix_x)
            ax[0].set_title('X')
            ax[1].imshow(confusion_matrix_y)
            ax[1].set_title('Y')
            ax[2].imshow(confusion_matrix_z)
            ax[2].set_title('Z')
            plt.suptitle('Confusion Matrix')
            if save:
                dir = os.path.join(save_path, config.ident)
                os.makedirs(dir, exist_ok=True)
                plt.savefig(os.path.join(save_path, config.ident, f'synthetic_confusion_matrix_{cam_num}.png'), dpi=300)
                plt.close()
            else:
                plt.show()


def real_inference(config, model, data_path, save=False):
    '''Performs inference on the real data.
    Args:
        config (EvalConfig): Configuration object
        model (nn.Module): Model to perform inference
        data_path (str): Path to the real data
        do_regression (bool): If True, regression metrics are calculated (time consuming)
        save (bool): If True, the results are saved
    '''
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    # set model to evaluation mode
    model = model.to(device)
    model.eval()

    # load dataset
    transforms = get_transforms(config, mode='test')
    denorm = UnNormalizeImgCoords()
    dataset = RealInferenceDataset(data_path, transforms=transforms)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(config.BATCH_SIZE, 16),
                                             worker_init_fn=seed_worker, generator=g)

    # initialize metrics functions
    # x and y are image coordinates
    def metric_pos2D_fn(pred_2D, gt_2D, mask):
        B, T, __ = pred_2D.shape
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    pred_2D[b, t], gt_2D[b, t] = np.array([0, 0]), np.array([0, 0]) # set to same value such that difference is 0
        return np.sum(np.sum(np.sqrt(np.sum((pred_2D - gt_2D) ** 2, axis=-1)) * mask, axis=1) / np.sum(mask, axis=1))

    # initialize metrics
    metrics_pos2D = 0
    TP, TN, FP, FN = 0, 0, 0, 0 # For frontspin vs backspin
    number = 0
    scores = []  # needed for ROC-AUC and number of missortings
    labels = []  # needed for ROC-AUC and number of missortings

    # iterate over the dataset
    with torch.no_grad():
        # because regression takes so long, first calculate the other metrics
        for i, data in enumerate(tqdm(testloader)):
            r_img, table_img, mask, times, hits, Mint, Mext, spin_class = data
            r_img, table_img, mask = r_img.to(device), table_img.to(device), mask.to(device)
            Mint, Mext = Mint.to(device), Mext.to(device)
            B, T, D = r_img.shape

            # forward pass
            pred_rotation, pred_position = model(r_img, table_img, mask)
            # transform prediction into local coordinate system
            if config.transform_mode == 'global':
                pred_rotation_local = transform_rotationaxes(pred_rotation, pred_position)
            else:
                pred_rotation_local = pred_rotation

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            for b in range(B):
                # binary metrics
                if spin_class[b] == 1: #Frontspin
                    if pred_rotation_local[b, 1] > 0:
                        TP += 1
                    else:
                        FN += 1
                elif spin_class[b] == 2: #Backspin
                    if pred_rotation_local[b, 1] < 0:
                        TN += 1
                    else:
                        FP += 1
                # ROC-AUC and missortings
                if spin_class[b] in [1, 2]: # only consider if spin class is annotated
                    scores.append(pred_rotation_local[b, 1].item())
                    labels.append(1 if spin_class[b] == 1 else 0) # The methods use Frontspin=1 and Backspin=0

            # denormalization of image coordinates to calculate the 2D metric
            data_gt = denorm({'r_img': r_img.cpu().numpy(), 'table_img': table_img.cpu().numpy()})
            r_img, table_img = data_gt['r_img'], data_gt['table_img']

            # calculate metrics
            pred_pos_2D = cam2img(world2cam(pred_position, Mext), Mint)
            metrics_pos2D += metric_pos2D_fn(pred_pos_2D.cpu().numpy(), r_img, mask.cpu().numpy())

            # plot image coordinates
            for b in range(B):
                r_img_b = r_img[b, mask.cpu().numpy()[b] == 1]
                pred_position_2D_b = pred_pos_2D[b, mask[b] == 1].cpu().numpy()
                plot_imagecoords(r_img_b, pred_position_2D_b, table_img[b],
                                 title=f'reprojection{number+b}',
                                 config=config, save=save)

            number += B

        # calculate metrics
        metrics_pos2D /= number
        normed_metrics_pos2D = metrics_pos2D / (WIDTH**2 + HEIGHT**2)**0.5 # normalize by image size
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_plus = 2 * TP / (2 * TP + FP + FN)
        f1_minus = 2 * TN / (2 * TN + FN + FP)
        macro_f1 = (f1_plus + f1_minus) / 2

        # plot confusion matrix for binary prediction
        confusion_matrix = create_confusion_matrix(TP, TN, FP, FN, title=None, dpi=300)
        if save:
            save_path = get_save_path(config.folder)
            dir = os.path.join(save_path, config.ident)
            os.makedirs(dir, exist_ok=True)
            plt.imsave(os.path.join(dir, 'confusion_matrix.png'), confusion_matrix)
        else:
            plt.imshow(confusion_matrix)
            plt.show()

        # ROC-AUC
        roc_auc = roc_auc_score(labels, scores)
        # Number of missortings
        missortings, opt_threshold = count_missortings(labels, scores)
        # print(f'Number of missortings {missortings} out of {len(labels)} at optimal threshold {opt_threshold}')
        # plot ROC curve
        dir = os.path.join(save_path, config.ident)
        os.makedirs(dir, exist_ok=True)
        plot_roc_curve(labels, scores, save_path=os.path.join(dir, 'roc_curve.png'), show_thresholds=True)

        print('Real Evaluation')
        print(f'2D Reprojection Error: {metrics_pos2D:.1f}pixel, 2D Reprojection Error: {normed_metrics_pos2D*100:.2f}%, Accuracy: {accuracy*100:.1f}%, Macro F1: {macro_f1:.3f}, ROC-AUC: {roc_auc:.3f}')

        if save:
            metrics_dict = {
                '2D_reprojection_error': normed_metrics_pos2D,
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'roc_auc': roc_auc,
            }
            dir = os.path.join(save_path, config.ident)
            os.makedirs(dir, exist_ok=True)
            pd.DataFrame([metrics_dict]).to_json(os.path.join(dir, f'real_metrics.json'))

        print('---------------------------')


def _get_bounce_coordinates(r_img, Mint, Mext):
    '''Given the image coordinate at time of bounce, the corresponding world coordinate is calculated using the knowledge of the table height.
    Args:
        r_img (np.array): Detected ball position at bounce. Euclidean, not homogenous coordinates!
        Mint (np.array): Intrinsic camera matrix at bounce. 4x4 matrix.
        Mext (np.array): Extrinsic camera matrix at bounce. 3x3 matrix.
    '''
    h = TABLE_HEIGHT + 0.02  # height of the table in world coordinates (known z_w)
    # get inverted camera matrices
    Next = np.linalg.inv(Mext)
    # get camera coordinates with arbitrary scaling
    rho = np.linalg.solve(Mint[:3, :3], np.array([r_img[0], r_img[1], 1]))
    # get z coordinate in camera coordinates by solving z_w = Mext^-1[2,:] @ (rho * z_c) + Mext^-1[2,3]
    z_c = (h - Next[2, 3]) / ( Next[2, 0] * rho[0] + Next[2, 1] * rho[1] + Next[2, 2])
    # using the previously calculated z_c to calculate all world coordinates
    r_cam = z_c * rho
    r_world = np.linalg.solve(Mext, np.concatenate([r_cam, np.array([1])]))
    r_world = r_world[:3] / r_world[3] # homogeneous to euclidean coordinates
    return r_world


def _min_fun(x0, w0, r_img, times, Mint, Mext, hits=None):
    '''Objective function for the least squares optimization of regression_metrics().
    Args:
        x0 (np.array): Initial guess for the optimization (r0, v0), shape (6,)
        w0 (np.array): Initial rotation, shape (3,)
        r_img (np.array): Ground truth mage coordinates, shape (T, 2)
        times (np.array): Timestamps of the detections, shape (T,)
        Mint (np.array): Intrinsic camera matrix
        Mext (np.array): Extrinsic camera matrix
        hits (np.array): Hit time
    '''
    # set up r0 and v0 from x0 (these parameters are optimized by the regression)
    r0 = x0[:3]
    speed, phi, theta = x0[3], x0[4], x0[5] # phi and theta in radians
    v0 = np.array([
        speed * np.sin(theta) * np.cos(phi),
        speed * np.sin(theta) * np.sin(phi),
        speed * np.cos(theta)
    ], dtype=np.float64)

    # physics simulation with regressed r0, v0 and fixed w0
    regressed_r = _mujoco_simulation(r0, v0, w0, times)
    # error of reprojected trajectory
    regressed_r_img = cam2img(world2cam(regressed_r, Mext), Mint)
    error = np.mean(np.linalg.norm(regressed_r_img - r_img, axis=-1))

    # Get the index of the hit in the trajectory
    assert len(hits) == 1, 'Only one hit is supported.'
    diff = np.abs(times - hits[0])
    hit_idx = np.argmin(diff)

    # error of the 3D bounce location at time of the hit (the only 3D location we know)
    bounce_coords = _get_bounce_coordinates(r_img[hit_idx], Mint, Mext)
    condition = np.linalg.norm(100*(bounce_coords - regressed_r[hit_idx])) # factor 100 to compare cm to pixel instead of m to pixel
    error += condition
    return error


def plot_imagecoords(r_gt, r_pred, table_img, title='', config=None, save=False):
    '''Plot the image coordinates of original and regressed trajectory + plot the table points in the image plane.
    Args:
        r_gt (np.array): Ground truth image coordinates
        r_pred (np.array): Predicted image coordinates
        table_img (np.array): Table points in image coordinates
    '''
    # draw lines between the table points
    for connection in table_connections:
        plt.plot(table_img[connection, 0], table_img[connection, 1], 'k')
    # draw ground truth trajectory
    plt.plot(r_gt[:, 0], r_gt[:, 1], 'r', label='Ground truth')
    # draw regressed trajectory
    plt.plot(r_pred[:, 0], r_pred[:, 1], 'b--', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, WIDTH)
    plt.ylim(HEIGHT, 0)
    #plt.title(title)
    plt.legend()
    if save:
        assert config is not None, 'Need config for saving'
        save_path = get_save_path(config.folder)
        dir = os.path.join(save_path, config.ident)
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f'imagecoords_{title}.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def evalute_multiple(model_paths):
    base_path = paths.checkpoint_path
    for model_path in model_paths:
        model_path = os.path.join(base_path, model_path)
        loaded_dict = torch.load(model_path, weights_only=False)
        config = EvalConfig(loaded_dict['additional_info']['name'], loaded_dict['additional_info']['size'],
                            loaded_dict['additional_info']['tabletoken_mode'], loaded_dict['additional_info']['transform_mode'])
        config.ident = loaded_dict['identifier']
        config.pos_embedding = loaded_dict['additional_info']['pos_embedding']
        folder = os.path.normpath(model_path).split(os.sep)[-3]
        config.folder = folder

        # Delete old inference results
        save_path = get_save_path(config.folder)
        dir = os.path.join(save_path, config.ident)
        if os.path.exists(dir):
            shutil.rmtree(dir)

        model = get_model(config.name, config.size, config.tabletoken_mode, config.pos_embedding)
        model.load_state_dict(loaded_dict['model_state_dict'])
        folder = os.path.normpath(model_path).split(os.sep)[-3]
        print('Loaded model from folder', folder)
        print('Loaded model with identifier', loaded_dict['identifier'])
        synthetic_inference(config, model, save=True)
        data_path = os.path.join(paths.data_path, 'tabletennis_annotations_processed')
        config.BATCH_SIZE = 1
        real_inference(config, model, data_path, save=True)


def main():
    base_path = paths.checkpoint_path
    model_path = os.path.join(base_path, args.model_path)
    loaded_dict = torch.load(model_path, weights_only=False)
    config = EvalConfig(loaded_dict['additional_info']['name'], loaded_dict['additional_info']['size'],
                        loaded_dict['additional_info']['tabletoken_mode'], loaded_dict['additional_info']['transform_mode'])
    config.ident = loaded_dict['identifier']
    config.pos_embedding = loaded_dict['additional_info']['pos_embedding']
    folder = os.path.normpath(model_path).split(os.sep)[-3]
    config.folder = folder

    # Delete old inference results
    save_path = get_save_path(config.folder)
    dir = os.path.join(save_path, config.ident)
    if os.path.exists(dir):
        shutil.rmtree(dir)

    model = get_model(config.name, config.size, config.tabletoken_mode, config.pos_embedding)
    model.load_state_dict(loaded_dict['model_state_dict'])
    folder = os.path.normpath(model_path).split(os.sep)[-3]
    print('Loaded model from folder', folder)
    print('Loaded model with identifier', loaded_dict['identifier'])
    synthetic_inference(config, model, save=True)
    data_path = os.path.join(paths.data_path, 'tabletennis_annotations_processed')
    config.BATCH_SIZE = 1
    real_inference(config, model, data_path, save=True)


if __name__ == '__main__':
    main()
    pass
