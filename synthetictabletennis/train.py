import os
if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--blur_strength', type=float, default=0.4)
    parser.add_argument('--stop_prob', type=float, default=0.5)
    parser.add_argument('--randomize_std', type=float, default=2)
    parser.add_argument('--model_name', type=str, default='connectstage')
    parser.add_argument('--model_size', type=str, default='large')
    parser.add_argument('--token_mode', type=str, default='stacked')
    parser.add_argument('--loss_mode', type=str, default='distance')
    parser.add_argument('--loss_target', type=str, default='both')
    parser.add_argument('--transform_mode', type=str, default='global')
    parser.add_argument('--pos_embedding', type=str, default='rotary')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
if __name__ == '__main__' and args.debug:
    torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
import random
import numpy as np

from helper import SummaryWriter, seed_worker
from helper import mujoco_metric, binary_metrics
from helper import update_ema
from helper import create_confusion_matrix, transform_rotationaxes
from helper import save_model
from synthetictabletennis.data import TableTennisDataset
from synthetictabletennis.transformations import get_transforms
from synthetictabletennis.model import get_model
from synthetictabletennis.config import TrainConfig

device = 'cuda:0'
debug = False

def run(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    logs_path = config.get_logs_path(debug)
    writer = SummaryWriter(logs_path)

    model = get_model(config.name, config.size, config.tabletoken_mode, config.pos_embedding).to(device)
    model_ema = get_model(config.name, config.size, config.tabletoken_mode, config.pos_embedding).to(device)
    model_ema = update_ema(model, model_ema, 0) # copies the model to model_ema completely

    num_workers = 0 if debug else min(config.BATCH_SIZE, 16)
    train_transforms = get_transforms(config, 'train')
    val_transforms = get_transforms(config, 'val')
    trainset = TableTennisDataset('train', transforms=train_transforms)
    valset = TableTennisDataset('val', transforms=val_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True,
                                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config.BATCH_SIZE, shuffle=False,
                                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_metric = 1e8
    val(model_ema, valloader, writer, -1, device, config)
    for epoch in range(config.NUM_EPOCHS):
        model_ema = train(model, model_ema, trainloader, optimizer, writer, epoch, device, config)
        metric = val(model_ema, valloader, writer, epoch, device, config)
        # save model if metric improved
        if metric < best_metric:
            best_metric = metric
            save_model(model_ema, config, epoch, debug)


def train(model, model_ema, trainloader, optimizer, writer, epoch, device, config):
    # L2 Loss per component
    if config.loss_mode == 'mse':
        loss_fn = torch.nn.MSELoss()
    # L2 loss for the whole vector
    elif config.loss_mode == 'distance':
        loss_fn = lambda angle, pred_angle: torch.sum(torch.sqrt(torch.sum((angle-pred_angle)**2, dim=1)))
    # Combining loss for angle and norm
    elif config.loss_mode == 'absang':
        loss_fn = lambda angle, pred_angle: torch.abs(torch.norm(angle, dim=1) - torch.norm(pred_angle, dim=1)).mean() + torch.abs(torch.rad2deg(torch.acos(torch.einsum('bi, bi -> b', angle, pred_angle) / (torch.linalg.norm(angle, dim=1) * torch.linalg.norm(pred_angle, dim=1) + 1e-3)))).mean()
    else:
        raise ValueError(f'Unknown loss mode {config.loss_mode}')

    # If we are using the multistage model, we want to prevent that the rotation influences the learning of the positions
    if config.name == 'multistage':
        model.full_backprop = False if config.loss_target == 'both' else True

    model.train()
    iteration = epoch * len(trainloader)
    for i, data in enumerate(tqdm(trainloader)):
        r_img, table_img, mask, r_world, rotation, __, hits, __, __ = data
        r_img, table_img, mask, rotation = r_img.to(device), table_img.to(device), mask.to(device), rotation.to(device)
        r_world = r_world.to(device)

        optimizer.zero_grad()
        pred_rotation, pred_position = model(r_img, table_img, mask)
        # If network output is in ball's coordinate system, transform the ground truth to the local coordinate system
        if config.transform_mode == 'local':
            rotation = transform_rotationaxes(rotation, r_world)
        loss_rot = loss_fn(pred_rotation, rotation) if config.loss_target in ['rotation', 'both'] else torch.tensor(0)
        loss_pos = torch.sum(torch.nn.functional.mse_loss(pred_position, r_world, reduction='none') * mask.unsqueeze(-1)) / torch.sum(mask) if config.loss_target in ['position', 'both'] else torch.tensor(0)
        loss = loss_rot + loss_pos
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # gradient clipping
        optimizer.step()

        model_ema = update_ema(model, model_ema, config.ema_decay)

        writer.add_scalar('train/loss', loss.item(), iteration+i)
        writer.add_scalar('train/loss rotation', loss_rot.item(), iteration+i)
        writer.add_scalar('train/loss position', loss_pos.item(), iteration+i)

    return model_ema


def val(model, valloader, writer, epoch, device, config):
    loss_fn = torch.nn.MSELoss()
    metric_fn = lambda x, y: torch.sum(torch.sqrt(torch.sum((x-y)**2, dim=1)))
    metricx_fn = lambda x, y: torch.sum(torch.abs(x[:, 0] - y[:, 0]))
    metricy_fn = lambda x, y: torch.sum(torch.abs(x[:, 1] - y[:, 1]))
    metricz_fn = lambda x, y: torch.sum(torch.abs(x[:, 2] - y[:, 2]))
    metricabs_fn = lambda x, y: torch.sum(torch.abs(torch.norm(x, dim=1) - torch.norm(y, dim=1)))
    metricangle_fn = lambda x, y: torch.sum(torch.rad2deg(torch.acos(torch.einsum('bi, bi -> b', x, y) / (torch.norm(x, dim=1) * torch.norm(y, dim=1)))))
    metric_pos_fn = lambda x, y, mask: torch.sum(torch.sum(torch.sqrt(torch.sum((x-y)**2, dim=-1)) * mask, dim=1) / torch.sum(mask, dim=1))

    model.eval()

    for cam_num in range(valloader.dataset.num_cameras):
        valloader.dataset.cam_num = cam_num
        loss, metric, relmetric, metricx, metricy, metricz, metricabs, metricangle = 0, 0, 0, 0, 0, 0, 0, 0
        metric_trajectory, metric_trajectory_before, metric_trajectory_after = 0, 0, 0
        metric_position = 0
        TPs, TNs, FPs, FNs = 0, 0, 0, 0
        number = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader)):
                r_img, table_img, mask, r_world, rotation, times, hits, __, __ = data
                r_img, table_img, mask, rotation = r_img.to(device), table_img.to(device), mask.to(device), rotation.to(device)
                r_world = r_world.to(device)
                B, T, D = r_img.shape
                pred_rotation, pred_position = model(r_img, table_img, mask)

                # All metrics are calculated in the ball's coordinate system -> transform (predicted and) gt rotations accordingly
                rotation = transform_rotationaxes(rotation, r_world)
                if config.transform_mode == 'global':
                    pred_rotation = transform_rotationaxes(pred_rotation, r_world)

                loss = loss_fn(pred_rotation, rotation)
                metric += metric_fn(pred_rotation, rotation)
                relmetric += (torch.sqrt(torch.sum((pred_rotation-rotation)**2, dim=1)) / torch.linalg.norm(rotation, dim=-1)).sum()
                metricx += metricx_fn(pred_rotation, rotation)
                metricy += metricy_fn(pred_rotation, rotation)
                metricz += metricz_fn(pred_rotation, rotation)
                metricabs += metricabs_fn(pred_rotation, rotation)
                metricangle += metricangle_fn(pred_rotation, rotation)
                metric_position += metric_pos_fn(pred_position, r_world, mask)
                tmp = binary_metrics(pred_rotation, rotation)
                TPs, TNs, FPs, FNs = TPs + tmp[0], TNs + tmp[1], FPs + tmp[2], FNs + tmp[3]
                if (epoch % 100 == 0 or epoch == -1) and cam_num==0:
                    tmp = mujoco_metric(
                        r_world.cpu().numpy(), times.cpu().numpy(), rotation.cpu().numpy(),
                        pred_rotation.cpu().numpy(), mask.cpu().numpy(), hits.cpu().numpy(),
                        num_workers=min(config.BATCH_SIZE, 16)
                    )
                    metric_trajectory, metric_trajectory_before, metric_trajectory_after = metric_trajectory + tmp[0], metric_trajectory_before + tmp[1], metric_trajectory_after + tmp[2]
                number += B
            loss /= number
            metric /= number
            relmetric /= number
            metricx /= number
            metricy /= number
            metricz /= number
            metricabs /= number
            metricangle /= number
            metric_trajectory /= number
            metric_trajectory_before /= number
            metric_trajectory_after /= number
            metric_position /= number
            accuracy = (TPs + TNs) / (TPs + TNs + FPs + FNs)

        writer.add_scalar(f'val{cam_num}/loss', loss.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric', metric.item(), epoch)
        writer.add_scalar(f'val{cam_num}/relative metric', relmetric.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric x', metricx.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric y', metricy.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric z', metricz.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric abs', metricabs.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric angle', metricangle.item(), epoch)
        # Calculating the trajectory metric takes a lot of time...
        if (epoch % 100 == 0 or epoch == -1) and cam_num==0:
            writer.add_scalar(f'val{cam_num}/metric trajectory', metric_trajectory, epoch)
            writer.add_scalar(f'val{cam_num}/metric trajectory before', metric_trajectory_before, epoch)
            writer.add_scalar(f'val{cam_num}/metric trajectory after', metric_trajectory_after, epoch)
        writer.add_scalar(f'val{cam_num}/accuracy x', accuracy[0].item(), epoch)
        writer.add_scalar(f'val{cam_num}/accuracy y', accuracy[1].item(), epoch)
        writer.add_scalar(f'val{cam_num}/accuracy z', accuracy[2].item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric position', metric_position.item(), epoch)
        # Writing the confusion matrix only every tenth time to save space
        if epoch % 10 == 0 or epoch == -1:
            TPs, TNs, FPs, FNs = TPs.cpu().numpy(), TNs.cpu().numpy(), FPs.cpu().numpy(), FNs.cpu().numpy()
            writer.add_image(f'val{cam_num}/confusion matrix x', create_confusion_matrix(TPs[0], TNs[0], FPs[0], FNs[0]), epoch, dataformats='HWC')
            writer.add_image(f'val{cam_num}/confusion matrix y', create_confusion_matrix(TPs[1], TNs[1], FPs[1], FNs[1]), epoch, dataformats='HWC')
            writer.add_image(f'val{cam_num}/confusion matrix z', create_confusion_matrix(TPs[2], TNs[2], FPs[2], FNs[2]), epoch, dataformats='HWC')
        if cam_num == 0:
            # Add predicted and gt rotations as text to tensorboard
            writer.add_text(f'Predicted Rotations', str(np.round(pred_rotation[:4].cpu().numpy(), 1)), epoch)
            writer.add_text(f'GT Rotations', str(np.round(rotation[:4].cpu().numpy(), 1)), epoch)
            #Add hparams to tensorboard
            writer.add_hparams2(config.get_hparams(), {'metric': metric.item()})

    valloader.dataset.cam_num = 0
    model.train()

    return metric.item()


def main():
    global debug
    debug = args.debug
    config = TrainConfig(args.lr, args.model_name, args.model_size, debug, args.folder, args.exp_id)
    config.blur_strength = args.blur_strength
    config.stop_prob = args.stop_prob
    config.randomize_std = args.randomize_std
    config.tabletoken_mode = args.token_mode
    config.loss_mode = args.loss_mode
    config.loss_target = args.loss_target
    config.transform_mode = args.transform_mode
    config.pos_embedding = args.pos_embedding
    run(config)


if __name__ == '__main__':
    main()