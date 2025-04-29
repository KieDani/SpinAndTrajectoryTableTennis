import numpy as np
import random
import torch
import os
import pandas as pd

from helper import get_Mext, cam2img, world2cam
from helper import HEIGHT, WIDTH, base_fx, base_fy
from helper import get_data_path
from helper import table_points, TABLE_HEIGHT


class TableTennisDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', transforms=None):
        self.mode = mode
        path = get_data_path()
        self.data_paths = sorted([os.path.join(path, f'trajectory_{i:04}') for i, __ in enumerate(os.listdir(path))])
        self.rnd = random.Random(0)
        self.rnd.shuffle(self.data_paths)
        # randomly split in train, val, test
        if mode == 'train':
            self.data_paths = self.data_paths[:int(0.7 * len(self.data_paths))]
        elif mode == 'val':
            self.data_paths = self.data_paths[int(0.7 * len(self.data_paths)):int(0.8 * len(self.data_paths))]
        elif mode == 'test':
            self.data_paths = self.data_paths[int(0.8 * len(self.data_paths)):]
        else:
            raise ValueError(f'Unknown mode {mode}')
        self.length = len(self.data_paths)

        self.sequence_len = 50 # crop sequence if it is longer, else padding of sequence
        self.static_camera = False
        if self.static_camera: print('Static camera is used for training!')

        self.transforms = transforms

        # minimum and maximum phi that is sampled for the camera position
        self.sampled_phis = (20, 160)

        self.num_cameras = 1 if mode in ['val', 'test'] else 1
        self.cam_num = 0


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        r_world = np.load(os.path.join(data_path, 'positions.npy'))
        times = np.load(os.path.join(data_path, 'times.npy'))
        hits = np.load(os.path.join(data_path, 'hits.npy'))
        rotation = np.load(os.path.join(data_path, 'rotations.npy'))[0]
        blur_positions = np.load(os.path.join(data_path, 'blur_positions.npy'))
        blur_times = np.load(os.path.join(data_path, 'blur_times.npy'))
        if self.mode == 'train' and self.static_camera == False:
            Mint, Mext, r_img, table_img = self.sample_camera(r_world)
        else:
            Mint = np.load(os.path.join(data_path, 'Mint.npy'))
            Mext = np.load(os.path.join(data_path, 'Mext.npy'))
            assert np.sum(Mext[1:] - Mext[:-1]) < 1e-6, 'Actual computations are only correct for static camera'
            Mint, Mext = Mint[0], Mext[0]
            Mext, Mint = self.transform_evaluation_camera(self.cam_num, Mext, Mint)
            r_cam = world2cam(r_world, Mext)
            r_img = cam2img(r_cam, Mint)
            table_cam = world2cam(table_points, Mext)
            table_img = cam2img(table_cam, Mint)

        # mask is needed to indicate which values are padding (0) and which are real values (1)
        T, __ = r_img.shape
        mask = np.empty((self.sequence_len,), dtype=np.bool)
        mask[:T] = True # real values
        mask[T:] = False # padding

        # crop or pad sequence
        max_t = min(T, self.sequence_len)
        tmp = np.zeros((self.sequence_len, 2))
        tmp[:max_t] = r_img
        r_img = tmp
        tmp = np.zeros((self.sequence_len, 3))
        tmp[:max_t] = r_world
        r_world = tmp
        tmp = np.zeros((self.sequence_len))
        tmp[:max_t] = times
        times = tmp

        # apply transforms
        data = {
            'r_img': r_img,
            'r_world': r_world,
            'times': times,
            'hits': hits,
            'rotation': rotation,
            'mask': mask,
            'table_img': table_img,
            'Mint': Mint,
            'Mext': Mext,
            'blur_positions': blur_positions,
            'blur_times': blur_times
        }
        if self.transforms is not None: data = self.transforms(data)
        r_img, table_img, mask, r_world, rotation = data['r_img'], data['table_img'], data['mask'], data['r_world'], data['rotation']
        times, hits, Mint, Mext = data['times'], data['hits'], data['Mint'], data['Mext']
        blur_positions, blur_times = data['blur_positions'], data['blur_times']

        r_img, table_img, mask = torch.tensor(r_img, dtype=torch.float32), torch.tensor(table_img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        r_world, rotation, times = torch.tensor(r_world, dtype=torch.float32), torch.tensor(rotation, dtype=torch.float32), torch.tensor(times, dtype=torch.float32)
        hits, Mint, Mext = torch.tensor(hits, dtype=torch.float32), torch.tensor(Mint, dtype=torch.float32), torch.tensor(Mext, dtype=torch.float32)
        # r_img is input, r_world is not really used. Rotation is the label. Rest is additional information that could be used for regressions
        return r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext

    def sample_camera(self, r_world):
        valid = False
        while not valid:
            fx, fy = self.rnd.uniform(0.8 * base_fx, 2.5 * base_fx), self.rnd.uniform(0.8 * base_fy, 2.5 * base_fy)
            Mint = np.array([[fx, 0, (WIDTH - 1) / 2], [0, fy, (HEIGHT - 1) / 2], [0, 0, 1]])

            # extrinsic matrix
            # distance between 5m and 15m
            distance = self.rnd.uniform(10, 15)
            # phi angle between 20 and 160 degrees
            phi = self.rnd.uniform(self.sampled_phis[0], self.sampled_phis[1])
            # theta angle between 30 and 70 degrees
            theta = self.rnd.uniform(30, 70)
            # lookat point somewhere around the center of the table
            lookat = np.array((self.rnd.uniform(-0.2, 0.2), self.rnd.uniform(-0.2, 0.2), TABLE_HEIGHT))

            # camera location
            c = np.array([distance * np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                          distance * np.sin(np.radians(theta)) * np.sin(np.radians(phi)),
                          distance * np.cos(np.radians(theta))])
            c += np.array([0., 0., TABLE_HEIGHT])
            # forward direction
            f = -(c - lookat) / np.linalg.norm(c - lookat)
            # right direction (choose a random vector approximately in the x-y plane)
            epsilon = self.rnd.uniform(-0.1, 0.1)  # small random value that controls the deviation from the x-y plane
            r = np.array([-f[1] / f[0] - f[2] / f[0] * epsilon, 1, epsilon])
            r /= np.linalg.norm(r)
            # up direction
            u = -np.cross(f, r)
            if u[2] < 0:  # The up vector has to be in the positive z direction
                r = np.array([f[1] / f[0] - f[2] / f[0] * epsilon, -1, epsilon])  # choose the other direction for r_y
                r /= np.linalg.norm(r)
                u = -np.cross(f, r)
            u /= np.linalg.norm(u)
            # extrinsic matrix
            Mext = get_Mext(c, f, r)

            # calculate image coordinates of trajectory with estimated camera matrices
            r_cam = world2cam(r_world, Mext)
            r_img = cam2img(r_cam, Mint)
            # calculate the table position in image coordinates
            table_cam = world2cam(table_points, Mext)
            table_img = cam2img(table_cam, Mint)
            # check if trajectory is completely inside the image
            valid = np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT])))
            # check if trajectory is not too small in the image
            valid = valid and (r_img[:, 0].max() - r_img[:, 0].min() > 0.15 * WIDTH or r_img[:, 1].max() - r_img[:, 1].min() > 0.15 * HEIGHT)
        return Mint, Mext, r_img, table_img

    def transform_evaluation_camera(self, cam_num, Mext, Mint):
        '''Load specific evaluation cameras'''
        assert self.mode in ['val', 'test'], 'Evaluation camera can only be used for validation or test set'
        if cam_num == 0:
            pass
        else:
            raise ValueError(f'Unknown camera number {cam_num}')
        return Mext, Mint


class RealInferenceDataset(torch.utils.data.Dataset):
    '''Dataset for real data inference'''
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms

        self.data_paths = sorted([os.path.join(path, f'trajectory_{i:04}') for i, __ in enumerate(os.listdir(path))])
        self.length = len(self.data_paths)

        self.sequence_len = 50 # crop sequence if it is longer, else padding of sequence
        self.static_camera = False
        if self.static_camera: print('Static camera is used!')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        r_img = np.load(os.path.join(data_path, 'r_img.npy'))
        times = np.load(os.path.join(data_path, 'times.npy'))
        hits = np.load(os.path.join(data_path, 'hits.npy'))
        # TODO: At the moment we have one matrix per sequence -> In the future maybe adjust it to one matrix per frame
        Mint = np.load(os.path.join(data_path, 'Mint.npy'))
        Mext = np.load(os.path.join(data_path, 'Mext.npy'))
        spin_class = np.load(os.path.join(data_path, 'spin_class.npy'))

        # TODO: In the future, use annotated points directly instead of regressed matrices
        table_cam = world2cam(table_points, Mext)
        table_img = cam2img(table_cam, Mint)

        # mask is needed to indicate which values are padding (0) and which are real values (1)
        T, __ = r_img.shape
        mask = np.empty((self.sequence_len,), dtype=np.bool)
        mask[:T] = True
        mask[T:] = False

        # crop or pad sequence
        max_t = min(T, self.sequence_len)
        tmp = np.zeros((self.sequence_len, 2))
        tmp[:max_t] = r_img
        r_img = tmp
        tmp = np.zeros((self.sequence_len))
        tmp[:max_t] = times
        times = tmp

        # apply transforms
        data = {
            'r_img': r_img,
            'times': times,
            'hits': hits,
            'mask': mask,
            'table_img': table_img,
            'Mint': Mint,
            'Mext': Mext,
            'spin_class': spin_class
        }
        if self.transforms is not None: data = self.transforms(data)
        r_img, table_img, mask = data['r_img'], data['table_img'], data['mask']
        times, hits, Mint, Mext, spin_class = data['times'], data['hits'], data['Mint'], data['Mext'], data['spin_class']

        dtype = torch.float32
        r_img, table_img, mask = torch.tensor(r_img, dtype=dtype), torch.tensor(table_img, dtype=dtype), torch.tensor(mask, dtype=dtype)
        times, hits, Mint, Mext = torch.tensor(times, dtype=dtype), torch.tensor(hits, dtype=dtype), torch.tensor(Mint, dtype=dtype), torch.tensor(Mext, dtype=dtype)
        spin_class = torch.tensor(spin_class, dtype=dtype)

        return r_img, table_img, mask, times, hits, Mint, Mext, spin_class


if __name__ == '__main__':
    import tqdm
    from helper import transform_rotationaxes
    dataset = TableTennisDataset(mode='test')
    avgrot_pos = np.array([0., 0, 0])
    avgrot_neg = np.array([0., 0, 0])
    stdrot_pos = np.array([0., 0, 0])
    stdrot_neg = np.array([0., 0, 0])
    num_pos = np.array([0., 0, 0])
    num_neg = np.array([0., 0, 0])
    number_frames = 0
    for i, data in enumerate(tqdm.tqdm(dataset)):
        rotation, r_gt = data[4], data[3]
        mask = data[2]
        rotation = transform_rotationaxes(torch.tensor(rotation), torch.tensor(r_gt)).numpy()
        avgrot_pos += np.where(rotation > 0, rotation, 0)
        stdrot_pos += np.where(rotation > 0, rotation ** 2, 0)
        num_pos += np.where(rotation > 0, 1, 0)
        avgrot_neg += np.where(rotation < 0, rotation, 0)
        stdrot_neg += np.where(rotation < 0, rotation ** 2, 0)
        num_neg += np.where(rotation < 0, 1, 0)
        number_frames += mask.sum()
    print('Positive rotations:', avgrot_pos / num_pos)
    print('Std of positive:', np.sqrt(stdrot_pos / num_pos - (avgrot_pos / num_pos) ** 2))
    print('Number of positive:', num_pos)
    print('Negative rotations:', avgrot_neg / num_neg)
    print('Std of negative:', np.sqrt(stdrot_neg / num_neg - (avgrot_neg / num_neg) ** 2))
    print('Number of negative:', num_neg)
    print('Number of frames:', number_frames)

    print('-'*50)

    from paths import data_path
    dataset = RealInferenceDataset(path=os.path.join(data_path, 'tabletennis_annotations_processed/'))
    number_frames = 0
    num_pos, num_neg = 0, 0
    for i, data in enumerate(tqdm.tqdm(dataset)):
        mask = data[2]
        spin_class = data[7]
        if spin_class == 1:
            num_pos += 1
        elif spin_class == 2:
            num_neg += 1
        else:
            print('Number of frames without spin class:', mask.sum())
            continue # spin class was not annotated
        number_frames += mask.sum()
    print('Number of frames:', number_frames)
    print('Number of positive:', num_pos)
    print('Number of negative:', num_neg)






