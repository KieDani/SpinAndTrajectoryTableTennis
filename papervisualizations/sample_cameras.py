import numpy as np
import matplotlib.pyplot as plt

from synthetictabletennis.data import TableTennisDataset
from helper import table_points, table_connections, HEIGHT, WIDTH
from helper import world2cam, cam2img
from helper import get_data_path


def main():
    dataset = TableTennisDataset('train')
    sample = dataset[0]
    r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext = sample
    length = int(mask.sum().round())
    r_img = r_img[:length]
    table_img = table_img[:length]
    r_world = r_world[:length]

    sampled_r_imgs = []
    sampled_table_imgs = []
    for i in range(1, 10):
        stuff = dataset.sample_camera(r_world) # (Mint, Mext, r_img, table_img)
        sampled_r_imgs.append(stuff[2])
        sampled_table_imgs.append(stuff[3])

    # Visualize the sampled cameras
    for i, (r, t) in enumerate(zip([r_img]+sampled_r_imgs, [table_img]+sampled_table_imgs)):
        for connection in table_connections:
            plt.plot(t[connection, 0], t[connection, 1], 'k')
        plt.plot(r[:, 0], r[:, 1])
        plt.scatter(r[0, 0], r[0, 1], c='g', label='Start')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(0, WIDTH)
        plt.ylim(HEIGHT, 0)
        # plt.show()
        plt.savefig(f'camera{i}.png', dpi=300)
        plt.close()
        print(f'Saved camera{i}.png')


if __name__ == '__main__':
    main()
