import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from synthetictabletennis.inference import evalute_multiple

architecture = [
    'architecture/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:both_mode:distance_trans:global_02172025-025037',
    'architecture/lr:1.00e-04_bs:64_name:multistage_mode:stacked_size:large_target:both_mode:distance_trans:global_02172025-025111',
    'architecture/lr:1.00e-04_bs:64_name:singlestage_mode:stacked_size:large_target:both_mode:distance_trans:global_02172025-025132',
]

tabletoken = [
    'tabletoken/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:both_mode:distance_trans:global_02172025-025238',
    'tabletoken/lr:1.00e-04_bs:64_name:connectstage_mode:dynamic_size:large_target:both_mode:distance_trans:global_02172025-025314',
    'tabletoken/lr:1.00e-04_bs:64_name:connectstage_mode:free_size:large_target:both_mode:distance_trans:global_02172025-025340',
]

augmentations = [
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b4s0r0_target:both_mode:distance_trans:global_02192025-041619',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b0s5r0_target:both_mode:distance_trans:global_02192025-041659',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b0s0r2_target:both_mode:distance_trans:global_02192025-041758',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b4s0r2_target:both_mode:distance_trans:global_02192025-041856',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b4s5r0_target:both_mode:distance_trans:global_02192025-041951',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b0s5r2_target:both_mode:distance_trans:global_02192025-042048',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b4s5r2_target:both_mode:distance_trans:global_02192025-042158',
    'augmentations/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:b0s0r0_target:both_mode:distance_trans:global_02192025-042316',
]

modelsize = [
    'modelsize/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:small_target:both_mode:distance_trans:global_02222025-085040',
    'modelsize/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:base_target:both_mode:distance_trans:global_02222025-085124',
    'modelsize/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:both_mode:distance_trans:global_02222025-213040',
    'modelsize/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:huge_target:both_mode:distance_trans:global_02232025-043114',
]

coordsystem = [
    'coordsystem/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:both_mode:distance_trans:global_02232025-124018',
    'coordsystem/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:both_mode:distance_trans:local_02222025-084625',
]

posembedding = [
    'posencoding/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:rot_target:both_mode:distance_trans:global_02232025-092111',
    'posencoding/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_exp:add_target:both_mode:distance_trans:global_02222025-084731',
]

losstarget = [
    'losstarget/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:both_mode:distance_trans:global_02222025-084951',
    'losstarget/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:rotation_mode:distance_trans:global_02222025-084855',
    'losstarget/lr:1.00e-04_bs:64_name:connectstage_mode:stacked_size:large_target:position_mode:distance_trans:global_02232025-093127',
]


def get_paths(experiment_name: str):
    if experiment_name == 'augmentations':
        return [os.path.join(name, 'model.pt') for name in augmentations]
    elif experiment_name == 'tabletoken':
        return [os.path.join(name, 'model.pt') for name in tabletoken]
    elif experiment_name == 'losstarget':
        return [os.path.join(name, 'model.pt') for name in losstarget]
    elif experiment_name == 'modelsize':
        return [os.path.join(name, 'model.pt') for name in modelsize]
    elif experiment_name == 'coordsystem':
        return [os.path.join(name, 'model.pt') for name in coordsystem]
    elif experiment_name == 'posembedding':
        return [os.path.join(name, 'model.pt') for name in posembedding]
    elif experiment_name == 'architecture':
        return [os.path.join(name, 'model.pt') for name in architecture]
    else:
        raise ValueError(f'Unknown experiment name: {experiment_name}')


def main():
    experiment_names = ['architecture', 'tabletoken', 'augmentations', 'modelsize', 'coordsystem', 'posembedding', 'losstarget']
    for experiment_name in experiment_names:
        paths = get_paths(experiment_name)
        evalute_multiple(model_paths=paths)


if __name__ == '__main__':
    main()