# Towards Ball Spin and Trajectory Analysis in Table Tennis Broadcast Videos via Physically Grounded Synthetic-to-Real Transfer

This repository is the official implementation of the paper [Towards Ball Spin and Trajectory Analysis in Table Tennis Broadcast Videos via Physically Grounded Synthetic-to-Real Transfer](https://arxiv.org/abs/2504.19863). 

We are happy to announce that the paper is accepted at the CVSports workshop at CVPR 2025.

## Preparation
You can download the data and the trained models [here](https://mediastore.rz.uni-augsburg.de/get/GefuOVBcA7/). 
Extact the data.
You will obtain the folders "data_tabletennis" and "saved_models".
Then, you have to set the following variables in paths.py:
```paths
data_path = <path_to_datasets>/data_tabletennis
checkpoint_path = <path_to_models>/saved_models
logs_path = <path_to_logs>
```
Choose <path_to_logs> arbitrarily. Here, the logs, checkpoints and evaluation metrics of your runs will be saved.

## Requirements
Install the required packages with:
```setup
pip install -r requirements.txt
```

## Training
To train the model(s) in the paper, run this command:
```train
bash start_experiments.sh
```
Alternatively, you choose the individual runs from the start_experiments.sh file and run them separately.
If you only want to train the best model from the paper, run:
```train
python -m synthetictabletennis.train --folder best --gpu 0
```
The folder flag determines where the model weights and logs are saved. 
If your machine has multiple GPUs, you can choose the GPU with the --gpu flag.
Note, that your training might overwrite the downloaded models that we provide.

## Evaluation
To evaluate the provided models in the paper, run:
```eval
python eval_experiments.py
```
If you want to evaluate a model that you trained yourself, run:
```eval
python inference.py --model_path <folder>/<identifier>/model.pt
```
You will find the model_path for your run in the logs folder. 
The folder is the same as in the training script's flag. 
The identifier is more complex and even contains the date and time of the training, so look it up in the logs folder.
You can for example evaluate the training run for the best model from above as follows:
```eval
python inference.py --model_path best/<identifier>/model.pt
```
The metrics are printed to the console and are also saved in the logs folder.

## Simulation of Synthetic Data
To simulate synthetic data, run:
```synth
xvfb-run -a -s "-screen 0 1400x900x24" bash
python -m simulation.mujocosimulation --num_trajectories <num_trajectories> --folder <folder>
```
You should have many CPU cores available for this task, otherwise it will take a very long time.
In our paper, we simulated 50000 trajectories and the folder is called "data50000". 
If you want to use your own generated data in this project, you have to adjust get_data_path() in helper.py, otherwise the provided data50000 folder will be used.
Take care: If you use data50000 as foldername, the provided data50000 folder will be overwritten.

## Cite
If you find this code useful, please cite our paper:
```bibtex
    @article{kienzle2025,
      author = {Kienzle, Daniel and Sch{\"o}n, Robin and Lienhart, Rainer and Satoh, Shin'Ichi},
      title = {Towards Ball Spin and Trajectory Analysis in Table Tennis Broadcast Videos via Physically Grounded Synthetic-to-Real Transfer},
      journal = {IEEE/CVF International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
      year = {2025},
    }
```
