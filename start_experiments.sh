#!/bin/bash

# Architecture
python -m synthetictabletennis.train --lr 1e-4 --folder architecture --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder architecture --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --model_name multistage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder architecture --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --model_name singlestage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;

# Tabletoken
python -m synthetictabletennis.train --lr 1e-4 --folder tabletoken --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder tabletoken --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --model_name connectstage --model_size large --token_mode dynamic --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder tabletoken --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --model_name connectstage --model_size large --token_mode free --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;

# Augmentations
python -m synthetictabletennis.train --lr 1e-4 --folder augmentations --blur_strength 0.0 --stop_prob 0.0 --randomize_std 0 --exp_id b0s0r0 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder augmentations --blur_strength 0.4 --stop_prob 0.0 --randomize_std 0 --exp_id b4s0r0 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder augmentations --blur_strength 0.0 --stop_prob 0.5 --randomize_std 0 --exp_id b0s5r0 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder augmentations --blur_strength 0.0 --stop_prob 0.0 --randomize_std 2 --exp_id b0s0r2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder augmentations --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --exp_id b4s5r2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;

# Coordinate System
python -m synthetictabletennis.train --lr 1e-4 --folder coordsystem --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder coordsystem --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode local --pos_embedding rotary --gpu 0 ;

# Positional Encoding
python -m synthetictabletennis.train --lr 1e-4 --folder posencoding --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --exp_id rot --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder posencoding --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --exp_id add --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding added --gpu 0 ;

# Loss Target
python -m synthetictabletennis.train --lr 1e-4 --folder losstarget --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder losstarget --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target rotation --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder losstarget --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target position --transform_mode global --pos_embedding rotary --gpu 0 ;

# Model Size
python -m synthetictabletennis.train --lr 1e-4 --folder modelsize --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size large --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder modelsize --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size small --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder modelsize --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size base --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;
python -m synthetictabletennis.train --lr 1e-4 --folder modelsize --blur_strength 0.4 --stop_prob 0.5 --randomize_std 2 --model_name connectstage --model_size huge --token_mode stacked --loss_mode distance --loss_target both --transform_mode global --pos_embedding rotary --gpu 0 ;


