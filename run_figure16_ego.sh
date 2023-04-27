#!/bin/bash

# caltech_birds2011
python3 profiler.py --model_name resnet50 \
                    --num_classes 200

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer

python3 plot_curves.py --x_tag wall_time \
                       --y_tag test/accuracy \
                       --single True \
                       --elastic_trainer_path CUB200_ElasticTrainer \
                       --figure_id 1 \
                       --figure_name Figure_16_a_ego.pdf

python3 plot_curves.py --x_tag wall_time \
                       --y_tag test/classification_loss \
                       --single True \
                       --elastic_trainer_path CUB200_ElasticTrainer \
                       --figure_id 2 \
                       --figure_name Figure_16_c_ego.pdf

# oxford_iiit_pet
python3 profiler.py --model_name resnet50 \
                    --num_classes 37

python3 main.py --model_name resnet50 \
                --dataset_name oxford_iiit_pet \
                --train_type elastic_training \
                --run_name PET37_ElasticTrainer

python3 plot_curves.py --x_tag wall_time \
                       --y_tag test/accuracy \
                       --single True \
                       --elastic_trainer_path PET37_ElasticTrainer \
                       --figure_id 3 \
                       --figure_name Figure_16_b_ego.pdf

python3 plot_curves.py --x_tag wall_time \
                       --y_tag test/classification_loss \
                       --single True \
                       --elastic_trainer_path PET37_ElasticTrainer \
                       --figure_id 4 \
                       --figure_name Figure_16_d_ego.pdf