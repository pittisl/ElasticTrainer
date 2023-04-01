#!/bin/bash

# caltech_birds2011
python3 profiler.py --model_name resnet50 \
                    --num_classes 200

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/accuracy \
                        --single True \
                        --elastic_trainer_path CUB200_ElasticTrainer \
                        --figure_id 1

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/classification_loss \
                        --single True \
                        --elastic_trainer_path CUB200_ElasticTrainer \
                        --figure_id 2

# oxford_iiit_pet
python3 profiler.py --model_name resnet50 \
                    --num_classes 37

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/accuracy \
                        --single True \
                        --elastic_trainer_path PET37_ElasticTrainer \
                        --figure_id 3

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/classification_loss \
                        --single True \
                        --elastic_trainer_path PET37_ElasticTrainer \
                        --figure_id 4

# stanford_dogs
python3 profiler.py --model_name resnet50 \
                    --num_classes 120

python3 main.py --model_name resnet50 \
                --dataset_name stanford_dogs \
                --train_type elastic_training \
                --run_name DOG120_ElasticTrainer

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/accuracy \
                        --single True \
                        --elastic_trainer_path DOG120_ElasticTrainer \
                        --figure_id 5

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/classification_loss \
                        --single True \
                        --elastic_trainer_path DOG120_ElasticTrainer \
                        --figure_id 6