#!/bin/bash

# caltech_birds2011
python3 profiler.py --model_name resnet50 \
                    --num_classes 200

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_035 \
                --rho 0.367

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_040 \
                --rho 0.4

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_050 \
                --rho 0.533

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_060 \
                --rho 0.6

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_070 \
                --rho 0.7