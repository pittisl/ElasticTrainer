#!/bin/bash

# caltech_birds2011
python3 profiler.py --model_name resnet50 \
                    --num_classes 200

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_035 \
                --rho 0.367 \
                --save_txt True

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_040 \
                --rho 0.4 \
                --save_txt True

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_050 \
                --rho 0.533 \
                --save_txt True

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_060 \
                --rho 0.6 \
                --save_txt True

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer_070 \
                --rho 0.7 \
                --save_txt True

python3 plot_bars_v1.py --path_to_rho35 CUB200_ElasticTrainer_035 \
                        --path_to_rho40 CUB200_ElasticTrainer_040 \
                        --path_to_rho50 CUB200_ElasticTrainer_050 \
                        --path_to_rho60 CUB200_ElasticTrainer_060 \
                        --path_to_rho70 CUB200_ElasticTrainer_070 \
                        --figure_id 1 \
                        --figure_name Figure_17_ac_ego.pdf \
                        --ego True