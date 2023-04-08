#!/bin/bash

# caltech_birds2011 resnet50
python3 profiler.py --model_name resnet50 \
                    --num_classes 200

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name resnet50_ElasticTrainer \
                --rho 0.4 \
                --save_txt True

# caltech_birds2011 vgg16
python3 profiler.py --model_name vgg16 \
                    --num_classes 200

python3 main.py --model_name vgg16 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name vgg16_ElasticTrainer \
                --rho 0.4 \
                --save_txt True

# caltech_birds2011 mobilenetv2
python3 profiler.py --model_name mobilenetv2 \
                    --num_classes 200

python3 main.py --model_name mobilenetv2 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name mobilenetv2_ElasticTrainer \
                --rho 0.4 \
                --save_txt True

python3 plot_bars_v2.py --path_to_elastic_trainer_resnet50 resnet50_ElasticTrainer \
                        --path_to_elastic_trainer_vgg16 vgg16_ElasticTrainer \
                        --path_to_elastic_trainer_mobilenetv2 mobilenetv2_ElasticTrainer \
                        --figure_id 1 \
                        --figure_name Figure_19_ego \
                        --ego True