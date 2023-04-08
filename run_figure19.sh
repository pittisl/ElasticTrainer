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

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type full_training \
                --run_name resnet50_Full_training \
                --save_txt True

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type traditional_tl_training \
                --run_name resnet50_Traditional_TL \
                --save_txt True

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type bn_plus_bias_training \
                --run_name resnet50_BN+Bias \
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

python3 main.py --model_name vgg16 \
                --dataset_name caltech_birds2011 \
                --train_type full_training \
                --run_name vgg16_Full_training \
                --save_txt True

python3 main.py --model_name vgg16 \
                --dataset_name caltech_birds2011 \
                --train_type traditional_tl_training \
                --run_name vgg16_Traditional_TL \
                --save_txt True

python3 main.py --model_name vgg16 \
                --dataset_name caltech_birds2011 \
                --train_type bn_plus_bias_training \
                --run_name vgg16_BN+Bias \
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

python3 main.py --model_name mobilenetv2 \
                --dataset_name caltech_birds2011 \
                --train_type full_training \
                --run_name mobilenetv2_Full_training \
                --num_epochs 24 \
                --save_txt True

python3 main.py --model_name mobilenetv2 \
                --dataset_name caltech_birds2011 \
                --train_type traditional_tl_training \
                --run_name mobilenetv2_Traditional_TL \
                --save_txt True

python3 main.py --model_name mobilenetv2 \
                --dataset_name caltech_birds2011 \
                --train_type bn_plus_bias_training \
                --run_name mobilenetv2_BN+Bias \
                --save_txt True

python3 plot_bars_v2.py --path_to_elastic_trainer_resnet50 resnet50_ElasticTrainer \
                                 --path_to_elastic_trainer_vgg16 vgg16_ElasticTrainer \
                                 --path_to_elastic_trainer_mobilenetv2 mobilenetv2_ElasticTrainer \
                                 --path_to_full_training_resnet50 resnet50_Full_training \
                                 --path_to_full_training_vgg16 vgg16_Full_training \
                                 --path_to_full_training_mobilenetv2 mobilenetv2_Full_training \
                                 --path_to_traditional_tl_resnet50 resnet50_Traditional_TL \
                                 --path_to_traditional_tl_vgg16 vgg16_Traditional_TL \
                                 --path_to_traditional_tl_mobilenetv2 mobilenetv2_Traditional_TL \
                                 --path_to_bn_plus_bias_resnet50 resnet50_BN+Bias \
                                 --path_to_bn_plus_bias_vgg16 vgg16_BN+Bias \
                                 --path_to_bn_plus_bias_mobilenetv2 mobilenetv2_BN+Bias \
                                 --figure_id 1 \
                                 --figure_name Figure_19 \
                                 --ego False
