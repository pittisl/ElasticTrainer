#!/bin/bash

# caltech_birds2011
python3 profiler.py --model_name resnet50 \
                    --num_classes 200

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type elastic_training \
                --run_name CUB200_ElasticTrainer

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type full_training \
                --run_name CUB200_Full_training

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type traditional_tl_training \
                --run_name CUB200_Traditional_TL

python3 main.py --model_name resnet50 \
                --dataset_name caltech_birds2011 \
                --train_type bn_plus_bias_training \
                --run_name CUB200_BN+Bias

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/accuracy \
                        --single False \
                        --elastic_trainer_path CUB200_ElasticTrainer \
                        --full_training_path CUB200_Full_training \
                        --traditional_tl_path CUB200_Traditional_TL \
                        --bn_plus_bias_path CUB200_BN+Bias \
                        --figure_id 1 \
                        --figure_name Figure_15_a.pdf

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/classification_loss \
                        --single False \
                        --elastic_trainer_path CUB200_ElasticTrainer \
                        --full_training_path CUB200_Full_training \
                        --traditional_tl_path CUB200_Traditional_TL \
                        --bn_plus_bias_path CUB200_BN+Bias \
                        --figure_id 2 \
                        --figure_name Figure_15_d.pdf

# oxford_iiit_pet
python3 profiler.py --model_name resnet50 \
                    --num_classes 37

python3 main.py --model_name resnet50 \
                --dataset_name oxford_iiit_pet \
                --train_type elastic_training \
                --run_name PET37_ElasticTrainer

python3 main.py --model_name resnet50 \
                --dataset_name oxford_iiit_pet \
                --train_type full_training \
                --run_name PET37_Full_training

python3 main.py --model_name resnet50 \
                --dataset_name oxford_iiit_pet \
                --train_type traditional_tl_training \
                --run_name PET37_Traditional_TL

python3 main.py --model_name resnet50 \
                --dataset_name oxford_iiit_pet \
                --train_type bn_plus_bias_training \
                --run_name PET37_BN+Bias

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/accuracy \
                        --single False \
                        --elastic_trainer_path PET37_ElasticTrainer \
                        --full_training_path PET37_Full_training \
                        --traditional_tl_path PET37_Traditional_TL \
                        --bn_plus_bias_path PET37_BN+Bias \
                        --figure_id 3 \
                        --figure_name Figure_15_b.pdf

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/classification_loss \
                        --single False \
                        --elastic_trainer_path PET37_ElasticTrainer \
                        --full_training_path PET37_Full_training \
                        --traditional_tl_path PET37_Traditional_TL \
                        --bn_plus_bias_path PET37_BN+Bias \
                        --figure_id 4 \
                        --figure_name Figure_15_e.pdf

# stanford_dogs
python3 profiler.py --model_name resnet50 \
                    --num_classes 120

python3 main.py --model_name resnet50 \
                --dataset_name stanford_dogs \
                --train_type elastic_training \
                --run_name DOG120_ElasticTrainer

python3 main.py --model_name resnet50 \
                --dataset_name stanford_dogs \
                --train_type full_training \
                --run_name DOG120_Full_training

python3 main.py --model_name resnet50 \
                --dataset_name stanford_dogs \
                --train_type traditional_tl_training \
                --run_name DOG120_Traditional_TL

python3 main.py --model_name resnet50 \
                --dataset_name stanford_dogs \
                --train_type bn_plus_bias_training \
                --run_name DOG120_BN+Bias

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/accuracy \
                        --single False \
                        --elastic_trainer_path DOG120_ElasticTrainer \
                        --full_training_path DOG120_Full_training \
                        --traditional_tl_path DOG120_Traditional_TL \
                        --bn_plus_bias_path DOG120_BN+Bias \
                        --figure_id 5 \
                        --figure_name Figure_15_c.pdf

python3 plot_figures.py --x_tag wall_time \
                        --y_tag test/classification_loss \
                        --single False \
                        --elastic_trainer_path DOG120_ElasticTrainer \
                        --full_training_path DOG120_Full_training \
                        --traditional_tl_path DOG120_Traditional_TL \
                        --bn_plus_bias_path DOG120_BN+Bias \
                        --figure_id 6 \
                        --figure_name Figure_15_f.pdf