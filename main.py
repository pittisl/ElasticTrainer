from utils import (port_datasets, 
                   port_pretrained_models, 
                   RepeatTimer, record_once, 
                   sig_stop_handler)
from train import (full_training, 
                   traditional_tl_training, 
                   bn_plus_bias_training, 
                   elastic_training,
                   elastic_training_weight_magnitude,
                   elastic_training_grad_magnitude)
import argparse
import signal

# import logging
# logging.getLogger('tensorflow').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Training a NN model with selected schemes')
parser.add_argument('--model_name', type=str, default='resnet50', help='valid model names are resnet50, vgg16, mobilenetv2, vit')
parser.add_argument('--dataset_name', type=str, default='caltech_birds2011', help='valid dataset names are caltech_birds2011, stanford_dogs, oxford_iiit_pet')
parser.add_argument('--train_type', type=str, default='elastic_training', help='valid training schemes are full_training, traditional_tl_training,\
                    bn_plus_bias_training, elastic_training')
parser.add_argument('--input_size', type=int, default=224, help='input resolution, e.g., 224 stands for 224x224')
parser.add_argument('--batch_size', type=int, default=4, help='batch size used to run during profiling')
parser.add_argument('--num_classes', type=int, default=200, help='number of categories model can classify')
parser.add_argument('--optimizer', type=str, default='sgd', help='valid optimizers are sgd and adam, adam is recommended for vit')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for sgd')
parser.add_argument('--num_epochs', type=int, default=12, help='number of training epochs')

parser.add_argument('--interval', type=float, default=4, help='interval (in epoch) of tensor importance evaluation')
parser.add_argument('--rho', type=float, default=0.533, help='speedup ratio')

args = parser.parse_args()

model_name = args.model_name
dataset_name = args.dataset_name
train_type = args.train_type
input_size = args.input_size
batch_size = args.batch_size
num_classes = args.num_classes
optimizer = args.optimizer
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_epochs = args.num_epochs
interval = args.interval
rho = args.rho

run_name = model_name + '_' + dataset_name + '_' + train_type
logdir = 'logs'
timing_info = model_name + '_' + str(input_size) + '_' + str(num_classes) + '_' + str(batch_size) + '_' + 'profile'

global timer
timer = RepeatTimer(15, record_once)
timer.start()

signal.signal(signal.SIGINT, sig_stop_handler)
signal.signal(signal.SIGTERM, sig_stop_handler)

print('### Porting NN model...')

model = port_pretrained_models(
    model_type=model_name,
    input_shape=(input_size, input_size, 3),
    num_classes=num_classes,
)

print('### Porting dataset...')

train_dataset, test_dataset = port_datasets(
    dataset_name=dataset_name,
    input_shape=(input_size, input_size, 3),
    batch_size=batch_size,
)

print('### Start training...')

if train_type == 'full_training':
    full_training(
        model,
        train_dataset,
        test_dataset,
        run_name,
        logdir,
        optim=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=num_epochs,
    )
elif train_type == 'traditional_tl_training':
    traditional_tl_training(
        model,
        train_dataset,
        test_dataset,
        run_name,
        logdir,
        optim=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=num_epochs,
    )
elif train_type == 'bn_plus_bias_training':
    bn_plus_bias_training(
        model,
        train_dataset,
        test_dataset,
        run_name,
        logdir,
        optim=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=num_epochs,
    )
elif train_type == 'elastic_training':
    elastic_training(
        model,
        model_name,
        train_dataset,
        test_dataset,
        run_name,
        logdir,
        timing_info,
        optim=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=num_epochs,
        interval=interval,
        rho=rho,
    )

elif train_type == 'elastic_training_weight_magnitude':
    elastic_training_weight_magnitude(
        model,
        model_name,
        train_dataset,
        test_dataset,
        run_name,
        logdir,
        timing_info,
        optim=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=num_epochs,
        interval=interval,
        rho=rho,
    )

elif train_type == 'elastic_training_grad_magnitude':
    elastic_training_grad_magnitude(
        model,
        model_name,
        train_dataset,
        test_dataset,
        run_name,
        logdir,
        timing_info,
        optim=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=num_epochs,
        interval=interval,
        rho=rho,
    )

else:
    raise NotImplementedError(f"Training scheme {train_type} has not been implemented yet")

timer.cancel()
