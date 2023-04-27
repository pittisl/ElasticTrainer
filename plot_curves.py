import tensorflow as tf
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing import tag_types
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from utils import my_bool


def read_data_from_tfboard_logs(path, x_tag, y_tag):
    """load logged training metrics into numpy arrays

    Args:
        path (str): path to the training log
        x_tag (str): one from ['wall_time', 'step']
        y_tag (str): one from ['train/accuracy', 'train/classification_loss', 
                               'train/learnig_rate', 'test/classification_loss', 
                               'test/accuracy']
    Returns:
        x, y: numpy arrays
    """
    size_guidance = {
        tag_types.TENSORS: 20,
    }
    event_acc = EventAccumulator(path, size_guidance=size_guidance)
    event_acc.Reload()
    event_list = event_acc.Tensors(y_tag)
    if x_tag == 'wall_time':
        x = [e.wall_time for e in event_list]
        x = np.array(x)
        x = x - x[0]
        x = (x + x[1]) / 3600 # convert to hours
    else:
        x = [e.step for e in event_list]
    y = np.array([tf.make_ndarray(e.tensor_proto).item() for e in event_list])
    if 'accuracy' in y_tag:
        y *= 100 # convert to %
    return x, y


def plot_single_curve(
    x_tag,
    y_tag,
    path_to_elastic_training,
    figure_id,
    figure_name,
):
    """plot ElasticTrainer results as a single curve excluding baselines

    Args:
        x_tag (str): one from ['wall_time', 'step']
        y_tag (str): one from ['train/accuracy', 'train/classification_loss', 
                               'train/learnig_rate', 'test/classification_loss', 
                               'test/accuracy']
        path_to_elastic_training (str): path to ElasticTrainer's log
        figure_id (str): id of plotted figure
    """
    et_x, et_y = read_data_from_tfboard_logs(path_to_elastic_training, x_tag, y_tag)
    font = {'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
    }
    plt.figure(figure_id)
    plt.plot(et_x, et_y, "ks-", label="ElasticTrainer",  linewidth=3)
    plt.xlabel('Wall-clock time (h)', fontdict=font)
    if 'accuracy' in y_tag:
        plt.ylabel('Accuracy (%)', fontdict=font)
    else:
        plt.ylabel('Loss', fontdict=font)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name, format="pdf", bbox_inches="tight")
    # plt.show()
    

def plot_multiple_curves(
    x_tag,
    y_tag,
    path_to_elastic_training,
    path_to_full_training,
    path_to_traditional_tl,
    path_to_bn_plus_bias,
    figure_id,
    figure_name,
):
    """plot training results as curves including baselines

    Args:
        x_tag (str): one from ['wall_time', 'step']
        y_tag (str): one from ['train/accuracy', 'train/classification_loss', 
                               'train/learnig_rate', 'test/classification_loss', 
                               'test/accuracy']
        path_to_elastic_training (str): path to ElasticTrainer's log
        path_to_full_training (str): path to Full Training's log
        path_to_traditional_tl (str): path to Traditional TL's log
        path_to_bn_plus_bias (str): path to BN+Bias's log
        figure_id (str): id of plotted figure
    """
    et_x, et_y = read_data_from_tfboard_logs(path_to_elastic_training, x_tag, y_tag)
    ft_x, ft_y = read_data_from_tfboard_logs(path_to_full_training, x_tag, y_tag)
    ttl_x, ttl_y = read_data_from_tfboard_logs(path_to_traditional_tl, x_tag, y_tag)
    bpb_x, bpb_y = read_data_from_tfboard_logs(path_to_bn_plus_bias, x_tag, y_tag)
    
    font = {'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
    }
    
    plt.figure(figure_id)
    plt.plot(et_x, et_y, "ks-", label="ElasticTrainer",  linewidth=3)
    plt.plot(ft_x, ft_y, "rs-", label="Full Training", linewidth=3)
    plt.plot(ttl_x, ttl_y, "bs-", label="Traditional TL", linewidth=3)
    plt.plot(bpb_x, bpb_y, "gs-", label="BN+Bias", linewidth=3)
    plt.xlabel('Wall-clock time (h)', fontdict=font)
    if 'accuracy' in y_tag:
        plt.ylabel('Accuracy (%)', fontdict=font)
    else:
        plt.ylabel('Loss', fontdict=font)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name, format="pdf", bbox_inches="tight")
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot experiment results as curves')
    parser.add_argument('--x_tag', type=str, default='wall_time', help="one from ['wall_time', 'step']")
    parser.add_argument('--y_tag', type=str, default='accuracy', help="['train/accuracy', 'train/classification_loss',\
        'train/learnig_rate', 'test/classification_loss', 'test/accuracy']")
    parser.add_argument('--single', type=my_bool, default=True, help='whether to exclude baseline schemes')
    parser.add_argument('--elastic_trainer_path', type=str, default='TBD', help='path to log of elastic_trainer')
    parser.add_argument('--full_training_path', type=str, default='TBD', help='path to log of full_training')
    parser.add_argument('--traditional_tl_path', type=str, default='TBD', help='path to log of elastic_trainer')
    parser.add_argument('--bn_plus_bias_path', type=str, default='TBD', help='path to log of bn_plus_bias')
    parser.add_argument('--figure_id', type=int, default=1, help='figure id')
    parser.add_argument('--figure_name', type=str, default='TBD', help='figure name')
    
    args = parser.parse_args()
    
    x_tag = args.x_tag
    y_tag = args.y_tag
    single = args.single
    elastic_trainer_path = args.elastic_trainer_path
    full_training_path = args.full_training_path
    traditional_tl_path = args.traditional_tl_path
    bn_plus_bias_path = args.bn_plus_bias_path
    figure_id = args.figure_id
    figure_name = args.figure_name
    
    if single:
        plot_single_curve(
            x_tag,
            y_tag,
            'logs/' + elastic_trainer_path,
            figure_id,
            'figures/' + figure_name,
        )
    else:
        plot_multiple_curves(
            x_tag,
            y_tag,
            'logs/' + elastic_trainer_path,
            'logs/' + full_training_path,
            'logs/' + traditional_tl_path,
            'logs/' + bn_plus_bias_path,
            figure_id,
            'figures/' + figure_name,
        )

if __name__ == '__main__':
    main()