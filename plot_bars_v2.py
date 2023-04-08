import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from utils import my_bool


def plot_different_models(
    path_to_elastic_trainer_resnet50,
    path_to_elastic_trainer_vgg16,
    path_to_elastic_trainer_mobilenetv2,
    path_to_full_training_resnet50,
    path_to_full_training_vgg16,
    path_to_full_training_mobilenetv2,
    path_to_traditional_tl_resnet50,
    path_to_traditional_tl_vgg16,
    path_to_traditional_tl_mobilenetv2,
    path_to_bn_plus_bias_resnet50,
    path_to_bn_plus_bias_vgg16,
    path_to_bn_plus_bias_mobilenetv2,
    figure_id,
    figure_name,
):
    et_r50 = np.loadtxt(path_to_elastic_trainer_resnet50) # [time(h), accuracy(%)]
    et_v16 = np.loadtxt(path_to_elastic_trainer_vgg16)
    et_mv2 = np.loadtxt(path_to_elastic_trainer_mobilenetv2)
    ft_r50 = np.loadtxt(path_to_full_training_resnet50)
    ft_v16 = np.loadtxt(path_to_full_training_vgg16)
    ft_mv2 = np.loadtxt(path_to_full_training_mobilenetv2)
    ttl_r50 = np.loadtxt(path_to_traditional_tl_resnet50)
    ttl_v16 = np.loadtxt(path_to_traditional_tl_vgg16)
    ttl_mv2 = np.loadtxt(path_to_traditional_tl_mobilenetv2)
    bpb_r50 = np.loadtxt(path_to_bn_plus_bias_resnet50)
    bpb_v16 = np.loadtxt(path_to_bn_plus_bias_vgg16)
    bpb_mv2 = np.loadtxt(path_to_bn_plus_bias_mobilenetv2)
    
    X = ['ResNet50','VGG16','MobileNetV2']
    et = [et_r50[1], et_v16[1], et_mv2[1]]
    ft = [ft_r50[1], ft_v16[1], ft_mv2[1]]
    ttl = [ttl_r50[1], ttl_v16[1], ttl_mv2[1]]
    bpb = [bpb_r50[1], bpb_v16[1], bpb_mv2[1]]
    
    plt.figure(figure_id)
    
    def subcategorybar1(X, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge")   
        plt.xticks(_X, X)
        plt.ylabel('Accuracy (%)', fontdict={'family': 'Arial',
                                            'color':  'black',
                                            'weight': 'bold',
                                            'size': 16,
                                            })
        plt.legend(['ElasticTrainer', 'Full training', 'Traditional TL', 'BN+Bias'])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    subcategorybar1(X, [et, ft, ttl, bpb])
    # plt.show()
    plt.savefig(figure_name + '_accuracy.pdf', format="pdf", bbox_inches="tight")
    
    #########
    
    et = [et_r50[0], et_v16[0], et_mv2[0]]
    ft = [ft_r50[0], ft_v16[0], ft_mv2[0]]
    ttl = [ttl_r50[0], ttl_v16[0], ttl_mv2[0]]
    bpb = [bpb_r50[0], bpb_v16[0], bpb_mv2[0]]
    
    plt.figure(figure_id)
    
    def subcategorybar2(X, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge")   
        plt.xticks(_X, X)
        plt.ylabel('Wall-clock time (h)', fontdict={'family': 'Arial',
                                            'color':  'black',
                                            'weight': 'bold',
                                            'size': 16,
                                            })
        plt.legend(['ElasticTrainer', 'Full training', 'Traditional TL', 'BN+Bias'])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    subcategorybar2(X, [et, ft, ttl, bpb])
    # plt.show()
    plt.savefig(figure_name + '_time.pdf', format="pdf", bbox_inches="tight")


def plot_different_models_ego(
    path_to_elastic_trainer_resnet50,
    path_to_elastic_trainer_vgg16,
    path_to_elastic_trainer_mobilenetv2,
    figure_id,
    figure_name,
):
    et_r50 = np.loadtxt(path_to_elastic_trainer_resnet50) # [time(h), accuracy(%)]
    et_v16 = np.loadtxt(path_to_elastic_trainer_vgg16)
    et_mv2 = np.loadtxt(path_to_elastic_trainer_mobilenetv2)
    
    X = ['ResNet50','VGG16','MobileNetV2']
    et = [et_r50[1], et_v16[1], et_mv2[1]]
    
    plt.figure(figure_id)
    
    def subcategorybar1(X, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge")   
        plt.xticks(_X, X)
        plt.ylabel('Accuracy (%)', fontdict={'family': 'Arial',
                                            'color':  'black',
                                            'weight': 'bold',
                                            'size': 16,
                                            })
        # plt.legend(['ElasticTrainer', 'Full training', 'Traditional TL', 'BN+Bias'])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    subcategorybar1(X, [et])
    # plt.show()
    plt.savefig(figure_name + '_accuracy.pdf', format="pdf", bbox_inches="tight")
    
    ##########
    et = [et_r50[0], et_v16[0], et_mv2[0]]
    
    plt.figure(figure_id + 1)
    
    def subcategorybar2(X, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge")   
        plt.xticks(_X, X)
        plt.ylabel('Wall-clock time (h)', fontdict={'family': 'Arial',
                                                    'color':  'black',
                                                    'weight': 'bold',
                                                    'size': 16,
                                                    })
        plt.legend(['ElasticTrainer', 'Full training', 'Traditional TL', 'BN+Bias'])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    subcategorybar2(X, [et])
    # plt.show()
    plt.savefig(figure_name + '_time.pdf', format="pdf", bbox_inches="tight")
    
    
def main():
    parser = argparse.ArgumentParser(description='Plot experiment results as bars')
    parser.add_argument('--path_to_elastic_trainer_resnet50', type=str, default='TBD')
    parser.add_argument('--path_to_elastic_trainer_vgg16', type=str, default='TBD')
    parser.add_argument('--path_to_elastic_trainer_mobilenetv2', type=str, default='TBD')
    
    parser.add_argument('--path_to_full_training_resnet50', type=str, default='TBD')
    parser.add_argument('--path_to_full_training_vgg16', type=str, default='TBD')
    parser.add_argument('--path_to_full_training_mobilenetv2', type=str, default='TBD')
    
    parser.add_argument('--path_to_traditional_tl_resnet50', type=str, default='TBD')
    parser.add_argument('--path_to_traditional_tl_vgg16', type=str, default='TBD')
    parser.add_argument('--path_to_traditional_tl_mobilenetv2', type=str, default='TBD')
    
    parser.add_argument('--path_to_bn_plus_bias_resnet50', type=str, default='TBD')
    parser.add_argument('--path_to_bn_plus_bias_vgg16', type=str, default='TBD')
    parser.add_argument('--path_to_bn_plus_bias_mobilenetv2', type=str, default='TBD')
    parser.add_argument('--figure_id', type=int, default=1, help='figure id')
    parser.add_argument('--figure_name', type=str, default='TBD', help='figure name')
    parser.add_argument('--ego', type=my_bool, default=False, help='Whether to exclude baseline schemes')
    
    args = parser.parse_args()
    
    path_to_elastic_trainer_resnet50 = args.path_to_elastic_trainer_resnet50
    path_to_elastic_trainer_vgg16 = args.path_to_elastic_trainer_vgg16
    path_to_elastic_trainer_mobilenetv2 = args.path_to_elastic_trainer_mobilenetv2
    
    path_to_full_training_resnet50 = args.path_to_full_training_resnet50
    path_to_full_training_vgg16 = args.path_to_full_training_vgg16
    path_to_full_training_mobilenetv2 = args.path_to_full_training_mobilenetv2
    
    path_to_traditional_tl_resnet50 = args.path_to_traditional_tl_resnet50
    path_to_traditional_tl_vgg16 = args.path_to_traditional_tl_vgg16
    path_to_traditional_tl_mobilenetv2 = args.path_to_traditional_tl_mobilenetv2
    
    path_to_bn_plus_bias_resnet50 = args.path_to_bn_plus_bias_resnet50
    path_to_bn_plus_bias_vgg16 = args.path_to_bn_plus_bias_vgg16
    path_to_bn_plus_bias_mobilenetv2 = args.path_to_bn_plus_bias_mobilenetv2
    
    figure_id = args.figure_id
    figure_name = args.figure_name
    ego = args.ego
    
    if ego:
        plot_different_models_ego(
            'logs/' + path_to_elastic_trainer_resnet50 + '.txt',
            'logs/' + path_to_elastic_trainer_vgg16 + '.txt',
            'logs/' + path_to_elastic_trainer_mobilenetv2 + '.txt',
            figure_id,
            'figures/' + figure_name,
        )
    else:
        plot_different_models(
            'logs/' + path_to_elastic_trainer_resnet50 + '.txt',
            'logs/' + path_to_elastic_trainer_vgg16 + '.txt',
            'logs/' + path_to_elastic_trainer_mobilenetv2 + '.txt',
            'logs/' + path_to_full_training_resnet50 + '.txt',
            'logs/' + path_to_full_training_vgg16 + '.txt',
            'logs/' + path_to_full_training_mobilenetv2 + '.txt',
            'logs/' + path_to_traditional_tl_resnet50 + '.txt',
            'logs/' + path_to_traditional_tl_vgg16 + '.txt',
            'logs/' + path_to_traditional_tl_mobilenetv2 + '.txt',
            'logs/' + path_to_bn_plus_bias_resnet50 + '.txt',
            'logs/' + path_to_bn_plus_bias_vgg16 + '.txt',
            'logs/' + path_to_bn_plus_bias_mobilenetv2 + '.txt',
            figure_id,
            'figures/' + figure_name,
        )

if __name__ == '__main__':
    main()