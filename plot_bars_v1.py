import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from utils import my_bool

def plot_different_speedup_ratios(
    path_to_rho35,
    path_to_rho40,
    path_to_rho50,
    path_to_rho60,
    path_to_rho70,
    path_to_full_training,
    path_to_traditional_tl,
    path_to_bn_plus_bias,
    path_to_prunetrain,
    figure_id,
    figure_name,
):
    rho35 = np.loadtxt(path_to_rho35) # [time(h), accuracy(%)]
    rho40 = np.loadtxt(path_to_rho40)
    rho50 = np.loadtxt(path_to_rho50)
    rho60 = np.loadtxt(path_to_rho60)
    rho70 = np.loadtxt(path_to_rho70)
    ft = np.loadtxt(path_to_full_training)
    ttl = np.loadtxt(path_to_traditional_tl)
    bpb = np.loadtxt(path_to_bn_plus_bias)
    pt = np.loadtxt(path_to_prunetrain)
    
    fig = plt.figure(figure_id)

    x = ['rho=35%', 'rho=40%', 'rho=50%', 'rho=60%', 'rho=70%', 'Full training', \
        'Traditional TL', 'BN+Bias', 'PruneTrain']

    y1 = np.array([rho35[1], rho40[1], rho50[1], rho60[1], rho70[1], ft[1], ttl[1], bpb[1], pt[1]])
    y2 = np.array([rho35[0], rho40[0], rho50[0], rho60[0], rho70[0], ft[0], ttl[0], bpb[0], pt[0]])
    pad = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def subcategorybar(X, vals, color, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge", color=color)   
        plt.xticks(_X, X)
        plt.xticks(rotation=45, ha='right')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    subcategorybar(x, [y1, pad], [0, 0.4470, 0.7410])
    plt.ylabel('Accuracy (%)', fontdict={'family': 'Arial',
                                        'color':  [0, 0.4470, 0.7410],
                                        'weight': 'bold',
                                        'size': 16,
                                        })

    fig.axes[1] = fig.axes[0].twinx()

    subcategorybar(x, [pad, y2], [0.8500, 0.3250, 0.0980])
    plt.ylabel('Wall-clock time (h)', fontdict={'family': 'Arial',
                                        'color':  [0.8500, 0.3250, 0.0980],
                                        'weight': 'bold',
                                        'size': 16,
                                        })

    fig.axes[0].tick_params(axis='y', colors=[0, 0.4470, 0.7410])
    fig.axes[0].spines['left'].set_color([0, 0.4470, 0.7410])
    fig.axes[1].spines['left'].set_color([0, 0.4470, 0.7410])

    fig.axes[1].tick_params(axis='y', colors=[0.8500, 0.3250, 0.0980])
    fig.axes[1].spines['right'].set_color([0.8500, 0.3250, 0.0980])
    fig.axes[0].spines['right'].set_color([0.8500, 0.3250, 0.0980])

    plt.tight_layout()
    # plt.show()
    plt.savefig(figure_name, format="pdf", bbox_inches="tight")
    

def plot_different_speedup_ratios_ego(
    path_to_rho35,
    path_to_rho40,
    path_to_rho50,
    path_to_rho60,
    path_to_rho70,
    figure_id,
    figure_name,
):
    rho35 = np.loadtxt(path_to_rho35) # [time(h), accuracy(%)]
    rho40 = np.loadtxt(path_to_rho40)
    rho50 = np.loadtxt(path_to_rho50)
    rho60 = np.loadtxt(path_to_rho60)
    rho70 = np.loadtxt(path_to_rho70)
    
    fig = plt.figure(figure_id)

    x = ['rho=35%', 'rho=40%', 'rho=50%', 'rho=60%', 'rho=70%']

    y1 = np.array([rho35[1], rho40[1], rho50[1], rho60[1], rho70[1]])
    y2 = np.array([rho35[0], rho40[0], rho50[0], rho60[0], rho70[0]])
    pad = [0, 0, 0, 0, 0]

    def subcategorybar(X, vals, color, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge", color=color)   
        plt.xticks(_X, X)
        plt.xticks(rotation=45, ha='right')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    subcategorybar(x, [y1, pad], [0, 0.4470, 0.7410])
    plt.ylabel('Accuracy (%)', fontdict={'family': 'Arial',
                                        'color':  [0, 0.4470, 0.7410],
                                        'weight': 'bold',
                                        'size': 16,
                                        })

    fig.axes[1] = fig.axes[0].twinx()

    subcategorybar(x, [pad, y2], [0.8500, 0.3250, 0.0980])
    plt.ylabel('Wall-clock time (h)', fontdict={'family': 'Arial',
                                        'color':  [0.8500, 0.3250, 0.0980],
                                        'weight': 'bold',
                                        'size': 16,
                                        })

    fig.axes[0].tick_params(axis='y', colors=[0, 0.4470, 0.7410])
    fig.axes[0].spines['left'].set_color([0, 0.4470, 0.7410])
    fig.axes[1].spines['left'].set_color([0, 0.4470, 0.7410])

    fig.axes[1].tick_params(axis='y', colors=[0.8500, 0.3250, 0.0980])
    fig.axes[1].spines['right'].set_color([0.8500, 0.3250, 0.0980])
    fig.axes[0].spines['right'].set_color([0.8500, 0.3250, 0.0980])

    plt.tight_layout()
    # plt.show()
    plt.savefig(figure_name, format="pdf", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results as bars')
    parser.add_argument('--path_to_rho35', type=str, default='TBD')
    parser.add_argument('--path_to_rho40', type=str, default='TBD')
    parser.add_argument('--path_to_rho50', type=str, default='TBD')
    parser.add_argument('--path_to_rho60', type=str, default='TBD')
    parser.add_argument('--path_to_rho70', type=str, default='TBD')
    parser.add_argument('--path_to_full_training', type=str, default='TBD')
    parser.add_argument('--path_to_traditional_tl', type=str, default='TBD')
    parser.add_argument('--path_to_bn_plus_bias', type=str, default='TBD')
    parser.add_argument('--path_to_prunetrain', type=str, default='TBD')
    parser.add_argument('--figure_id', type=int, default=1, help='figure id')
    parser.add_argument('--figure_name', type=str, default='TBD', help='figure name')
    parser.add_argument('--ego', type=my_bool, default=False, help='Whether to exclude baseline schemes')
    
    args = parser.parse_args()
    
    path_to_rho35 = args.path_to_rho35
    path_to_rho40 = args.path_to_rho40
    path_to_rho50 = args.path_to_rho50
    path_to_rho60 = args.path_to_rho60
    path_to_rho70 = args.path_to_rho70
    path_to_full_training = args.path_to_full_training
    path_to_traditional_tl = args.path_to_traditional_tl
    path_to_bn_plus_bias = args.path_to_bn_plus_bias
    path_to_prunetrain = args.path_to_prunetrain
    figure_id = args.figure_id
    figure_name = args.figure_name
    ego = args.ego
    
    if ego:
        plot_different_speedup_ratios_ego(
            'logs/' + path_to_rho35 + '.txt',
            'logs/' + path_to_rho40 + '.txt',
            'logs/' + path_to_rho50 + '.txt',
            'logs/' + path_to_rho60 + '.txt',
            'logs/' + path_to_rho70 + '.txt',
            figure_id,
            'figures/' + figure_name,
        )
    else:
        plot_different_speedup_ratios(
            'logs/' + path_to_rho35 + '.txt',
            'logs/' + path_to_rho40 + '.txt',
            'logs/' + path_to_rho50 + '.txt',
            'logs/' + path_to_rho60 + '.txt',
            'logs/' + path_to_rho70 + '.txt',
            'logs/' + path_to_full_training + '.txt',
            'logs/' + path_to_traditional_tl + '.txt',
            'logs/' + path_to_bn_plus_bias + '.txt',
            'logs/' + path_to_prunetrain + '.txt',
            figure_id,
            'figures/' + figure_name,
        )

if __name__ == '__main__':
    main()

# fig = plt.figure(1)

# x = ['rho=35%', 'rho=40%', 'rho=50%', 'rho=60%', 'rho=70%', 'Full training', \
#     'Traditional TL', 'BN+Bias', 'PruneTrain']

# y1 = [10, 20, 30, 20, 20, 20, 20, 20, 20]
# y2 = [40, 50, 20, 30, 30, 30, 30, 30, 30]
# pad = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# def subcategorybar(X, vals, color, width=0.8):
#     n = len(vals)
#     _X = np.arange(len(X))
#     for i in range(n):
#         plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
#                 width=width/float(n), align="edge", color=color)   
#     plt.xticks(_X, X)
#     plt.xticks(rotation=45, ha='right')
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
    
# subcategorybar(x, [y1, pad], [0, 0.4470, 0.7410])
# plt.ylabel('Accuracy (%)', fontdict={'family': 'Arial',
#                                      'color':  [0, 0.4470, 0.7410],
#                                      'weight': 'bold',
#                                      'size': 16,
#                                      })

# fig.axes[1] = fig.axes[0].twinx()

# subcategorybar(x, [pad, y2], [0.8500, 0.3250, 0.0980])
# plt.ylabel('Wall-clock time (h)', fontdict={'family': 'Arial',
#                                      'color':  [0.8500, 0.3250, 0.0980],
#                                      'weight': 'bold',
#                                      'size': 16,
#                                      })

# fig.axes[0].tick_params(axis='y', colors=[0, 0.4470, 0.7410])
# fig.axes[0].spines['left'].set_color([0, 0.4470, 0.7410])
# fig.axes[1].spines['left'].set_color([0, 0.4470, 0.7410])

# fig.axes[1].tick_params(axis='y', colors=[0.8500, 0.3250, 0.0980])
# fig.axes[1].spines['right'].set_color([0.8500, 0.3250, 0.0980])
# fig.axes[0].spines['right'].set_color([0.8500, 0.3250, 0.0980])

# plt.tight_layout()
# plt.show()

