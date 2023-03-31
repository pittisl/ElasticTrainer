from utils import port_pretrained_models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorboard_plugin_profile.protobuf import tf_stats_pb2 #, kernel_stats_pb2
from tensorboard_plugin_profile.convert.tf_stats_proto_to_gviz import generate_chart_table
# from tensorboard_plugin_profile.convert.kernel_stats_proto_to_gviz import generate_kernel_reports_table
from tqdm import tqdm
import argparse
import time
import csv
import os


def profile_backpropagation(
    model,
    input_shape,
    batch_size,
    num_iterations,
    logdir):
    """
    This function profiles NN ops in backward pass.

    Args:
        model (tf.keras.Model): NN model to profile
        input_shape (tuple): input shape of NN model
        batch_size (int): batch size for backward pass
        num_iterations (int): number of backward passes to run
        logdir (str): path to where profile is recorded
    """
    
    if os.path.exists(logdir):
        print(f"Profile '{logdir}' already exists")
        return
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        return gradients
    
    # dummy training data
    x = tf.random.normal((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    y = tf.ones((batch_size,))

    print("Warmup...")
    for k in tqdm(range(2)):
        train_step(x, y)
    
    t0 = time.time()

    print("Profiling the model...")
    tf.profiler.experimental.start(logdir)
    for k in range(num_iterations):
        with tf.profiler.experimental.Trace('train', step_num=k, _r=1):
            train_step(x, y)
    tf.profiler.experimental.stop()

    t1 = time.time()
    
    print("Finished profiling!")
    print("Elasped time (s):", t1 - t0)


def convert_pb_to_csv(logdir, outdir):
    """
    This function extracts timing-related info from the recorded profile.

    Args:
        logdir (str): path to where profile is recorded
        outdir (str): path to where the extracted timing info is stored
    """
    if os.path.exists(outdir):
        print(f"Extracted timing info '{outdir}' already exists")
        return
    
    tf_profile_path = logdir + '/plugins/profile/'
    tf_stats_path = ''
    for root, subdirs, files in os.walk(logdir):
        if tf_profile_path in root:
            fn = ''
            for f in files:
                if 'tensorflow_stats.pb' in f:
                    fn = f
            tf_stats_path = root + '/' + fn
    
    with tf.io.gfile.GFile(tf_stats_path, 'rb') as f:
        tf_stats_db = tf_stats_pb2.TfStatsDatabase()
        tf_stats_db.ParseFromString(f.read())
    
    csv_table = generate_chart_table(tf_stats_db.with_idle,
                                     tf_stats_db.device_type).ToCsv()
    with open(outdir, 'w') as f:
        f.write(csv_table)


def profile_parser(
    model,
    model_type,
    num_iterations, 
    filedir, 
    draw_figure=False):
    """
    This function constructs tensor timings from the profiled op timings.

    Args:
        model (tf.keras.Model): NN model
        model_type (str): type of NN model
        num_iterations (int): number of iterations of backward passes in profiling
        filedir (str): where the timing-related info is stored
        draw_figure (bool, optional): whether to plot the tensor timings. Defaults to False.

    Returns:
        tensor timings t_dw and t_dy
    """
    
    if model_type in ('resnet50', 'vgg16'):
        
        all_stats = []
        with open(filedir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_reader.__next__()
            for r in csv_reader:
                all_stats.append(r)
        
        op_total_time = []
        op_names = []
        # extract gradient related ops
        for op_stat in all_stats:
            if 'gradient_tape' in op_stat[3]:
                op_total_time.append(float(op_stat[5]))
                op_names.append(op_stat[3])
        
        base_layers = model.layers[4].layers
        custom_layers = model.layers[5:]
        model_layers = [*base_layers, *custom_layers]

        t_dw = [0.0 for k in range(len(model.trainable_weights))]
        t_dy = [0.0 for k in range(len(model.trainable_weights))]
        
        weight_count = 0

        for l in model_layers:
            if '_conv' in l.name:
                if l.use_bias:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'Conv2DBackpropFilter' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'Conv2DBackpropInput' in op:
                            t_dy[weight_count] += t # include TransposeNCHWToNHWC
                        elif l.name in op and 'BiasAddGrad' in op:
                            t_dw[weight_count + 1] = t
                            t_dy[weight_count + 1] = 0
                    weight_count += 2
                else:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'Conv2DBackpropFilter' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'Conv2DBackpropInput' in op:
                            t_dy[weight_count] += t # include TransposeNCHWToNHWC
                    weight_count += 1

            elif '_bn' in l.name:
                for op, t in zip(op_names, op_total_time):
                    if l.name in op and 'FusedBatchNormGrad' in op:
                        # for gamma
                        t_dw[weight_count] = 0
                        t_dy[weight_count] = 0
                        # for beta
                        t_dw[weight_count + 1] = 0
                        t_dy[weight_count + 1] = t
                weight_count += 2
            elif 'dense' in l.name:
                if l.use_bias:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'MatMul/MatMul_1' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'MatMul/MatMul' in op:
                            t_dy[weight_count] = t 
                        elif l.name in op and 'BiasAddGrad' in op:
                            t_dw[weight_count + 1] = t
                            t_dy[weight_count + 1] = 0
                    weight_count += 2
                else:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'MatMul/MatMul_1' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'MatMul/MatMul' in op:
                            t_dy[weight_count] = t 
                    weight_count += 1

            else:
                # fuse backprop time of non-trainables to the previous trainable layer
                for op, t in zip(op_names, op_total_time):
                    if l.name in op and weight_count > 0:
                        t_dy[weight_count - 1] += t
        
        # the first layer never propagates input grads, just remove t_dy[0]
        # t_dy = t_dy[1:] # 1~N-1

        t_dw = np.array(t_dw) / num_iterations # (us)
        t_dy = np.array(t_dy) / num_iterations # (us)
        
        print(f'# model trainbles: {len(model.trainable_weights)}')
        print(f'# t_dw: {weight_count}, # t_dy: {weight_count}')
        
        if draw_figure:
            fig = plt.figure()
            plt.barh(np.arange(t_dw.shape[0]), t_dw, color ='navy')
            #plt.xticks(rotation=45)
            plt.xlabel('t_dw (us)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Layer ID', fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
            
            fig = plt.figure()
            plt.barh(np.arange(t_dy.shape[0]), t_dy, color ='navy')
            #plt.xticks(rotation=45)
            plt.xlabel('t_dy (us)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Layer ID', fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
        
        return tf.convert_to_tensor(t_dw/1000.0, tf.float32), tf.convert_to_tensor(t_dy/1000.0, tf.float32) # (ms) 
    
    elif model_type == 'mobilenetv2':
        
        all_stats = []
        with open(filedir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_reader.__next__()
            for r in csv_reader:
                all_stats.append(r)
        
        op_total_time = []
        op_names = []
        # extract gradient related ops
        for op_stat in all_stats:
            if 'gradient_tape' in op_stat[3]:
                op_total_time.append(float(op_stat[5]))
                op_names.append(op_stat[3])
        
        base_layers = model.layers[4].layers
        custom_layers = model.layers[5:]
        model_layers = [*base_layers, *custom_layers]

        t_dw = [0.0 for k in range(len(model.trainable_weights))]
        t_dy = [0.0 for k in range(len(model.trainable_weights))]
        
        weight_count = 0

        for l in model_layers:
            # take care of the standard conv
            if ('Conv1' == l.name) or ('Conv_1' == l.name) or ((l.name).endswith('_project')) or ((l.name).endswith('_expand')):
                if l.use_bias:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'Conv2DBackpropFilter' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'Conv2DBackpropInput' in op:
                            t_dy[weight_count] += t # include TransposeNCHWToNHWC
                        elif l.name in op and 'BiasAddGrad' in op:
                            t_dw[weight_count + 1] = t
                            t_dy[weight_count + 1] = 0
                    weight_count += 2
                else:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'Conv2DBackpropFilter' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'Conv2DBackpropInput' in op:
                            t_dy[weight_count] += t # include TransposeNCHWToNHWC
                    weight_count += 1
            # take care of the lightweight conv
            elif ((l.name).endswith('_depthwise')):
                if l.use_bias:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'DepthwiseConv2dNativeBackpropFilter' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'DepthwiseConv2dNativeBackpropInput' in op:
                            t_dy[weight_count] += t # include TransposeNCHWToNHWC
                        elif l.name in op and 'BiasAddGrad' in op:
                            t_dw[weight_count + 1] = t
                            t_dy[weight_count + 1] = 0
                    weight_count += 2
                else:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'DepthwiseConv2dNativeBackpropFilter' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'DepthwiseConv2dNativeBackpropInput' in op:
                            t_dy[weight_count] += t # include TransposeNCHWToNHWC
                    weight_count += 1
            

            elif ('bn' in l.name) or ('BN' in l.name):
                for op, t in zip(op_names, op_total_time):
                    if l.name in op and 'FusedBatchNormGrad' in op:
                        # for gamma
                        t_dw[weight_count] = 0
                        t_dy[weight_count] = 0
                        # for beta
                        t_dw[weight_count + 1] = 0
                        t_dy[weight_count + 1] = t
                weight_count += 2

            elif 'dense' in l.name:
                if l.use_bias:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'MatMul/MatMul_1' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'MatMul/MatMul' in op:
                            t_dy[weight_count] = t 
                        elif l.name in op and 'BiasAddGrad' in op:
                            t_dw[weight_count + 1] = t
                            t_dy[weight_count + 1] = 0
                    weight_count += 2
                else:
                    for op, t in zip(op_names, op_total_time):
                        if l.name in op and 'MatMul/MatMul_1' in op:
                            t_dw[weight_count] = t
                        elif l.name in op and 'MatMul/MatMul' in op:
                            t_dy[weight_count] = t 
                    weight_count += 1

            else:
                # fuse backprop time of non-trainables to the previous trainable layer
                for op, t in zip(op_names, op_total_time):
                    if l.name in op and weight_count > 0:
                        t_dy[weight_count - 1] += t
        
        # the first layer never propagates input grads, just remove t_dy[0]
        # t_dy = t_dy[1:] # 1~N-1

        t_dw = np.array(t_dw) / num_iterations # (us)
        t_dy = np.array(t_dy) / num_iterations # (us)
        
        print(f'# model trainbles: {len(model.trainable_weights)}')
        print(f'# t_dw: {weight_count}, # t_dy: {weight_count}')
        
        if draw_figure:
            fig = plt.figure()
            plt.barh(np.arange(t_dw.shape[0]), t_dw, color ='navy')
            #plt.xticks(rotation=45)
            plt.xlabel('t_dw (us)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Layer ID', fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
            
            fig = plt.figure()
            plt.barh(np.arange(t_dy.shape[0]), t_dy, color ='navy')
            #plt.xticks(rotation=45)
            plt.xlabel('t_dy (us)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Layer ID', fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
        
        return tf.convert_to_tensor(t_dw/1000.0, tf.float32), tf.convert_to_tensor(t_dy/1000.0, tf.float32) # (ms)
    
    elif model_type == 'vit':
        all_stats = []
        with open(filedir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_reader.__next__()
            for r in csv_reader:
                all_stats.append(r)
        
        op_total_time = []
        op_names = []
        # extract gradient related ops
        for op_stat in all_stats:
            if 'gradient_tape' in op_stat[3]:
                op_total_time.append(float(op_stat[5]))
                op_names.append(op_stat[3])
        
        t_dw = [0.0 for k in range(len(model.trainable_weights))]
        t_dy = [0.0 for k in range(len(model.trainable_weights))]

        for weight_count, w in enumerate(model.trainable_weights):
            if 'embedding/kernel' in w.name:
                for op, t in zip(op_names, op_total_time):
                    if 'embedding' in op and 'Conv2DBackpropFilter' in op:
                        t_dw[weight_count] = t
            elif 'query/kernel' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/MultiHeadDotProductAttention_1/query/Tensordot/MatMul/MatMul_1' in op:
                        t_dw[weight_count] = t
                    elif s + '/MultiHeadDotProductAttention_1/query/Tensordot/MatMul/MatMul' in op:
                        t_dy[weight_count] = t
            elif 'key/kernel' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/MultiHeadDotProductAttention_1/key/Tensordot/MatMul/MatMul_1' in op:
                        t_dw[weight_count] = t
                    elif s + '/MultiHeadDotProductAttention_1/key/Tensordot/MatMul/MatMul' in op:
                        t_dy[weight_count] = t
            elif 'key/bias' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/MultiHeadDotProductAttention_1/truediv/RealDiv' in op:
                        t_dy[weight_count] += t # c = 1/sqrt(d)
                    elif s + '/MultiHeadDotProductAttention_1/mul' in op:
                        t_dy[weight_count] += t # *c
                    elif s + '/MultiHeadDotProductAttention_1/MatMul/' in op:
                        t_dy[weight_count] += t # query*key

            elif 'value/kernel' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/MultiHeadDotProductAttention_1/value/Tensordot/MatMul/MatMul_1' in op:
                        t_dw[weight_count] = t
                    elif s + '/MultiHeadDotProductAttention_1/value/Tensordot/MatMul/MatMul' in op:
                        t_dy[weight_count] = t

            elif 'value/bias' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/MultiHeadDotProductAttention_1/MatMul_1/' in op:
                        t_dy[weight_count] += t # att*value

            elif 'out/kernel' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/MultiHeadDotProductAttention_1/out/Tensordot/MatMul/MatMul_1' in op:
                        t_dw[weight_count] = t
                    elif s + '/MultiHeadDotProductAttention_1/out/Tensordot/MatMul/MatMul' in op:
                        t_dy[weight_count] = t

            elif 'Dense_0/kernel' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/Dense_0/Tensordot/MatMul/MatMul_1' in op:
                        t_dw[weight_count] = t
                    elif s + '/Dense_0/Tensordot/MatMul/MatMul' in op:
                        t_dy[weight_count] = t 

            elif 'Dense_1/kernel' in w.name:
                s = w.name.split('/')[1]
                for op, t in zip(op_names, op_total_time):
                    if s + '/Dense_1/Tensordot/MatMul/MatMul_1' in op:
                        t_dw[weight_count] = t
                    elif s + '/Dense_1/Tensordot/MatMul/MatMul' in op:
                        t_dy[weight_count] = t
        
        t_dw = np.array(t_dw) / num_iterations # (us)
        t_dy = np.array(t_dy) / num_iterations # (us)
        
        print(f'# model trainbles: {len(model.trainable_weights)}')
        print(f'# t_dw: {weight_count+1}, # t_dy: {weight_count+1}')
        
        if draw_figure:
            fig = plt.figure()
            plt.barh(np.arange(t_dw.shape[0]), t_dw, color ='navy')
            #plt.xticks(rotation=45)
            plt.xlabel('t_dw (us)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Layer ID', fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
            
            fig = plt.figure()
            plt.barh(np.arange(t_dy.shape[0]), t_dy, color ='navy')
            #plt.xticks(rotation=45)
            plt.xlabel('t_dy (us)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Layer ID', fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
        
        return tf.convert_to_tensor(t_dw/1000.0, tf.float32), tf.convert_to_tensor(t_dy/1000.0, tf.float32) # (ms)

    else:
        raise NotImplementedError("This model has not been implemented yet")  


def main():
    parser = argparse.ArgumentParser(description='Tensor timing profiling')
    parser.add_argument('--model_name', type=str, default='resnet50', help='valid model names are resnet50, vgg16, mobilenetv2, vit')
    parser.add_argument('--input_size', type=int, default=224, help='input resolution, e.g., 224 stands for 224x224')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size used to run during profiling')
    parser.add_argument('--num_classes', type=int, default=200, help='number of categories model can classify')
    # parser.add_argument('--num_iterations', type=int, default=5, help='number of backward passes to run during profiling')
    
    args = parser.parse_args()
    
    model_name = args.model_name
    input_size = args.input_size
    batch_size = args.batch_size
    num_classes = args.num_classes

    model = port_pretrained_models(
        model_type=model_name,
        input_shape=(input_size, input_size, 3),
        num_classes=num_classes,
    )
    
    run_name = model_name + '_' + str(input_size) + '_' + str(num_classes) + '_' + str(batch_size) + '_' + 'profile'

    profile_backpropagation(
        model,
        (input_size, input_size, 3),
        batch_size,
        5,
        'logs/' + run_name,
    )

    convert_pb_to_csv('logs/' + run_name, 'profile_extracted/' + run_name)

    t_dw, t_dy = profile_parser(
        model, 
        model_name,
        5, 
        'profile_extracted/' + run_name,
        draw_figure=False,
    )

    # for w, t1, t2 in zip(model.trainable_weights, t_dw, t_dy):
    #    print(w.name, t1.numpy(), t2.numpy())

if __name__ == '__main__':
    main()
