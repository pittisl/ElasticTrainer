from selection_solver_DP import selection_DP, downscale_t_dy_and_t_dw
from profiler import profile_parser
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import time
from tqdm import tqdm
import os
from utils import clear_cache_and_rec_usage


def full_training(
    model,
    ds_train,
    ds_test,
    run_name,
    logdir,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
):
    """All NN weights will be trained"""

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)

    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    runid = run_name + '_FULL_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    

    print(f"RUNID: {runid}")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    training_step = 0
    best_validation_acc = 0

    clear_cache_and_rec_usage()

    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):
        
        t0 = time.time()
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1

            train_step(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()

        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                #model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)
    
        clear_cache_and_rec_usage()
    
    print("total time excluding validation (s):", total_time_0)
    print("total time including validation (s):", total_time_1)
    # sig_stop_handler(None, None)

def bn_plus_bias_training(
    model,
    ds_train,
    ds_test,
    run_name,
    logdir,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
):
    """Only train normalization, bias, and last layer weights"""
    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    runid = run_name + '_FULL_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    bias_plus_bn_weights = list()
    # print('model.trainable_weights: {}'.format([x.name for x in model.trainable_weights]))
    for i in model.trainable_weights:
        tmp_name = i.name
        if 'gamma' in tmp_name or 'beta' in tmp_name or 'bias' in tmp_name or 'dense' in tmp_name or 'head' in tmp_name:
            bias_plus_bn_weights.append(i)
    # print('bias_plus_bn_weights: {}'.format([x.name for x in bias_plus_bn_weights]))
    model_trainable_weights = bias_plus_bn_weights

    print(f"RUNID: {runid}")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, model_trainable_weights)
        optimizer.apply_gradients(zip(gradients, model_trainable_weights))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    training_step = 0
    best_validation_acc = 0

    clear_cache_and_rec_usage()
    
    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):
        
        t0 = time.time()
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1


            train_step(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()

        cls_loss.reset_states()
        accuracy.reset_states()
        
        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                #model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()
        
        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)

        clear_cache_and_rec_usage()
    
    print("total time excluding validation (s):", total_time_0)
    print("total time including validation (s):", total_time_1)
    # sig_stop_handler(None, None)

def traditional_tl_training(
    model,
    ds_train,
    ds_test,
    run_name,
    logdir,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
):
    """Only train last layer weights"""
    
    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    runid = run_name + '_FULL_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    dense_weights = list()
    # print('model.trainable_weights: {}'.format([x.name for x in model.trainable_weights]))
    for i in model.trainable_weights:
        tmp_name = i.name
        if 'dense' in tmp_name or 'head' in tmp_name:
            dense_weights.append(i)
    # print('bias_plus_bn_weights: {}'.format([x.name for x in dense_weights]))
    model_trainable_weights = dense_weights

    print(f"RUNID: {runid}")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, model_trainable_weights)
        optimizer.apply_gradients(zip(gradients, model_trainable_weights))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    training_step = 0
    best_validation_acc = 0

    clear_cache_and_rec_usage()
    
    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):
        
        t0 = time.time()
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1


            train_step(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()

        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                #model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)

        clear_cache_and_rec_usage()

    print("total time excluding validation (s):", total_time_0)
    print("total time including validation (s):", total_time_1)
    # sig_stop_handler(None, None)

def elastic_training(
    model,
    model_name,
    ds_train,
    ds_test,
    run_name,
    logdir,
    timing_info,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
    interval=4,
    rho=0.533,
):
    """Train with ElasticTrainer"""

    def rho_for_backward_pass(rho):
        return (rho - 1/3)*3/2
    
    t_dw, t_dy = profile_parser(
        model,
        model_name,
        5,
        'profile_extracted/' + timing_info,
        draw_figure=False,
    )
    #np.savetxt('t_dy.out', t_dy)
    #np.savetxt('t_dw.out', t_dw)
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
    t_dy_q = np.flip(t_dy_q)
    t_dw_q = np.flip(t_dw_q)

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    runid = run_name + '_DP_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)


    @tf.function
    def compute_dw(x, y):
        with tf.GradientTape() as tape:
            y_pred_0 = model(x, training=True)
            loss_0 = loss_fn_cls(y, y_pred_0)
        grad_0 = tape.gradient(loss_0, model.trainable_weights)
        w_0 = [w.value() for w in model.trainable_weights] # record initial weight values
        optimizer.apply_gradients(zip(grad_0, model.trainable_weights))
        w_1 = [w.value() for w in model.trainable_weights] # record weight values after applying optimizer
        dw_0 = [w_1_k - w_0_k for (w_0_k, w_1_k) in zip(w_0, w_1)] # compute weight changes
        with tf.GradientTape() as tape:
            y_pred_1 = model(x, training=True)
            loss_1 = loss_fn_cls(y, y_pred_1)
        grad_1 = tape.gradient(loss_1, model.trainable_weights)
        I = [tf.reduce_sum((grad_1_k * dw_0_k)) for (grad_1_k, dw_0_k) in zip(grad_1, dw_0)]
        I = tf.convert_to_tensor(I)
        I = I / tf.reduce_max(tf.abs(I))
        # restore weights
        for k, w in enumerate(model.trainable_weights):
            w.assign(w_0[k])
        return dw_0, I

    training_step = 0
    best_validation_acc = 0
    
    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):

        t0 = time.time()
        if epoch % interval == 0:
            for x_probe, y_probe in ds_train.take(1):
                dw, I = compute_dw(x_probe, y_probe)
                I = -I.numpy()
                I = np.flip(I)
                #np.savetxt('importance.out', I)
                rho_b = rho_for_backward_pass(rho)
                max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_b*disco)
                m = np.flip(m)
                print("m:", m)
                print("max importance:", max_importance)
                print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))
                var_list = []
                all_vars = model.trainable_weights
                for k, m_k in enumerate(m):
                    if tf.equal(m_k, 1):
                        var_list.append(all_vars[k])
                train_step_cpl = tf.function(train_step)
                
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1

            train_step_cpl(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()


        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                # model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)
    
    print("total time excluding validation (s):", total_time_0)
    print("total time including validation (s):", total_time_1)
    # sig_stop_handler(None, None)

def elastic_training_weight_magnitude(
    model,
    model_name,
    ds_train,
    ds_test,
    run_name,
    logdir,
    timing_info,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
    interval=4,
    rho=0.533,
):
    """Train with ElasticTrainer but use weight magnitude as importance metric"""

    def rho_for_backward_pass(rho):
        return (rho - 1/3)*3/2
    
    t_dw, t_dy = profile_parser(
        model,
        model_name,
        5,
        'profile_extracted/' + timing_info,
        draw_figure=False,
    )
    #np.savetxt('t_dy.out', t_dy)
    #np.savetxt('t_dw.out', t_dw)
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
    t_dy_q = np.flip(t_dy_q)
    t_dw_q = np.flip(t_dw_q)

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
        
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    runid = run_name + '_DP_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    training_step = 0
    best_validation_acc = 0
    
    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):

        t0 = time.time()
        if epoch % interval == 0:
            I = np.array([tf.reduce_sum(tf.abs(w.value())) for w in model.trainable_weights])
            I = np.flip(I)
            #np.savetxt('importance.out', I)
            rho_b = rho_for_backward_pass(rho)
            max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_b*disco)
            m = np.flip(m)
            print("m:", m)
            print("max importance:", max_importance)
            print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))
            var_list = []
            all_vars = model.trainable_weights
            for k, m_k in enumerate(m):
                if tf.equal(m_k, 1):
                    var_list.append(all_vars[k])
            train_step_cpl = tf.function(train_step)
                
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1

            train_step_cpl(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()


        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                # model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)
    
    print("total time excluding validation (s):", total_time_0)
    print("total time including validation (s):", total_time_1)
    # sig_stop_handler(None, None)

def elastic_training_grad_magnitude(
    model,
    model_name,
    ds_train,
    ds_test,
    run_name,
    logdir,
    timing_info,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
    interval=4,
    rho=0.4,
):
    """Train with ElasticTrainer but use gradient magnitude as importance metric"""

    def rho_for_backward_pass(rho):
        return (rho - 1/3)*3/2
    
    t_dw, t_dy = profile_parser(
        model,
        model_name,
        5,
        'profile_extracted/' + timing_info,
        draw_figure=False,
    )
    #np.savetxt('t_dy.out', t_dy)
    #np.savetxt('t_dw.out', t_dw)
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
    t_dy_q = np.flip(t_dy_q)
    t_dw_q = np.flip(t_dw_q)

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    runid = run_name + '_DP_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)


    @tf.function
    def compute_dw(x, y):
        with tf.GradientTape() as tape:
            y_pred_0 = model(x, training=True)
            loss_0 = loss_fn_cls(y, y_pred_0)
        grad_0 = tape.gradient(loss_0, model.trainable_weights)
        I = [tf.reduce_sum(tf.abs(grad_0_k)) for grad_0_k in grad_0]
        I = tf.convert_to_tensor(I)
        return I

    training_step = 0
    best_validation_acc = 0
    
    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):

        t0 = time.time()
        if epoch % interval == 0:
            for x_probe, y_probe in ds_train.take(1):
                I = compute_dw(x_probe, y_probe)
                I = I.numpy()
                I = np.flip(I)
                #np.savetxt('importance.out', I)
                rho_b = rho_for_backward_pass(rho)
                max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_b*disco)
                m = np.flip(m)
                print("m:", m)
                print("max importance:", max_importance)
                print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))
                var_list = []
                all_vars = model.trainable_weights
                for k, m_k in enumerate(m):
                    if tf.equal(m_k, 1):
                        var_list.append(all_vars[k])
                train_step_cpl = tf.function(train_step)
                
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1

            train_step_cpl(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()


        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                # model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)
    
    print("total time excluding validation (s):", total_time_0)
    print("total time including validation (s):", total_time_1)
    # sig_stop_handler(None, None)
