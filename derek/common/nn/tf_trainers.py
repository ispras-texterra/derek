import tensorflow as tf


def get_optimizer(props, learning_rate):
    name = props["optimizer"]
    if name == "adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif name == "adagrad":
        return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif name == "momentum":
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=props['momentum'])
    elif name == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif name == "nadam":
        return tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
    elif name == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise Exception("Unknown optimizer!")


def get_train_op(loss, optimizer, props):
    grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
    gradients, variables = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(gradients, props['clip_norm'])
    return optimizer.apply_gradients(zip(clipped_grads, variables))
