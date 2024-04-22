# -*- coding: utf-8 -*-
from input import *
from model import *
import tensorflow as tf

def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        save_summary_steps=100,
        save_checkpoints_steps=100,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=100,
        session_config=session_config
    )
    model = CrossDQN()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn_estimator, config=config)
    return estimator


def save_estimator(estimator, export_dir):
    def _serving_input_receiver_fn():
        receiver_tensors = {
            'user_id': tf.placeholder(tf.int64, [None, 1], name='user_id'),
            'behavior_poi_id_list': tf.placeholder(tf.int64, [None, 10], name='behavior_poi_id_list'),
            'ad_id_list': tf.placeholder(tf.int32, [None, 5], name='ad_id_list'),
            'oi_id_list': tf.placeholder(tf.int32, [None, 5], name='oi_id_list'),
            'context_id': tf.placeholder(tf.int32, [None, 1], name='context_id'),
            'action':  tf.placeholder(tf.int32, [None, 15], name='action')
        }  # 假设 max_q_action_index 是一个 Tensor
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)
    export_dir = estimator.export_savedmodel(
        export_dir_base=export_dir, serving_input_receiver_fn=_serving_input_receiver_fn)
    return export_dir


def soft_argmax(x, beta=1.0):
    """Soft argmax function.
    Args:
        x (numpy.ndarray): Input array.
        beta (float): Softness parameter (default: 1.0).

    Returns:
        numpy.ndarray: Soft argmax result.
    """
    e_x = np.exp(beta * x)
    weights = e_x / np.sum(e_x)
    indices = np.arange(len(x))
    return np.sum(indices * weights)


if __name__ == '__main__':
    estimator = create_estimator()
    train_input_fn = input_fn_maker(DATA_PATH)
    # --- train ---
    for i in range(1):
        estimator.train(train_input_fn)
    save_estimator(estimator, PB_SAVE_PATH)

    # --- predict ---
    pre = estimator.predict(train_input_fn)
    data = next(pre)
    print(data, data['Q_value'])
    max_ind = int(round(soft_argmax(data['Q_value'])))
    print("max index is ", max_ind)
    print(WHOLE_ACTION[max_ind])
