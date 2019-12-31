import numpy as np
import tensorflow as tf
import os

from tensorpack import (ModelDesc, GlobalAvgPooling, FullyConnected,
    get_global_step_var, regularize_cost, imgaug, AugmentImageComponent,
    BatchData, MultiProcessRunner)
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

from aparse import ArgParser

ADAPTIVE_LR = False
BATCH_SIZE = 128

def get_weight(shape, gain=None, pos_scale=1., neg_scale=0., gain_mult=1,
        lr_mult=1, adaptive_lr=ADAPTIVE_LR,
        input_dims=None, output_dim=-1, name="weight"):
    """Weight with one output dim

    Assume the last dim to be the output dim.

    gain is a multiplier to the initial std, which depends on the type of non-linearity
    before the weighted layer which it belongs to.
    * For relu (pos_scale=1, neg_scale=0), gain = np.sqrt(2)
    * For leaky_relu (pos_scale=1, neg_scale=0.2), gain = np.sqrt(2 / (pos_scale**2 + neg_scale**2))
    * For tanh (pos_scale=1, neg_scale=1, approximately), gain = np.sqrt(1)

    lr_mult is the relative learning rate which is achieved by scale the learnable
    variable without changing the actual std. It also depends on the type of optimizer.
    """
    if gain is None:
        gain = np.sqrt(2. / (pos_scale**2 + neg_scale**2)) * gain_mult
    if input_dims is None:
        input_dims = list(range(len(shape)))
        input_dims.pop(output_dim)
    fan_in = np.prod([shape[i] for i in input_dims])
    std = gain / np.sqrt(fan_in)

    assert lr_mult >= 0
    if lr_mult == 0:
        init_weight = tf.initializers.random_normal(0, std)
        return tf.get_variable(name, shape=shape, trainable=False, initializer=init_weight)
    else:
        # The gradient computed using tf.train.AdamOptimizer is independent
        # of the multiplier of weight. If gradient is propotional to the multiplier
        # (e.g. tf.train.MomentumOptimizer), the multiplier need to be adjusted to
        # match lr_mult.
        w_mult = lr_mult if adaptive_lr else np.sqrt(lr_mult)
        init_std = std / w_mult
        init_weight = tf.initializers.random_normal(0, init_std)
        return tf.get_variable(name, shape=shape, initializer=init_weight) * w_mult

def get_chan_weight(chan, mean, std,
        lr_mult=1, adaptive_lr=ADAPTIVE_LR, name="chan_weight"):
    assert lr_mult >= 0
    if lr_mult == 0:
        init_weight = tf.initializers.random_normal(mean, std)
        return tf.get_variable(name, shape=(chan,), trainable=False, initializer=init_weight)
    else:
        w_mult = lr_mult if adaptive_lr else np.sqrt(lr_mult)
        init_weight = tf.initializers.random_normal(mean, std/w_mult)
        return tf.get_variable(name, shape=(chan,), initializer=init_weight) * w_mult

def NormConv2DScale(x, chan, kernel, stride,
        scale_mean=1., scale_std=0., # one initializer
        lr_mult_scale=1., lr_mult_weight=1.,
        eps=1e-8, centered=True, name="norm_conv2d_scale"):
    chan_in = x.get_shape().as_list()[-1]
    shape_weight = (kernel, kernel, chan_in, chan)
    #
    with tf.variable_scope(name):
        scale = get_chan_weight(chan_in, scale_mean, scale_std, lr_mult=lr_mult_scale, name="scale")
        w = get_weight(shape_weight, neg_scale=1., lr_mult=lr_mult_weight, name="weight")
        w_scaled = w * tf.cast(scale[:, tf.newaxis, tf.newaxis, tf.newaxis], w.dtype)
        if centered:
            mu, sigma2 = tf.nn.moments(w_scaled, axes=[0,1,2])
        else:
            mu, sigma2 = 0, tf.reduce_sum(tf.square(w_scaled), axis=[0,1,2])
        w_normalized = (w_scaled - mu) * tf.rsqrt(sigma2 + eps)
        y = tf.nn.conv2d(x, tf.cast(w_normalized, x.dtype), strides=[stride, stride, 1, 1], padding="SAME")
    return y
        
def ActBias(x, alpha=None, bias_mean=0., bias_std=0., lr_mult=1., name="act_bias"):
    chan_in = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        b = get_chan_weight(chan_in, bias_mean, bias_std, lr_mult, name="bias")
        if alpha is None:
            y = tf.nn.relu(x + tf.cast(b, x.dtype))
        else:
            y = tf.nn.leaky_relu(x + tf.cast(b, x.dtype), alpha)
    return y


class CifarResNet(ModelDesc):
    def __init__(self, n, name="cifar_resnet"):
        super(CifarResNet, self).__init__()
        self._n = n
        self._name = name
        self._mult_decay = 1.
        self._n_classes = 10

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    @staticmethod
    def build_resblock(x, chan, kernel=3, stride=1, res_mult=1., name="resblock"):
        chan_in = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            if chan_in == chan and stride == 1:
                shortcut = x
            else:
                shortcut = ActBias(x, name="shortcut_act")
                shortcut = NormConv2DScale(shortcut, chan, stride, stride, name="shortcut_conv")
            act_input = ActBias(x, name="act_input")
            conv1 = NormConv2DScale(act_input, chan, kernel, stride, name="conv1")
            act1 = ActBias(conv1, name="act1")
            conv2 = NormConv2DScale(act1, chan, kernel, 1, name="conv2")
            y = shortcut + conv2 * res_mult
        return y

    @staticmethod
    def build_group(x, n_blocks, chan, kernel=3, stride=1, res_mult=1., mult_decay=1., name="group"):
        sum_mult = 0.
        for b in range(n_blocks):
            x = CifarResNet.build_resblock(x, chan, kernel, (stride if b == 0 else 1),
                res_mult, name="b%d" % (b+1))
            sum_mult += res_mult
            res_mult *= mult_decay
        return x, res_mult, sum_mult

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()

        with tf.variable_scope(self._name):
            res_mult = 1.
            sum_mult = 1.
            x = NormConv2DScale(image, 16, 3, 1, name="conv_input")
            # shape = [batchsize, 32, 32, 16]
            x, res_mult, group_mult = CifarResNet.build_group(x, self._n, 16, stride=1,
                res_mult=res_mult, mult_decay=self._mult_decay, name="g1")
            sum_mult += group_mult
            # shape = [batchsize, 16, 16, 32]
            x, res_mult, group_mult = CifarResNet.build_group(x, self._n, 32, stride=2,
                res_mult=res_mult, mult_decay=self._mult_decay, name="g2")
            sum_mult += group_mult
            # shape = [batchsize, 8, 8, 64]
            x, res_mult, group_mult = CifarResNet.build_group(x, self._n, 64, stride=2,
                res_mult=res_mult, mult_decay=self._mult_decay, name="g3")
            sum_mult += group_mult
            # normalise the final output by the accumulated multiplier
            x = x / sum_mult
            #
            x = GlobalAvgPooling("gap", x)
            logits = FullyConnected("linear", x, self._n_classes)
            prob = tf.nn.softmax(logits, name="prob")

            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
            cost = tf.reduce_mean(cost, name="cross_entropy_loss")

            acc_top1 = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name="acc_top1")
            add_moving_summary(tf.reduce_mean(acc_top1, name="train_acc_top1"))

            wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(), 480000, 0.2, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            add_moving_summary(cost, wd_cost)

            return tf.add_n([cost, wd_cost], name="cost")

    def optimizer(self):
        lr = tf.get_variable("learning_rate", initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def get_data(phase):
    is_train = phase == "train"
    ds = dataset.Cifar10(phase)
    pp_mean = ds.get_per_pixel_mean(("train",))
    if is_train:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not is_train)
    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model for training')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        max_epoch=400,
        session_init=SmartInit(args.load),
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))