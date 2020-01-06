import numpy as np
import tensorflow as tf
import os
import argparse

from tensorpack import (ModelDesc, GlobalAvgPooling, FullyConnected, BatchNorm,
    get_global_step_var, regularize_cost, imgaug, AugmentImageComponent,
    BatchData, MultiProcessRunner, logger, TrainConfig, ModelSaver,
    InferenceRunner, ScalarStats, ClassificationError, ScheduledHyperParamSetter,
    SmartInit, launch_train_with_config, SyncMultiGPUTrainerParameterServer)
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

#from aparse import ArgParser


LR = 0.1
ADAPTIVE_LR = False
MULT_DECAY = 1.
CENTERED_WEIGHT = True
EPSILON=1e-8

BATCH_SIZE = 64
NUM_UNITS = None

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
        gain = np.sqrt(2. / (pos_scale**2 + neg_scale**2))
    if input_dims is None:
        input_dims = list(range(len(shape)))
        input_dims.pop(output_dim)
    fan_in = np.prod([shape[i] for i in input_dims])
    std = gain * gain_mult / np.sqrt(fan_in)

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

def _NormConv2DScale(x, chan, kernel, stride,
        scale_mean=1., scale_std=0., # one initializer
        lr_mult_scale=1., lr_mult_weight=1.,
        eps=1e-8, centered=CENTERED_WEIGHT, name="norm_conv2d_scale"):
    chan_in = x.get_shape().as_list()[-1]
    shape_weight = (kernel, kernel, chan_in, chan)
    #
    with tf.variable_scope(name):
        scale = get_chan_weight(chan_in, scale_mean, scale_std, lr_mult=lr_mult_scale, name="scale")
        w = get_weight(shape_weight, neg_scale=0, lr_mult=lr_mult_weight, name="W")
        w_scaled = w * tf.cast(scale[tf.newaxis, tf.newaxis, :, tf.newaxis], w.dtype)
        if centered:
            mu, sigma2 = tf.nn.moments(w_scaled, axes=[0,1,2])
        else:
            mu, sigma2 = 0, tf.reduce_sum(tf.square(w_scaled), axis=[0,1,2])
        w_normalized = (w_scaled - mu) * tf.rsqrt(sigma2 + eps)
        y = tf.nn.conv2d(x, tf.cast(w_normalized, x.dtype), strides=[1, stride, stride, 1], padding="SAME")
    return y

def Conv2D(x, chan, kernel, stride, name="conv2d"):
    chan_in = x.get_shape().as_list()[-1]
    shape_weight = (kernel, kernel, chan_in, chan)
    with tf.variable_scope(name):
        w = get_weight(shape_weight, neg_scale=0, name="W")
        y = tf.nn.conv2d(x, tf.cast(w, x.dtype), strides=[1, stride, stride, 1], padding="SAME")
    return y

def Scale(x, scale_mean, scale_std, name="scale"):
    chan_in = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        s = get_chan_weight(chan_in, scale_mean, scale_std)
        y = x * s
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

def NormConv2DScale(x, chan, kernel, stride,
        scale_mean=1., scale_std=0., # one initializer
        eps=EPSILON, name="norm_conv2d_scale"):
    with tf.variable_scope(name):
        x_scaled = Scale(x, scale_mean, scale_std)
        y = Conv2D(x_scaled, chan, kernel, stride)
        y_normalised = BatchNorm("bn_noaffine", y, epsilon=eps, center=False, scale=False)
    return y_normalised

def ScaleNormConv2D(x, chan, kernel, stride,
        scale_mean=1., scale_std=0., # one initializer
        eps=EPSILON, name="norm_conv2d_scale"):
    with tf.variable_scope(name):
        y = Conv2D(x, chan, kernel, stride)
        y_normalised = BatchNorm("scale_bn", y, epsilon=eps, center=False, scale=True)
    return y_normalised



class CifarResNet(ModelDesc):
    def __init__(self, n, mult_decay=MULT_DECAY, lr_init=LR*0.1, name="cifar_resnet"):
        super(CifarResNet, self).__init__()
        self._n = n
        self._name = name
        self._mult_decay = mult_decay
        self._n_classes = 10
        self._lr_init = lr_init

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    @staticmethod
    def build_resblock(x, chan, kernel=3, stride=1, res_mult=1., first=False, name="resblock"):
        """
        res_mult : float
            The multiplier for the residual branch. The shortcut is always counted as 1.
        first : bool
            Whether it is the first block or, equivalently, whether to first apply ActBias.
        """
        chan_in = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            # shortcut
            if chan_in == chan and stride == 1:
                shortcut = x
            else:
                shortcut = ActBias(x, name="act_shortcut")
                shortcut = ScaleNormConv2D(shortcut, chan, stride, stride, name="shortcut_conv")
            # residual branch
            act_residual = ActBias(x, name="act_residual")
            conv1 = ScaleNormConv2D(act_residual, chan, kernel, stride, name="n-conv-s1")
            act1 = ActBias(conv1, name="act1")
            conv2 = ScaleNormConv2D(act1, chan, kernel, 1, name="n-conv-s2")
            # join two paths
            y = shortcut + conv2 * res_mult
        return y

    @staticmethod
    def build_preact_resblock(x, chan, kernel=3, stride=1, res_mult=1., first=False, name="resblock"):
        """
        """
        chan_in = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            bn_input = BatchNorm("bn_input", x, epsilon=EPSILON, center=False, scale=True)
            act_input = ActBias(bn_input, name="act_input")
            # shortcut
            if first:
                input_shortcut = act_input
            else:
                input_shortcut = x
            if chan_in == chan and stride == 1:
                shortcut = input_shortcut
            else:
                shortcut = Conv2D(input_shortcut, chan, stride, stride, name="conv_shortcut")
            # residual branch
            conv1 = Conv2D(act_input, chan, kernel, stride, name="conv1")
            bn1 = BatchNorm("bn1", conv1, epsilon=EPSILON, center=False, scale=True)
            act1 = ActBias(bn1, name="act1")
            conv2 = Conv2D(act1, chan, kernel, 1, name="conv2")
            # join two paths
            y = shortcut + conv2 * res_mult
        return y

    @staticmethod
    def build_group(x, n_blocks, chan, kernel=3, stride=1, res_mult=None, mult_decay=1., name="group"):
        sum_mult = 1. # the main stem is always counted as 1
        if res_mult is None:
            res_mult = mult_decay
        assert isinstance(res_mult, (int, float))
        with tf.variable_scope(name):
            for b in range(n_blocks):
                first = (b == 0)
                x = CifarResNet.build_resblock(x / sum_mult, # normalised input
                    chan, kernel, (stride if first else 1),
                    res_mult, first, name="b%d" % (b+1))
                # NOTE As BatchNorm is used after each convolution. The normalizing factor,
                # sum_mult, will actually affects the lr_mult of the first convolution layer
                # in the residual branch, which causes difficulties in training.
                # TODO The effects of res_mult is to be investigated.
                #sum_mult += res_mult
                res_mult *= mult_decay
        return x / sum_mult

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()

        with tf.variable_scope(self._name):
            x = ScaleNormConv2D(image, 16, 3, 1, name="conv_input")
            # shape = [batchsize, 32, 32, 16]
            x = CifarResNet.build_group(x, self._n, 16, stride=1, mult_decay=self._mult_decay, name="g1")
            # shape = [batchsize, 16, 16, 32]
            x = CifarResNet.build_group(x, self._n, 32, stride=2, mult_decay=self._mult_decay, name="g2")
            # shape = [batchsize, 8, 8, 64]
            x = CifarResNet.build_group(x, self._n, 64, stride=2, mult_decay=self._mult_decay, name="g3")
            # normalise the final output by the accumulated multiplier
            #x = BatchNorm("bn_last", x, epsilon=EPSILON, center=False, scale=True)
            x = ActBias(x, name="act_top")
            #
            x = GlobalAvgPooling("gap", x)
            logits = FullyConnected("linear", x, self._n_classes)
            prob = tf.nn.softmax(logits, name="prob")

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name="cross_entropy_loss")

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name="wrong_vector")
        add_moving_summary(tf.reduce_mean(wrong, name="train_error"))

        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(), 480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)
        
        return tf.add_n([cost, wd_cost], name="cost")

    def optimizer(self):
        lr = tf.get_variable("learning_rate", initializer=self._lr_init, trainable=False)
        tf.summary.scalar("lr", lr)
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
                        type=int, default=3)
    parser.add_argument('--load', help='load model for training')
    parser.add_argument("--save-dir")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--mult-decay", type=float, default=MULT_DECAY)
    args = parser.parse_args()
    NUM_UNITS = args.num_units
    mult_decay = args.mult_decay
    lr_base = args.lr
    save_dir = args.save_dir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if save_dir is None:
        logger.auto_set_dir()
    else:
        logger.set_logger_dir(save_dir)

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=CifarResNet(n=NUM_UNITS, mult_decay=mult_decay, lr_init=lr_base*0.1),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, lr_base), (82, lr_base*0.1), (123, lr_base*0.01), (164, lr_base*0.002)])
        ],
        max_epoch=200,
        session_init=SmartInit(args.load),
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
