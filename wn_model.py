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
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_activation_summary
from tensorpack.utils.gpu import get_num_gpu

from aparse import ArgParser


LR = 0.1
ADAPTIVE_LR = False
MULT_DECAY = 1.
CENTERED_WEIGHT = False
EPSILON=1e-8
THETA_INIT = 0.
THETA_LR = 0.1

CIFAR_TRAIN_MEAN = np.array([125.30691805, 122.95039414, 113.86538318])
CIFAR_TRAIN_STD = np.array([62.99321928, 62.08870764, 66.70489964])
CIFAR_TRAIN_CHAN_MOMENT2 = np.square(CIFAR_TRAIN_STD)
CIFAR_TRAIN_PIXEL_MOMENT2 = np.array([3944.54483343, 3794.65961416, 4351.44963   ])

BATCH_SIZE = 64

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

def NormConv2DScale(x, chan, kernel, stride,
        center=False, input_moment2=0.5, # after ReLU
        scale_mean=1., scale_std=0., # one initializer
        center_mean=0., center_mom=0.,
        lr_mult_scale=1., lr_mult_weight=1.,
        eps=EPSILON, name="norm_conv2d_scale"):
    chan_in = x.get_shape().as_list()[-1]
    shape_weight = (kernel, kernel, chan_in, chan)
    #
    with tf.variable_scope(name):
        scale = get_chan_weight(chan_in, scale_mean, scale_std, lr_mult=lr_mult_scale, name="scale")
        w = get_weight(shape_weight, neg_scale=0, lr_mult=lr_mult_weight, name="W")
        w_scaled = w * tf.cast(scale[tf.newaxis, tf.newaxis, :, tf.newaxis], w.dtype)
        if isinstance(input_moment2, float):
            input_moment2 = tf.constant([input_moment2], dtype=w.dtype)
        sigma2 = tf.reduce_sum(tf.reduce_sum(tf.square(w_scaled), axis=[0,1]) \
                * input_moment2[:, tf.newaxis], axis=[0], name="sigma2")
        w_normalized = w_scaled * tf.rsqrt(sigma2 + eps)
        y = tf.nn.conv2d(x, tf.cast(w_normalized, x.dtype), strides=[1, stride, stride, 1], padding="SAME")
        if center:
            ma_mu = get_chan_weight(chan, center_mean, 0., lr_mult=0., name="ma_mu")
            curr_mu = tf.reduce_mean(y, axis=[0,1,2], name="curr_mu")
            mu = ma_mu * center_mom + curr_mu * (1-center_mom)
            update_mu = tf.assign(ma_mu, mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            y = y - mu
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

def SphericalAdd(x1, x2, theta_mean=0., theta_std=0., theta_lr_mult=1., name="sp_add"):
    chan = x1.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        theta = get_chan_weight(chan, theta_mean, theta_std, theta_lr_mult, name="theta")
        s1 = tf.math.cos(theta, name="s1")
        s2 = tf.math.sin(theta, name="s2")
        y = tf.add(x1*s1, x2*s2)
    return y


class CifarResNet(ModelDesc):
    def __init__(self, args, lr_init=LR*0.1, name="cifar_resnet"):
        super(CifarResNet, self).__init__()
        self._n = args["num_units"]
        self._n_classes = args["n_classes"] 
        self._center = args["center"]
        self._theta_init = args["theta_init"]
        self._theta_lr_mult = args["theta_lr"]
        self._lr_init = lr_init
        self._name = name

    @staticmethod
    def get_parser(ap=None, name="cifar_resnet"):
        ap = ArgParser(ap, name=name)
        ap.add("-n", "--num_units",
                help="number of units in each stage",
                type=int, default=3)
        ap.add("--n-classes", type=int, default=10)
        ap.add_flag("--center")
        ap.add("--theta-init", type=float, default=THETA_INIT)
        ap.add("--theta-lr", type=float, default=THETA_LR)
        return ap

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    @staticmethod
    def build_resblock(x, chan, kernel=3, stride=1, first=False,
            center=False, theta_init=THETA_INIT, theta_lr_mult=THETA_LR, name="resblock"):
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
                shortcut = NormConv2DScale(shortcut, chan, stride, stride, center=center, name="shortcut_conv")
            # residual branch
            act_residual = ActBias(x, name="act_residual")
            conv1 = NormConv2DScale(act_residual, chan, kernel, stride, center=center, name="n-conv-s1")
            act1 = ActBias(conv1, name="act1")
            conv2 = NormConv2DScale(act1, chan, kernel, 1, center=center, name="n-conv-s2")
            # join two paths
            y = SphericalAdd(shortcut, conv2, theta_mean=theta_init, theta_lr_mult=theta_lr_mult)
        return y

    @staticmethod
    def build_group(x, n_blocks, chan, kernel=3, stride=1, center=False,
            theta_init=THETA_INIT, theta_lr_mult=THETA_LR, name="group"):
        with tf.variable_scope(name):
            for b in range(n_blocks):
                first = (b == 0)
                x = CifarResNet.build_resblock(x, chan, kernel, (stride if first else 1),
                        first, center, theta_lr_mult, name="b%d" % (b+1))
        return x

    def build_graph(self, image, label):
        scale_image = 1./128.0
        image = image * scale_image
        image_moment2 = CIFAR_TRAIN_PIXEL_MOMENT2 * scale_image * scale_image
        assert tf.test.is_gpu_available()

        with tf.variable_scope(self._name):
            x = NormConv2DScale(image, 16, 3, 1,
                    center=self._center, input_moment2=image_moment2, name="conv_input")
            add_activation_summary(x, types=["mean", "rms", "histogram"])
            # shape = [batchsize, 32, 32, 16]
            x = CifarResNet.build_group(x, self._n, 16, stride=1, center=self._center,
                    theta_init=self._theta_init, theta_lr_mult=self._theta_lr_mult, name="g1")
            add_activation_summary(x, types=["mean", "rms", "histogram"])
            # shape = [batchsize, 16, 16, 32]
            x = CifarResNet.build_group(x, self._n, 32, stride=2, center=self._center,
                    theta_init=self._theta_init, theta_lr_mult=self._theta_lr_mult, name="g2")
            add_activation_summary(x, types=["mean", "rms", "histogram"])
            # shape = [batchsize, 8, 8, 64]
            x = CifarResNet.build_group(x, self._n, 64, stride=2, center=self._center,
                    theta_init=self._theta_init, theta_lr_mult=self._theta_lr_mult, name="g3")
            add_activation_summary(x, types=["mean", "rms", "histogram"])
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
        
        add_param_summary(('.*/theta', ['histogram']))
        add_param_summary(('.*/ma_mu', ['histogram']))
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
    ap = CifarResNet.get_parser()
    ap.add('--gpu', help='comma separated list of GPU(s) to use.')
    ap.add('--load', help='load model for training')
    ap.add("--save-dir")
    ap.add("--lr", type=float, default=0.1)
    args = ap.parse_args()
    lr_base = args["lr"]
    save_dir = args["save_dir"]

    if args["gpu"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu"]

    if save_dir is None:
        logger.auto_set_dir()
    else:
        logger.set_logger_dir(save_dir)

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=CifarResNet(args, lr_init=lr_base*0.1),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, lr_base), (82, lr_base*0.1), (123, lr_base*0.01), (164, lr_base*0.002)])
        ],
        max_epoch=200,
        session_init=SmartInit(args["load"]),
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
