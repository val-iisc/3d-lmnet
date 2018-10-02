import tensorflow as tf
import tflearn
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers



# ========================================= POINT CLOUD DECODER ==========================================
'''
Code from @author: optas
Modified for plm
'''
def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d, plm=False):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    _range = (n_layers - 1) if plm else n_layers
    for i in xrange(_range):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print layer

    if closing is not None:
        layer = closing(layer)
        print layer

    if not plm:
        return layer
    else:
        name = 'encoder_z_mean'
        scope = expand_scope_by_name(scope, name)
        z_mean = fully_connected(layer, n_filters[-1], activation='linear', weights_init='xavier', name=name, regularizer=None, weight_decay=weight_decay, reuse=reuse, scope=scope)
        name = 'encoder_z_log_sigma_sq'
        scope = expand_scope_by_name(scope, name)
        z_log_sigma_sq = fully_connected(layer, n_filters[-1], activation='softplus', weights_init='xavier', name=name, regularizer=None, weight_decay=weight_decay, reuse=reuse, scope=scope)
        return z_mean, z_log_sigma_sq

# ========================================= POINT CLOUD DECODER ==========================================
'''
Code from @author: optas
'''
def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer

# ========================================= IMAGE ENCODER ==========================================
def image_encoder(img_inp, FLAGS):
    '''
    Input:
        img_inp: tf placeholder of shape (B, HEIGHT, WIDTH, 3) corresponding to RGB image
    Returns:
        x_latent: tensor of shape (B, FLAGS.bottleneck) corresponding to the predicted latent vector
    Description:
        Main Architecture for Latent Matching Network
    '''
    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x1=x
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x2=x
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x3=x
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4=x
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #8 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x5=x

    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    x_latent=tflearn.layers.core.fully_connected(x,FLAGS.bottleneck,activation='linear',weight_decay=1e-3,regularizer='L2')
    return x_latent