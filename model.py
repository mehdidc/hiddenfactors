import numpy as np

from lasagne import layers, nonlinearities
from helpers import Unpool2DLayer
from lasagnekit.easy import InputOutputMapping
import theano.tensor as T

def build_fully(latent_size=4, nb_hidden=1000, w=28, h=28, c=1, output_dim=10):
    num_hidden_units = nb_hidden
    input_dim = w*h*c
    l_in = layers.InputLayer((None, input_dim))
    # encoder
    l_encoder1 = layers.DenseLayer(l_in, num_units=num_hidden_units)
    l_encoder2 = layers.DenseLayer(l_encoder1, num_units=num_hidden_units)
    l_encoder3 = layers.DenseLayer(l_encoder2, num_units=num_hidden_units)
    l_encoder4 = layers.DenseLayer(l_encoder3, num_units=num_hidden_units)
    l_encoder4 = layers.DenseLayer(l_encoder4, num_units=num_hidden_units, name="pre_code_layer")

    # learned representation
    l_observed = layers.DenseLayer(l_encoder4, num_units=output_dim,
                                      nonlinearity=T.nnet.softmax, name="observed")

    l_latent = layers.DenseLayer(l_encoder4,
                                 num_units=latent_size,
                                 nonlinearity=None, name="factors") # linear

    l_representation = layers.concat([l_observed, l_latent])

    # decoder
    l_decoder1 = layers.DenseLayer(l_representation, num_units=num_hidden_units)
    l_decoder2 = layers.DenseLayer(l_decoder1, num_units=num_hidden_units)
    l_decoder3 = layers.DenseLayer(l_decoder2, num_units=num_hidden_units)
    l_decoder4 = layers.DenseLayer(l_decoder3, num_units=num_hidden_units)
    l_encoder4 = layers.DenseLayer(l_encoder4, num_units=num_hidden_units)

    l_decoder_out = layers.DenseLayer(l_decoder4, num_units=input_dim,
                                       nonlinearity=nonlinearities.sigmoid)

    x_to_z = InputOutputMapping([l_in], [l_latent])
    x_to_y = InputOutputMapping([l_in], [l_observed])
    z_to_x = InputOutputMapping([l_observed, l_latent], [l_decoder_out])
    model = Model()
    model.x_to_z = x_to_z
    model.x_to_y = x_to_y
    model.z_to_x = z_to_x
    model.l_reconstruction = l_decoder_out
    model.l_latent = l_latent
    model.l_observed = l_observed
    return model

def build_convnet(nb_filters=64, size_filters=5, nb_hidden=1000,
                  w=32, h=32, c=1, output_dim=10, latent_size=100):
    nb_filters_encoder = nb_filters
    nb_filters_decoder = nb_filters
    size_filters_encoder = size_filters
    size_filters_decoder = size_filters

    l_in = layers.InputLayer((None, w*h*c), name="input")

    x_in_reshaped = layers.ReshapeLayer(l_in, ([0], c, w, h), name="input_r")

    l_conv = layers.Conv2DLayer(
        x_in_reshaped,
        num_filters=nb_filters_encoder,
        filter_size=(size_filters_encoder, size_filters_encoder),
        nonlinearity=nonlinearities.rectify,
        name="conv1"
    )

    l_pool = layers.Pool2DLayer(
            l_conv,
            pool_size=2,
            name="pool1"
    )

    l_conv = layers.Conv2DLayer(
        l_pool,
        num_filters=nb_filters_encoder,
        filter_size=(size_filters_encoder, size_filters_encoder),
        nonlinearity=nonlinearities.rectify,
        name="conv2"
    )

    l_pool = layers.Pool2DLayer(
            l_conv,
            pool_size=2,
            name="pool2"
    )

    l_hid = layers.DenseLayer(
        l_pool,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
        name="hid1"
    )

    l_hid = layers.DenseLayer(
        l_hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
        name="hid2"
    )

    l_observed = layers.DenseLayer(l_hid,
                                   num_units=output_dim,
                                   nonlinearity=nonlinearities.softmax,
                                   name="observed")
    l_latent = layers.DenseLayer(l_hid,
                                 num_units=latent_size,
                                 nonlinearity=nonlinearities.linear,
                                 b=None,
                                 name="latent")

    hid = layers.ConcatLayer([l_latent, l_observed], axis=1, name="hid2'")

    l_hid = layers.DenseLayer(
        hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )

    l_hid = layers.DenseLayer(
        l_hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )

    num_units = np.prod(l_pool.output_shape[1:])
    hid = layers.DenseLayer(l_hid, num_units=num_units)
    shape = l_pool.output_shape[1:]
    hid = layers.ReshapeLayer(hid, ([0],) + shape)
    l_unpool = Unpool2DLayer(
            hid,
            (2, 2)
    )

    l_unconv = layers.Conv2DLayer(
            l_unpool,
            num_filters=nb_filters_decoder,
            filter_size=(size_filters_decoder, size_filters_decoder),
            pad='full',
            nonlinearity=nonlinearities.rectify,
    )

    l_unpool = Unpool2DLayer(
            l_unconv,
            (2, 2)
    )

    l_unconv = layers.Conv2DLayer(
            l_unpool,
            num_filters=c,
            filter_size=(size_filters_decoder, size_filters_decoder),
            pad='full',
            nonlinearity=nonlinearities.linear,
    )
    l_decoder_out = layers.ReshapeLayer(l_unconv, ([0], c*w*h))
    l_decoder_out = layers.NonlinearityLayer(
            l_decoder_out,
            nonlinearities.sigmoid)

    x_to_z = InputOutputMapping([l_in], [l_latent])
    x_to_y = InputOutputMapping([l_in], [l_observed])
    z_to_x = InputOutputMapping([l_observed, l_latent], [l_decoder_out])
    model = Model()
    model.x_to_z = x_to_z
    model.x_to_y = x_to_y
    model.z_to_x = z_to_x
    model.l_reconstruction = l_decoder_out
    model.l_latent = l_latent
    model.l_observed = l_observed
    return model


class Model(object):
    def get_all_params(self, **t):
        return list(set(self.x_to_z.get_all_params(**t) +
                        self.x_to_y.get_all_params(**t) +
                        self.z_to_x.get_all_params(**t)))
