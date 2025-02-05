
# coding: utf-8

# In[1]:
import matplotlib as mpl
mpl.use('Agg')
from lasagnekit.easy import BatchOptimizer, LightweightModel
from lasagnekit.datasets.mnist import MNIST

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from lasagne import layers, updates,  nonlinearities
import theano.tensor as T
from theano.sandbox import rng_mrg
import theano
import numpy as np
from lasagne.layers import get_all_layers
from skimage.transform import resize
from lasagnekit.generative.capsule import Capsule
from collections import OrderedDict
from lasagne import layers, updates, nonlinearities
from lasagne.layers.helper import get_all_layers
from lasagne.layers import helper
from theano.sandbox import rng_mrg
from sklearn.cross_validation import train_test_split
from lasagnekit.datasets.fonts import Fonts

from lasagnekit.generative.capsule import Capsule
from lasagne.layers import Layer
from lasagnekit.datasets.fonts import Fonts


import sys
import uuid

from lightexperiments.light import Light
light = Light()



light.launch()
light.initials() # save the date and init the timer

light.file_snapshot() # save the content of the python file running
seed = 1234
np.random.seed(seed)
light.set_seed(seed) # save the content of the seed
light.tag("decorrelation") # for tagging your experiments

import argparse
import uuid
parser = argparse.ArgumentParser()
parser.add_argument("latent_size", type=int, default=5)
parser.add_argument("custom_id", type=str, default=str(uuid.uuid1()))

args = parser.parse_args()
print(args.custom_id)
light.set("custom_id", args.custom_id)
latent_size = args.latent_size

#light.tag("vary_hidden_factors")

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis = 2).repeat(2, axis = 3)

class SumLayer(Layer):
    def __init__(self,
                 incoming,
                 axis=1,
                 **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return input.sum(axis=self.axis)

    def get_output_shape_for(self, input):
        shape = list(self.input_shape)
        del shape[self.axis]
        return tuple(shape)




# # Load & pre-process data

from lasagnekit.datasets.cached import Cached
from lasagnekit.datasets.rescaled import Rescaled

# In[2]:


dataset = "fonts"
light.set("dataset", dataset)


if dataset == "mnist":
    data = MNIST()
elif dataset == "fonts":
    data = Fonts(kind="all_64",
                 labels_kind="letters")

w, h = 28, 28
light.set("w", w)
light.set("h", h)


data = Cached(Rescaled(data, (w, h)))
data.load()


X = data.X
y = data.y
output_dim = data.output_dim
real_w, real_h = data.img_dim


# In[3]:


from sklearn.preprocessing import label_binarize
y = label_binarize(y, np.arange(output_dim))
y = y.astype(np.float32)

X, y = shuffle(X, y, random_state=seed)
train, test = train_test_split(range(X.shape[0]), test_size=0.25)


nb_samples_learning_curve = 1000
nb_tries_learning_curve = 10

light.set("nb_samples_learning_curve", nb_samples_learning_curve)
light.set("nb_tries_learning_curve", nb_tries_learning_curve)


class MyBatchOptimizer(BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)

        all_of_them = [
            "acc_train",
            "acc_valid",
            "rec_train",
            "rec_valid",
            "crosscor_train",
            "crosscor_valid"
        ]
        for a in all_of_them:
            status[a] = 0.
        for i in range(nb_tries_learning_curve):

            s = np.arange(len(train))
            np.random.shuffle(s)
            s_train = s[0:nb_samples_learning_curve]


            s = np.arange(len(test))
            np.random.shuffle(s)
            s_test = s[0:nb_samples_learning_curve]


            status["acc_train"] += (self.model.predict(X[train][s_train])==y[train][s_train].argmax(axis=1)).mean()
            status["acc_valid"] += (self.model.predict(X[test][s_test])==y[test][s_test].argmax(axis=1)).mean()

            status["rec_train"] += self.model.get_reconstruction_error(X[train][s_train])
            status["rec_valid"] += self.model.get_reconstruction_error(X[test][s_test])

            status["crosscor_train"] += self.model.get_cross_correlation(X[train][s_train])
            status["crosscor_valid"] += self.model.get_cross_correlation(X[test][s_test])
        for a in all_of_them:
            status[a] /= nb_tries_learning_curve
        for k, v in status.items():
            light.append(k, float(v))
        return status


# ## Model definition

# In[10]:


def cross_entropy(truth, pred):
    return -(truth * T.log(pred) + (1 - truth) * T.log(1 - pred)).sum(axis=1).mean()

def mse(truth, pred):
    return (((truth - pred) ** 2).sum(axis=1)).mean()

def loss_function_y(y_true, y_pred):
    return (T.nnet.categorical_crossentropy(y_pred, y_true)).mean()

def corrupted_masking_noise(rng, x, corruption_level):
    return rng.binomial(size=x.shape, n=1, p=1 - corruption_level) * x

def corrupted_salt_and_pepper(rng, x, corruption_level):
    selected = rng.binomial(size=x.shape, n=1, p=corruption_level, dtype=theano.config.floatX)
    return x * (1 - selected) + selected * rng.binomial(size=x.shape, n=1, p=0.5, dtype=theano.config.floatX)

rng = rng_mrg.MRG_RandomStreams(seed)

def corruption_function(X):
    return corrupted_salt_and_pepper(rng, X, 0.5)


class Model:
    def get_all_params(self, **t):
        return list(set(self.x_to_z.get_all_params(**t) +
                        self.x_to_y.get_all_params(**t) +
                        self.z_to_x.get_all_params(**t)))


# ### model type

# In[11]:

model_type = "convnet" # or "convnet"
light.set(model_type, model_type)

# ### Fully connected

# In[12]:



print("latent_size", latent_size)

if model_type == "fully_connected":
    ## fully connected
    num_hidden_units = 2000
    light.set("latent_size", latent_size)
    light.set("num_hidden_units", num_hidden_units)

    l_in = layers.InputLayer((None, w*h))
    input_dim = w*h
    output_dim = y.shape[1]

    # encoder
    l_encoder1 = layers.DenseLayer(l_in, num_units=num_hidden_units)
    l_encoder2 = layers.DenseLayer(l_encoder1, num_units=num_hidden_units)
    l_encoder3 = layers.DenseLayer(l_encoder2, num_units=num_hidden_units)
    l_encoder4 = layers.DenseLayer(l_encoder3, num_units=num_hidden_units)

    # learned representation
    l_observed = layers.DenseLayer(l_encoder4, num_units=output_dim,
                                   nonlinearity=T.nnet.softmax)

    l_latent = layers.DenseLayer(l_encoder4,
                                 num_units=latent_size,
                                 nonlinearity=None) # linear

    l_representation = layers.concat([l_observed, l_latent])

    # decoder
    l_decoder1 = layers.DenseLayer(l_representation, num_units=num_hidden_units)
    l_decoder2 = layers.DenseLayer(l_decoder1, num_units=num_hidden_units)
    l_decoder3 = layers.DenseLayer(l_decoder2, num_units=num_hidden_units)
    l_decoder4 = layers.DenseLayer(l_decoder3, num_units=num_hidden_units)
    l_decoder_out = layers.DenseLayer(l_decoder4, num_units=input_dim,
                                       nonlinearity=nonlinearities.sigmoid)

    x_to_z = LightweightModel([l_in], [l_latent])
    x_to_y = LightweightModel([l_in], [l_observed])
    z_to_x = LightweightModel([l_observed, l_latent], [l_decoder_out])
    model = Model()
    model.x_to_z = x_to_z
    model.x_to_y = x_to_y
    model.z_to_x = z_to_x


# ### Convnet

# In[34]:

if model_type == "convnet":

    ## CNN
    nb_filters=64
    size_filters=5
    nb_hidden=1000

    light.set("latent_size", latent_size)
    light.set("nb_filters", nb_filters)
    light.set("size_filters", size_filters)
    light.set("num_hidden_units", nb_hidden)

    nb_filters_encoder = nb_filters
    nb_filters_decoder = nb_filters
    size_filters_encoder = size_filters
    size_filters_decoder = size_filters

    l_in = layers.InputLayer((None, w*h))


    x_in_reshaped = layers.ReshapeLayer(l_in, ([0], 1, w, h))

    # conv1
    l_conv = layers.Conv2DLayer(
        x_in_reshaped,
        num_filters=nb_filters_encoder,
        filter_size=(size_filters_encoder, size_filters_encoder),
        nonlinearity=nonlinearities.rectify,
        #dimshuffle=True,
    )

    l_conv = layers.Pool2DLayer(
            l_conv,
            pool_size=2,
    )

    #l_conv = layers.Conv2DLayer(
    #    l_conv,
    #    num_filters=nb_filters_encoder,
    #    filter_size=(size_filters_encoder, size_filters_encoder),
    #    nonlinearity=nonlinearities.rectify,
    #    #dimshuffle=True,
    #)

    l_hid = layers.DenseLayer(
        l_conv,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )

    l_hid = layers.DenseLayer(
        l_hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )

    #code layer

    l_observed = layers.DenseLayer(l_hid,
                                   num_units=output_dim,
                                    nonlinearity=T.nnet.softmax)

    l_latent = layers.DenseLayer(l_hid,
                                 num_units=latent_size,
                                 nonlinearity=None) # linear

    hid = layers.ConcatLayer([l_latent, l_observed], axis=1)

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



    # unflatten layer

    k = 1
    hid = layers.DenseLayer(l_hid,
                            num_units=nb_filters_decoder *
                                      (w + k*(size_filters_decoder - 1)) *
                                      (h + k*(size_filters_decoder - 1)) / 4  )
    hid = layers.ReshapeLayer(hid,
                              ([0], nb_filters_decoder,
                               (w + k*(size_filters_decoder - 1))/2,
                               (h + k*(size_filters_decoder - 1))/2))
    hid = Unpool2DLayer(
            hid,
            (2, 2)
    )
    l_unconv = layers.Conv2DLayer(
        hid,
        num_filters=nb_filters,
        filter_size=(size_filters_decoder, size_filters_decoder),
        nonlinearity=nonlinearities.linear,
    )

    #l_unconv = layers.Conv2DLayer(
    #    l_unconv,
    #    num_filters=nb_filters,
    #    filter_size=(size_filters_decoder, size_filters_decoder),
    #    nonlinearity=nonlinearities.linear,
    #    pad="full"
    #)

    l_unconv_sum = SumLayer(l_unconv, axis=1)

    l_decoder_out = layers.ReshapeLayer(l_unconv_sum, ([0], w*h))
    l_decoder_out = layers.NonlinearityLayer(l_decoder_out, nonlinearities.sigmoid)


    x_to_z = LightweightModel([l_in], [l_latent])
    x_to_y = LightweightModel([l_in], [l_observed])
    z_to_x = LightweightModel([l_observed, l_latent], [l_decoder_out])
    model = Model()
    model.x_to_z = x_to_z
    model.x_to_y = x_to_y
    model.z_to_x = z_to_x


from lasagnekit.misc.draw_net import draw_to_file
draw_to_file(get_all_layers(l_decoder_out), "model.svg")

# In[35]:

def cross_correlation(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return 0.5 * ((((a.dimshuffle(0, 'x', 1) * b.dimshuffle(0, 1, 'x'))).mean(axis=0))**2).sum()

input_variables = OrderedDict()
input_variables["X"] = dict(tensor_type=T.matrix)
input_variables["y"] = dict(tensor_type=T.matrix)


functions = dict(
    encode=dict(
        get_output=lambda model, X: model.x_to_z.get_output(X)[0],
        params=["X"]
    ),
    predict=dict(
        get_output=lambda model, X: (model.x_to_y.get_output(X)[0]).argmax(axis=1),
        params=["X"]
    ),
    reconstruct=dict(
        get_output=lambda model, X: layers.get_output(l_decoder_out, X),
        params=["X"]
    ),
    get_reconstruction_error=dict(
        get_output=lambda model, X: ((X-layers.get_output(l_decoder_out, X))**2).sum(axis=1).mean(),
        params=["X"]
    ),
    get_cross_correlation=dict(
        get_output=lambda model, X: cross_correlation(model.x_to_z.get_output(X)[0],
                                                      model.x_to_y.get_output(X)[0]),
        params=["X"]
    ),
    predict_proba=dict(
        get_output=lambda model, X: model.x_to_y.get_output(X)[0],
        params=["X"]
    )

)

learning_rate = 0.0001
batch_optimizer = MyBatchOptimizer(
    verbose=1,
    max_nb_epochs=100,
    batch_size=100,
    optimization_procedure=(updates.rmsprop,
                            {"learning_rate": learning_rate})
)
light.set("learning_rate", learning_rate)
light.set("max_nb_epochs", batch_optimizer.max_nb_epochs)
light.set("batch_size", batch_optimizer.batch_size)
light.set("optimization_method", "rmsprop")
light.set("learning_rate", learning_rate)

loss_rec_coef = 5
loss_supervised_coef = 10
loss_crosscor_coef = 10

def loss_function(model, tensors):
    x_to_z, x_to_y, z_to_x = model.x_to_z, model.x_to_y, model.z_to_x
    X_batch, y_batch = tensors["X"], tensors["y"]

    z, = x_to_z.get_output(X_batch)


    y_hat, = x_to_y.get_output(X_batch)
    X_hat, = z_to_x.get_output(y_hat, z)

    loss_rec = ((X_hat - X_batch) ** 2).sum(axis=1).mean()
    loss_supervised = ((y_hat - y_batch)**2).sum(axis=1).mean()

    return  loss_rec_coef * loss_rec + loss_supervised_coef*loss_supervised + loss_crosscor_coef * cross_correlation(z, y_hat)

capsule = Capsule(
    input_variables,
    model,
    loss_function,
    functions=functions,
    batch_optimizer=batch_optimizer,
)
Z_batch = T.matrix("z_batch")
capsule.decode = theano.function([Z_batch, capsule.v_tensors["y"]],
                                  layers.get_output(l_decoder_out, {l_latent: Z_batch,
                                                    l_observed: capsule.v_tensors["y"]}))

# ## Training

try:
    capsule.fit(X=X[train], y=y[train])
except KeyboardInterrupt:
    print("interruption...")

if model_type == "convnet":
    layers_enc = get_all_layers(model.x_to_z.output_layers[0])
    layers_dec = get_all_layers(model.z_to_x.output_layers[0])
    for l in layers_enc[2], layers_dec[-4]:
        W = l.W.get_value()[:, 0]
        light.append("features", light.insert_blob(W.tolist()))
elif model_type == "fully_connected":
    layers = get_all_layers(l_decoder_out)
    for W in (layers[1].W.get_value().T, layers[-1].W.get_value()):
        W = W.reshape((W.shape[0], w, h))
        light.append("features", light.insert_blob(W.tolist()))


nb_samples_cov = 5000
light.set("nb_samples_cov", nb_samples_cov)
z = capsule.encode(X[0:nb_samples_cov])
light.set("hidfactcov", np.cov(z.T).tolist())
light.set("hidfactcorr", np.corrcoef(z.T).tolist())

def sharpness(images):
    _, gy, gx = np.gradient(images)
    return (np.sqrt(gx**2 + gy**2)).mean(axis=(1, 2))

nb = 10000
x_ = X[test][0:nb]
x_hat_ = capsule.reconstruct(x_)

x_ = x_.reshape((x_.shape[0], w, h))
x_hat_ = x_hat_.reshape((x_hat_.shape[0], w, h))

# lower = less sharp, more blurry
print("original", sharpness(x_).mean())
print("reconstruction", sharpness(x_hat_).mean())

light.set("sharpness_test", sharpness(x_).tolist())
light.set("sharpness_test_rec", sharpness(x_hat_).tolist())

#generation
nb = 10 # nb of generate image
labels = np.arange(output_dim)# labels to consider (by default, all)
std_units=2# nb of std units of values of latent dim around the mean to consider

light.set("generation_nb_images_per_label", nb)
light.set("generation_std_units", std_units)

L = latent_size

x_ = X[train][0:10000]
z_ = capsule.encode(x_)
latent_std = np.std(z_, axis=0)
latent_mean = np.mean(z_, axis=0)

sharpness_generated_per_latent = []
for latent_dim in range(latent_size):
    print("hidden factor : {0}".format(latent_dim))
    ys = np.eye(output_dim)[labels].repeat(nb, axis=0)
    seq = np.linspace(latent_mean[latent_dim] - latent_std[latent_dim]*std_units,
                      latent_mean[latent_dim] + latent_std[latent_dim]*std_units,
                      nb)
    z = np.zeros((nb, L))
    z[:, latent_dim] = seq
    z = z.repeat(len(labels), axis=0)
    z = z.reshape((nb, len(labels), L))
    z = z.transpose((1, 0, 2))
    z = z.reshape((nb*len(labels), L))
    z = z.astype(np.float32)
    ys = ys.astype(np.float32)
    c = capsule.decode(z, ys)
    c = c.reshape((c.shape[0], w, h))
    sharpness_generated_per_latent.append((sharpness(c)).tolist())
light.set("sharpness_generated_per_latent", sharpness_generated_per_latent)

# In[47]:
light.endings() # save the duration
light.store_experiment() # update the DB
light.close() # close
