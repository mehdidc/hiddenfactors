import numpy as np
import theano.tensor as T
import os

import theano

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from skimage.io import imsave

from lasagne import layers, updates
from lasagnekit.datasets.mnist import MNIST
from lasagnekit.datasets.fonts import Fonts
from lasagnekit.datasets.chairs import Chairs
from lasagnekit.nnet.capsule import Capsule
from lasagnekit.easy import build_batch_iterator
from lasagnekit.easy import BatchOptimizer


from collections import OrderedDict

from invoke import task

from model import build_convnet
from helpers import cross_correlation

datasets = dict(
    mnist=lambda: MNIST(),
    fonts=lambda: Fonts(),
    chairs=lambda: Chairs(size=(64, 64), nb=32),
)


class MyBatchOptimizer(BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(MyBatchOptimizer, self).iter_update(
                epoch,
                nb_batches,
                iter_update_batch)
        return status


@task
def train(dataset="mnist"):

    data = datasets[dataset]()
    data.load()
    if hasattr(data, "online") and data.online is True:
        online = True
    else:
        online = False

    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.matrix)
    input_variables["y"] = dict(tensor_type=T.matrix)

    o = data.output_dim
    w, h = data.img_dim[0], data.img_dim[1]
    if len(data.img_dim) == 3:
        c = data.img_dim[2]
    else:
        c = 1

    def preprocess(X):
        X = 1 - data.X.transpose((0, 3, 1, 2))
        X = X.reshape((X.shape[0], -1))
        return X

    def deprocess(X):
        X = 1 - X
        X = X * 255
        X = X.astype(np.uint8)
        X = X.reshape((X.shape[0], c, w, h))
        X = X.transpose((0, 2, 3, 1))
        return X

    model = build_convnet(
            nb_filters=96, size_filters=5,
            nb_hidden=1000,
            w=w, h=h,
            c=c, output_dim=o, latent_size=100)
    l_rec = model.l_reconstruction

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
            get_output=lambda model, X: layers.get_output(l_rec, X),
            params=["X"]
        ),
        get_reconstruction_error=dict(
            get_output=lambda model, X: ((X-layers.get_output(l_rec, X))**2).sum(axis=1).mean(),
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
        max_nb_epochs=300000,
        batch_size=128,
        optimization_procedure=(updates.rmsprop,
                                {"learning_rate": learning_rate})
    )

    loss_rec_coef = 1
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

        return (loss_rec_coef * loss_rec +
                loss_supervised_coef * loss_supervised +
                loss_crosscor_coef * cross_correlation(z, y_hat))

    def transform(batch_index, batch_slice, tensors):
        t = OrderedDict()
        if online is True:
            data.load()
            X = preprocess(data.X)
            y = np.array(data.y)
            y = label_binarize(y, np.arange(data.output_dim))
            y = y.astype(np.float32)
            t["X"] = X
            t["y"] = y
        else:
            t["X"] = tensors["X"][batch_slice]
            t["y"] = tensors["y"][batch_slice]

        h = capsule.encode(t["X"])
        n = capsule.n
        capsule.h_avg = (n * capsule.h_avg + h.mean(axis=0)) / (n + 1)
        capsule.h_sqr_avg = (n * capsule.h_sqr_avg + (h**2).mean(axis=0)) / (n + 1)
        capsule.n += 1

        if len(batch_optimizer.stats) % 100 == 0:
            cnt = len(batch_optimizer.stats) / 10

            # reconstructions
            try:
                os.mkdir("rec/{}".format(cnt))
            except OSError:
                pass
            x_rec = capsule.reconstruct(t["X"])
            x_rec = deprocess(x_rec)
            # x_rec = 255 - x_rec # for better visualization
            for i, x in enumerate(x_rec):
                imsave("rec/{}/{}.png".format(cnt, i), x)

            # generations
            try:
                os.mkdir("gen/{}".format(cnt))
            except OSError:
                pass
            h_avg = capsule.h_avg
            h_var = capsule.h_sqr_avg - capsule.h_avg ** 2
            h_std = np.sqrt(h_var)
            nb = 20
            D = h_avg.shape[0]
            h_gen = h_avg[None, None, :] * np.ones((D, nb, 1))
            for d in range(D):
                vals = np.linspace(h_avg[d] - 3*h_std[d],
                                   h_avg[d] + 3*h_std[d], nb)
                h_gen[d, :, d] = vals
            h_gen = h_gen.reshape((D * nb, D))
            #h_gen = np.random.multivariate_normal(
            #        h_avg, np.diag(h_var), size=10)
            h_gen = h_gen.astype(np.float32)
            y = np.zeros((h_gen.shape[0], data.output_dim))
            y = y.astype(np.float32)
            x_rec = capsule.decode(h_gen, y)
            x_rec = deprocess(x_rec)
            # x_rec = 255 - x_rec  # for better visualization
            for i, x in enumerate(x_rec):
                imsave("gen/{}/{}.png".format(cnt, i), x)
        return t

    batch_iterator = build_batch_iterator(transform)

    capsule = Capsule(
        input_variables,
        model,
        loss_function,
        functions=functions,
        batch_optimizer=batch_optimizer,
        batch_iterator=batch_iterator
    )
    capsule.n = 0  # used to compute running average
    capsule.h_avg = 0
    capsule.h_sqr_avg = 0

    Z_batch = T.matrix("Z_batch")

    rec = layers.get_output(
            model.l_reconstruction,
            {model.l_latent: Z_batch,
             model.l_observed: capsule.v_tensors["y"]})
    capsule.decode = theano.function(
            [Z_batch, capsule.v_tensors["y"]],
            rec)

    if online is True:
        dummy = np.zeros((1, 1))
        X = dummy
        y = dummy
    else:
        X = data.X
        y = data.y
        y = label_binarize(y, np.arange(data.output_dim))
        y = y.astype(np.float32)
        X, y = shuffle(X, y, random_state=2)
    try:
        capsule.fit(X=X, y=y)
    except KeyboardInterrupt:
        print("interruption...")
