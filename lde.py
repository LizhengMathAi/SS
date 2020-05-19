import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv as sparse_inv

import tensorflow as tf
import keras

import datetime


np.set_printoptions(precision=2)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


class DataGenerator:
    """
    A@x = b
    ---
    ds + ls + us = A
    x <- T@x + c
    """
    def __str__(self):
        pass

    def __init__(self, n, diag_dominant=True):
        self.n = n

        self.mat = 2 * np.random.rand(n, n).astype(np.float) - 1
        if diag_dominant:
            self.mat = self.mat.T @ self.mat + n * np.eye(n)

        self.d = np.diag(self.mat)
        self.l = np.tril(self.mat, k=-1)
        self.u = np.triu(self.mat, k=1)

        self.root = 2 * np.random.rand(n).astype(np.float) - 1
        self.rhs = self.mat@self.root

        self.t = None
        self.c = None

    def contrast_error(self, num_layers=10, display=True):
        if display:
            print('=' * 8 + "\tStart to check {}\t".format(self.__str__()) + '=' * 8)
        xk = self.c.copy()

        for k in range(num_layers - 1):
            xk = self.t@xk + self.c

        return np.linalg.norm(xk - self.root)


class JacobiDataGenerator(DataGenerator):
    """
    T = -D^{-1}@(L + U)
    c = D^{-1}@b
    """
    def __str__(self):
        return "JacobiDataGenerator"

    def __init__(self, *args, **kwargs):
        super(JacobiDataGenerator, self).__init__(*args, **kwargs)

        self.t = -np.einsum('i,ij->ij', 1 / self.d, self.l + self.u)
        self.c = self.rhs / self.d


class GSDataGenerator(DataGenerator):
    """
    T = -(D + L)^{-1}@U
    c = (D + L)^{-1}@b
    """
    def __str__(self):
        return "GSDataGenerator"

    def __init__(self, *args, **kwargs):
        super(GSDataGenerator, self).__init__(*args, **kwargs)

        inv_dl = np.linalg.inv(np.diag(self.d) + self.l)

        self.t = -inv_dl@self.u
        self.c = inv_dl@self.rhs


class SORDataGenerator(DataGenerator):
    """
    T = (D + \omega * L)^{-1}@((1 - \omega) * D - \omega * U)
    c = \omega * (D + \omega * L)^{-1}@b
    """
    def __str__(self):
        return "SORDataGenerator"

    def __init__(self, omega=1, *args, **kwargs):
        super(SORDataGenerator, self).__init__(*args, **kwargs)

        inv_dl = np.linalg.inv(np.diag(self.d) + omega * self.l)
        self.t = inv_dl@((1 - omega) * np.diag(self.d) - self.u)
        self.c = omega * inv_dl@self.rhs


class Solver:
    @classmethod
    def sequence(cls, batch_size, n, diag_dominant=True, method='jacobi'):
        while True:
            ts = []
            cs = []
            roots = []
            for _ in range(batch_size):
                if method == 'jacobi':
                    data = JacobiDataGenerator(n=n, diag_dominant=diag_dominant)
                elif method == 'GS':
                    data = GSDataGenerator(n=n, diag_dominant=diag_dominant)
                elif method == 'SOR':
                    data = SORDataGenerator(omega=1, n=n, diag_dominant=diag_dominant)
                else:
                    raise ValueError("The argument `method` is invalid!")
                ts.append(data.t)
                cs.append(data.c)
                roots.append(data.root)

            yield {"input_1": np.stack(ts), "input_2": np.stack(cs)}, np.stack(roots)

    @classmethod
    def contrast(cls, model, test_size, n, num_layers, diag_dominant=True, method='jacobi'):
        ts = []
        cs = []
        roots = []
        contrast_errors = []
        for _ in range(test_size):
            if method == 'jacobi':
                data = JacobiDataGenerator(n=n, diag_dominant=diag_dominant)
            elif method == 'GS':
                data = GSDataGenerator(n=n, diag_dominant=diag_dominant)
            elif method == 'SOR':
                data = SORDataGenerator(omega=1, n=n, diag_dominant=diag_dominant)
            else:
                raise ValueError("The argument `method` is invalid!")

            contrast_errors.append(data.contrast_error(num_layers=num_layers, display=False))

            ts.append(data.t)
            cs.append(data.c)
            roots.append(data.root)
        contrast_error = sum(contrast_errors) / test_size

        xks = model.predict(x=[np.stack(ts), np.stack(cs)])
        norms = np.sqrt(np.sum(np.square(xks - np.stack(roots)), axis=1))
        error = np.mean(norms)

        return error, contrast_error

    @classmethod
    def update(cls, ts, cs, regularizer_rate):
        n = ts.get_shape().as_list()[-1]
        t_noise = tf.Variable(initial_value=np.zeros(shape=(1, n, n)), dtype=tf.float32)
        c_noise = tf.Variable(initial_value=np.zeros(shape=(1, n)), dtype=tf.float32)

        def update_formula(xks):
            mul = (ts + t_noise) * tf.reshape(xks, (-1, 1, xks.get_shape().as_list()[-1]))
            return tf.reduce_sum(mul, axis=2) + (cs + c_noise)

        layer = keras.layers.Lambda(update_formula)
        layer.losses.append(regularizer_rate * tf.reduce_sum(tf.square(t_noise)))
        layer.losses.append(regularizer_rate * tf.reduce_sum(tf.square(c_noise)))
        return layer

    @classmethod
    def model(cls, n, num_layers, regularizer_rate):
        input_ts = keras.layers.Input(shape=[n, n])
        input_cs = keras.layers.Input(shape=[n])

        xks = cls.update(input_ts, input_cs, regularizer_rate)(input_cs)
        for _ in range(num_layers - 1):
            xks = cls.update(input_ts, input_cs, regularizer_rate)(xks)

        return keras.models.Model(inputs=[input_ts, input_cs], outputs=xks)

    @classmethod
    def example(cls, n=1024, num_layers=4, regularizer_rate=0.001, batch_size=1024, global_step=512, method='jacobi'):

        model = cls.model(n, num_layers, regularizer_rate)
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mean_squared_error', metrics=['mse'])

        history = model.fit_generator(
            cls.sequence(batch_size=batch_size, n=n, diag_dominant=True, method=method), steps_per_epoch=global_step)

        error, contrast_error = cls.contrast(model, batch_size, n, num_layers, diag_dominant=True, method=method)

        columns = ["date", "n", "num_layers", "regularizer_rate", "batch_size", "global_step", "method", "mse",
                   "contrast_mse"]
        values = [
            str(datetime.datetime.now()),
            n, num_layers, regularizer_rate, batch_size, global_step, method,
            "{:.2e}".format(error), "{:.2e}".format(contrast_error)
        ]
        print("error:{:.2e}".format(error), ",\tcontrast_error:{:.2e}".format(contrast_error))

        return columns, values


# ===================================================================
tf.app.flags.DEFINE_integer("n", 64, "dimension.")
tf.app.flags.DEFINE_integer("num_layers", 4, "models layers.")
tf.app.flags.DEFINE_integer("global_step", 16, "global step.")
tf.app.flags.DEFINE_string("method", "GS", "iteration method.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    Solver.example(n=FLAGS.n, num_layers=FLAGS.num_layers, global_step=FLAGS.global_step, method=FLAGS.method)


if __name__ == '__main__':
    tf.app.run(main)