import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import fem_utils

import tensorflow as tf
import keras

import datetime


np.set_printoptions(precision=2)


class DataGenerator:
    """
    A@x = b
    ---
    ds + ls + us = A
    x <- T@x + c
    """
    def __str__(self):
        pass

    def __init__(self, w, n):

        self.mat, self.rhs, self.root = self.build(w, n)

        self.d = np.diag(self.mat)
        self.l = np.tril(self.mat, k=-1)
        self.u = np.triu(self.mat, k=1)
        self.rhs = self.mat@self.root

        self.t = None
        self.c = None

    @classmethod
    def build(cls, w, n):
        mat_0 = 2 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
        mat_1 = 3 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)

        root_0 = np.random.rand(n)
        root_1 = np.random.rand(n)

        rhs_0 = mat_0@root_0
        rhs_1 = mat_1@root_1

        mat = w * mat_0 + (1 - w) * mat_1
        root = w * root_0 + (1 - w) * root_1
        rhs = w * rhs_0 + (1 - w) * rhs_1

        return mat, rhs, root

    def error(self, num_layers=10, display=True):
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

        self.omega = omega
        inv_dl = np.linalg.inv(np.diag(self.d) + self.omega * self.l)
        self.t = inv_dl@((1 - self.omega) * np.diag(self.d) - self.u)
        self.c = self.omega * inv_dl@self.rhs


class TensorflowSolver:
    # ---------------------- neural network of remainder ----------------------
    @staticmethod
    def bn_layer(input_tensor):
        size = input_tensor.get_shape().as_list()[-1]

        mean, variance = tf.nn.moments(input_tensor, axes=[0])
        beta = tf.Variable(initial_value=tf.zeros(size, dtype=tf.float32), name="beta")
        gamma = tf.Variable(initial_value=tf.ones(size, dtype=tf.float32), name="gamma")

        return tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, 0.001)

    @staticmethod
    def fc_layer(input_tensor, out_channels):
        weights_shape = [input_tensor.get_shape().as_list()[-1], out_channels]

        weights_init = tf.truncated_normal(weights_shape, stddev=np.sqrt(2 / (weights_shape[0] + weights_shape[1])))
        # weights_init = tf.zeros(weights_shape, dtype=tf.float32)
        weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

        mul_tensor = tf.matmul(input_tensor, weights)

        bias = tf.Variable(initial_value=tf.zeros((weights_shape[1]), dtype=tf.float32), name="bias")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(bias)))
        return mul_tensor + bias

    def block(self, tensor, output_channels):
        channels = tensor.get_shape().as_list()[-1]

        tensor_1 = self.fc_layer(tensor, channels)
        tensor_1 = self.bn_layer(tensor_1)
        tensor_1 = tf.nn.relu(tensor_1)

        tensor_2 = self.fc_layer(tensor_1, channels)
        tensor_2 = self.bn_layer(tensor_2)
        tensor_2 = tf.nn.relu(tensor_2)

        tensor = tensor + tensor_2

        return self.fc_layer(tensor, output_channels)

    def remainder(self, ts, cs):

        with tf.name_scope("pre_processing"):
            t_flatten = tf.reshape(ts, shape=[-1, self.n ** 2])
            x = tf.concat([t_flatten, cs], axis=1)

        for i in range(8):
            with tf.name_scope("hidden_{}".format(i)):
                x = self.block(x, 64)

        weights_init = tf.zeros(shape=[x.get_shape().as_list()[-1], self.n], dtype=tf.float32)
        weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

        return tf.matmul(x, weights)

    # ---------------------- data generator ----------------------
    def sequence(self, batch_size, version):
        sample_size = 1000

        if version == "jacobi":
            def train_sequence():
                while True:
                    data = JacobiDataGenerator(w=np.random.randint(sample_size) / sample_size, n=self.n)
                    yield (data.t, data.c, data.root)

            def test_sequence():
                while True:
                    data = JacobiDataGenerator(w=np.random.rand(), n=self.n)
                    yield (data.t, data.c, data.root)
        elif version == "GS":
            def train_sequence():
                while True:
                    data = GSDataGenerator(w=np.random.randint(sample_size) / sample_size, n=self.n)
                    yield (data.t, data.c, data.root)

            def test_sequence():
                while True:
                    data = GSDataGenerator(w=np.random.rand(), n=self.n)
                    yield (data.t, data.c, data.root)
        elif version == "SOR":
            def train_sequence():
                while True:
                    data = SORDataGenerator(w=np.random.randint(sample_size) / sample_size, n=self.n)
                    yield (data.t, data.c, data.root)

            def test_sequence():
                while True:
                    data = SORDataGenerator(w=np.random.rand(), n=self.n)
                    yield (data.t, data.c, data.root)
        else:
            raise ValueError

        train_data = tf.data.Dataset.from_generator(train_sequence, (tf.float32, tf.float32, tf.float32))
        train_data = train_data.batch(batch_size=batch_size)
        train_samples = train_data.make_one_shot_iterator().get_next()

        test_data = tf.data.Dataset.from_generator(test_sequence, (tf.float32, tf.float32, tf.float32))
        test_data = test_data.batch(batch_size=batch_size)
        test_samples = test_data.make_one_shot_iterator().get_next()

        return train_samples, test_samples

    # ---------------------- train & test ----------------------
    def __init__(self, n, num_layers):
        self.n = n
        self.num_layers = num_layers

        self.input_ts = tf.placeholder(tf.float32, shape=[None, self.n, self.n], name="t")
        self.input_cs = tf.placeholder(tf.float32, shape=[None, self.n], name="c")
        self.input_roots = tf.placeholder(tf.float32, shape=[None, self.n], name="root")

        xks = tf.identity(self.input_cs)
        for _ in range(num_layers - 1):
            xks = tf.reduce_sum(self.input_ts * tf.reshape(xks, [-1, 1, self.n]), axis=2) + self.input_cs

        with tf.name_scope("remainder"):
            remainder = self.remainder(self.input_ts, self.input_cs)

        self.output_roots = xks + remainder

        with tf.name_scope("loss"):
            square_norm = tf.reduce_sum(tf.square(self.input_roots - self.output_roots), axis=1)
            self.loss = tf.reduce_mean(tf.sqrt(square_norm))

        with tf.name_scope("org_loss"):
            square_norm = tf.reduce_sum(tf.square(self.input_roots - xks), axis=1)
            self.org_loss = tf.reduce_mean(tf.sqrt(square_norm))

        with tf.name_scope("train_op"):
            regularizer = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # self.train_op = tf.train.GradientDescentOptimizer(1e-1).minimize(self.loss + 1e-6 * regularizer)
            self.train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(self.loss)

    def train(self, batch_size=16, global_step=1024, version='jacobi'):
        train_samples, test_samples = self.sequence(batch_size=batch_size, version=version)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(global_step):
                ts, cs, roots = sess.run(train_samples)
                sess.run(self.train_op, feed_dict={self.input_ts: ts, self.input_cs: cs, self.input_roots: roots})
                if i % 16 == 0:
                    print('\r', end='')
                    loss_val, org_loss_val = sess.run(
                        [self.loss, self.org_loss],
                        feed_dict={self.input_ts: ts, self.input_cs: cs, self.input_roots: roots})
                    print("{:7d}/{}\t\tloss:{:.4e}\t\torg_loss:{:.4e}".format(
                        batch_size * i, batch_size * global_step, loss_val, org_loss_val), end='')

            print()
            ts, cs, roots = sess.run(test_samples)

            xks = cs.copy()
            for k in range(self.num_layers - 1):
                xks = np.einsum("nij,nj->ni", ts, xks) + cs
            error = np.sqrt(np.sum(np.square(xks - roots), axis=1)).mean()

            contrast_error = sess.run(
                self.loss, feed_dict={self.input_ts: ts, self.input_cs: cs, self.input_roots: roots})

            print("error:{:.4e}".format(error), ",\tcontrast_error:{:.4e}".format(contrast_error))


# ===================================================================
tf.app.flags.DEFINE_integer("n", 8, "dimension.")
tf.app.flags.DEFINE_integer("l", 4, "number of layers.")
tf.app.flags.DEFINE_integer("b", 256, "batch size.")
tf.app.flags.DEFINE_integer("g", 1024, "global step.")
tf.app.flags.DEFINE_string("v", "GS", "iteration method.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    model = TensorflowSolver(n=FLAGS.n, num_layers=FLAGS.l)
    model.train(batch_size=FLAGS.b, global_step=FLAGS.g, version=FLAGS.v)


if __name__ == '__main__':
    tf.app.run(main)
