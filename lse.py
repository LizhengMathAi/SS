import numpy as np
from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import spsolve
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

        self.mat, self.rhs, _ = self.poisson(w, n)

        self.d = np.diag(self.mat)
        self.l = np.tril(self.mat, k=-1)
        self.u = np.triu(self.mat, k=1)

        self.t = None
        self.c = None

    @classmethod
    def poisson(cls, w, n, num_refine=3):
        """
        -\Delta u = f, \boldsymbol{x} \in \Omega
        u = g, \boldsymbol{x} \in \partial \Omega
        """
        def func_u(x):
            return np.cos(w * x[0] + (1 - w) * x[1])

        def func_f(x):
            return -(w ** 2 + (1 - w) ** 2) * np.cos(w * x[0] + (1 - w) * x[1])

        func_g = func_u

        # Start to solve
        mesh = fem_utils.SquareMesh(n=n)

        inner_nn = mesh.inner_node_ids.__len__()
        bound_nn = mesh.bound_node_ids.__len__()

        # compute mat of (\nabla \phi_{\boldsymbol{u}}, \nabla \phi_{\boldsymbol{u}})
        # P_1 X P_1
        gram_tensor = mesh.gram_grad_p1()
        mat = mesh.node_mul_node(gram_tensor)

        mat_1 = mat[mesh.inner_node_ids, mesh.inner_node_ids]
        mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

        mat_2 = mat[mesh.inner_node_ids, mesh.bound_node_ids]
        mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, bound_nn))

        bound_vertices = mesh.vertices[mesh.bound_node_ids]
        rhs_1 = func_g(bound_vertices.T)

        # compute mat of (\phi_p, f)
        integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
        rhs = mesh.node_mul_func(integer_tensor)
        rhs_2 = rhs[mesh.inner_node_ids]

        mat = mat_1
        rhs = rhs_2 - mat_2@rhs_1
        root = func_u(mesh.vertices[mesh.inner_node_ids].T)

        return mat.toarray(), rhs, root

    def residual_norm(self, num_layers=10, display=True):
        if display:
            print('=' * 8 + "\tStart to check {}\t".format(self.__str__()) + '=' * 8)
        xk = self.c.copy()

        for k in range(num_layers - 1):
            xk = self.t@xk + self.c

        return np.linalg.norm(self.mat@xk - self.rhs)


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
            t_flatten = tf.reshape(ts, shape=[-1, (self.n - 1) ** 4])
            x = tf.concat([t_flatten, cs], axis=1)

        for i in range(self.num_layers):
            with tf.name_scope("hidden_{}".format(i)):
                x = self.block(x, 256)

        weights_init = tf.zeros(shape=[x.get_shape().as_list()[-1], (self.n - 1) ** 2], dtype=tf.float32)
        weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

        return tf.matmul(x, weights)

    # ---------------------- data generator ----------------------
    def sequence(self, h, batch_size, version):

        sample_size = int(1 / h)

        if version == "jacobi":
            def train_sequence():
                while True:
                    data = JacobiDataGenerator(w=np.random.randint(sample_size) / sample_size, n=self.n)
                    yield (data.mat, data.rhs, data.t, data.c)

            def test_sequence():
                while True:
                    data = JacobiDataGenerator(w=np.random.rand(), n=self.n)
                    yield (data.mat, data.rhs, data.t, data.c)
        elif version == "GS":
            def train_sequence():
                while True:
                    data = GSDataGenerator(w=np.random.randint(sample_size) / sample_size, n=self.n)
                    yield (data.mat, data.rhs, data.t, data.c)

            def test_sequence():
                while True:
                    data = GSDataGenerator(w=np.random.rand(), n=self.n)
                    yield (data.mat, data.rhs, data.t, data.c)
        elif version == "SOR":
            def train_sequence():
                while True:
                    data = SORDataGenerator(w=np.random.randint(sample_size) / sample_size, n=self.n)
                    yield (data.mat, data.rhs, data.t, data.c)

            def test_sequence():
                while True:
                    data = SORDataGenerator(w=np.random.rand(), n=self.n)
                    yield (data.mat, data.rhs, data.t, data.c)
        else:
            raise ValueError

        train_data = tf.data.Dataset.from_generator(train_sequence, (tf.float32, tf.float32, tf.float32, tf.float32))
        train_data = train_data.batch(batch_size=batch_size)
        train_samples = train_data.make_one_shot_iterator().get_next()

        test_data = tf.data.Dataset.from_generator(test_sequence, (tf.float32, tf.float32, tf.float32, tf.float32))
        test_data = test_data.batch(batch_size=batch_size)
        test_samples = test_data.make_one_shot_iterator().get_next()

        return train_samples, test_samples

    # ---------------------- train & test ----------------------
    def __init__(self, n, num_layers):
        self.n = n
        self.num_layers = num_layers

        self.ts = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2, (self.n - 1) ** 2], name="t")
        self.cs = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2], name="c")

        self.mats = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2, (self.n - 1) ** 2], name="mat")
        self.rhses = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2], name="rhs")

        xks = tf.identity(self.cs)
        for _ in range(num_layers - 1):
            xks = tf.reduce_sum(self.ts * tf.reshape(xks, [-1, 1, (self.n - 1) ** 2]), axis=2) + self.cs

        with tf.name_scope("remainder"):
            remainder = self.remainder(self.ts, self.cs)

        self.output_roots = xks + remainder

        with tf.name_scope("loss"):
            residual = tf.reduce_sum(
                self.mats * tf.reshape(self.output_roots, [-1, 1, (self.n - 1) ** 2]), axis=2) - self.rhses
            square_norm = tf.reduce_sum(tf.square(residual), axis=1)
            self.loss = tf.reduce_mean(tf.sqrt(square_norm))

        with tf.name_scope("org_loss"):
            residual = tf.reduce_sum(self.mats * tf.reshape(xks, [-1, 1, (self.n - 1) ** 2]), axis=2) - self.rhses
            square_norm = tf.reduce_sum(tf.square(residual), axis=1)
            self.org_loss = tf.reduce_mean(tf.sqrt(square_norm))

        with tf.name_scope("train_op"):
            count = sum([sum(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print("count", count)
            regularizer = 1e-3 / count * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss + regularizer)

    def train(self, batch_size=16, global_step=1024, version='jacobi'):
        train_samples, test_samples = self.sequence(h=1e-3, batch_size=batch_size, version=version)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(global_step):
                ts, cs, mats, rhses = sess.run(train_samples)
                _, loss_val, org_loss_val = sess.run(
                    [self.train_op, self.loss, self.org_loss],
                    feed_dict={self.ts: ts, self.cs: cs, self.mats: mats, self.rhses: rhses})
                print("\r{:7d}/{}\t\tloss:{:.4e}\t\torg_loss:{:.4e}".format(
                    batch_size * i, batch_size * global_step, loss_val, org_loss_val), end='')

            print()
            ts, cs, mats, rhses = sess.run(test_samples)

            xks = cs.copy()
            for k in range(self.num_layers - 1):
                xks = np.einsum("nij,nj->ni", ts, xks) + cs
            residual = np.einsum("nij,nj->ni", mats, xks) - rhses
            error = np.sqrt(np.sum(np.square(residual), axis=1)).mean()

            contrast_error = sess.run(
                self.loss, feed_dict={self.ts: ts, self.cs: cs, self.mats: mats, self.rhses: rhses})

            print("error:{:.4e}".format(error), ",\tcontrast_error:{:.4e}".format(contrast_error))


# ===================================================================
tf.app.flags.DEFINE_integer("n", 8, "dimension.")
tf.app.flags.DEFINE_integer("l", 4, "number of layers.")
tf.app.flags.DEFINE_integer("b", 16, "batch size.")
tf.app.flags.DEFINE_integer("g", 1024, "global step.")
tf.app.flags.DEFINE_string("v", "GS", "iteration method.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    model = TensorflowSolver(n=FLAGS.n, num_layers=FLAGS.l)
    model.train(batch_size=FLAGS.b, global_step=FLAGS.g, version=FLAGS.v)


if __name__ == '__main__':
    tf.app.run(main)
