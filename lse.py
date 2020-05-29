import numpy as np
from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import fem_utils

import tensorflow as tf


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

        self.mat, self.rhs = self.pde(w, n)

        ind = [i for i in range(self.mat.data.__len__()) if self.mat.row[i] == self.mat.col[i]]
        self.d = coo_matrix((self.mat.data[ind], (self.mat.row[ind], self.mat.col[ind])), shape=self.mat.shape)

        ind = [i for i in range(self.mat.data.__len__()) if self.mat.row[i] > self.mat.col[i]]
        self.l = coo_matrix((self.mat.data[ind], (self.mat.row[ind], self.mat.col[ind])), shape=self.mat.shape)

        ind = [i for i in range(self.mat.data.__len__()) if self.mat.row[i] < self.mat.col[i]]
        self.u = coo_matrix((self.mat.data[ind], (self.mat.row[ind], self.mat.col[ind])), shape=self.mat.shape)

        self.t = None
        self.c = None

    @classmethod
    def pde(cls, w, n, num_refine=3):
        pass

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

        inv_d = inv(self.d)
        self.t = -inv_d@(self.l + self.u)
        self.c = inv_d@self.rhs


class GSDataGenerator(DataGenerator):
    """
    T = -(D + L)^{-1}@U
    c = (D + L)^{-1}@b
    """
    def __str__(self):
        return "GSDataGenerator"

    def __init__(self, *args, **kwargs):
        super(GSDataGenerator, self).__init__(*args, **kwargs)

        inv_dl = inv(self.d + self.l)
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
        inv_dl = inv(self.d + self.omega * self.l)
        self.t = inv_dl@((1 - self.omega) * self.d - self.u)
        self.c = self.omega * inv_dl@self.rhs


class SparseSolver:
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

    def remainder(self, feature):

        with tf.name_scope("pre_processing"):
            x = tf.reshape(feature, shape=[1, -1])

        for i in range(self.num_layers):
            with tf.name_scope("hidden_{}".format(i)):
                x = self.block(x, 256)

        weights_init = tf.zeros(shape=[x.get_shape().as_list()[-1], (self.n - 1) ** 2], dtype=tf.float32)
        weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))
        x = tf.matmul(x, weights)

        with tf.name_scope("post_processing"):
            x = tf.reshape(x, shape=[-1, 1])
        return x

    # ---------------------- data generator ----------------------
    _data_generator = None

    @property
    def data_generator(self):
        if self._data_generator is None:
            if self.version == "jacobi":
                self._data_generator = JacobiDataGenerator
            elif self.version == "GS":
                self._data_generator = GSDataGenerator
            elif self.version == "SOR":
                self._data_generator = SORDataGenerator
            else:
                raise ValueError
        return self._data_generator

    @property
    def train_samples(self, h=1e-3):

        sample_size = int(1 / h)

        data = self.data_generator(w=np.random.randint(sample_size) / sample_size, n=self.n)

        mat = tf.SparseTensorValue(np.stack(data.mat.nonzero(), axis=1), data.mat.data, data.mat.shape)
        rhs = data.rhs
        t = tf.SparseTensorValue(np.stack(data.t.nonzero(), axis=1), data.t.data, data.t.shape)
        c = data.c
        feature = np.concatenate([data.mat.data, data.rhs])

        return mat, rhs, t, c, feature

    @property
    def test_samples(self):
        data = self.data_generator(w=np.random.rand(), n=self.n)

        mat = tf.SparseTensorValue(np.stack(data.mat.nonzero(), axis=1), data.mat.data, data.mat.shape)
        rhs = data.rhs
        t = tf.SparseTensorValue(np.stack(data.t.nonzero(), axis=1), data.t.data, data.t.shape)
        c = data.c
        feature = np.concatenate([data.mat.data, data.rhs])

        return mat, rhs, t, c, feature

    # ---------------------- train & test ----------------------
    def __init__(self, n, num_layers, version='jacobi'):
        self.n = n
        self.num_layers = num_layers
        self.version = version

        dim = (self.n - 1) ** 2

        self.mat = tf.sparse_placeholder(tf.float32, name="mat")
        self.rhs = tf.placeholder(tf.float32, shape=(dim, ), name="rhs")

        self.t = tf.sparse_placeholder(tf.float32, name="t")
        self.c = tf.placeholder(tf.float32, shape=(dim, ), name="c")

        size = self.data_generator(w=np.random.rand(), n=self.n).mat.data.__len__()
        self.feature = tf.placeholder(tf.float32, shape=(size + dim, ), name="flatten_feature")

        xk = tf.reshape(self.c, shape=(-1, 1))
        for _ in range(num_layers - 1):
            xk = tf.sparse_tensor_dense_matmul(self.t, xk) + tf.reshape(self.c, [-1, 1])

        with tf.name_scope("remainder"):
            remainder = self.remainder(self.feature)

        self.output_roots = xk + remainder

        with tf.name_scope("loss"):
            residual = tf.sparse_tensor_dense_matmul(self.mat, self.output_roots) - tf.reshape(self.rhs, [-1, 1])
            self.loss = tf.sqrt(tf.reduce_sum(tf.square(residual)))
            tf.linalg.norm

        with tf.name_scope("org_loss"):
            residual = tf.sparse_tensor_dense_matmul(self.mat, xk) - tf.reshape(self.rhs, [-1, 1])
            self.org_loss = tf.sqrt(tf.reduce_sum(tf.square(residual)))

        with tf.name_scope("train_op"):
            count = sum([sum(v.get_shape().as_list()) for v in tf.trainable_variables()])
            regularizer = 1e-3 / count * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss + regularizer)

    def train(self, global_step=1024):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # ------------ train ------------
            for i in range(global_step):
                mat, rhs, t, c, feature = self.train_samples
                _, loss_val, org_loss_val = sess.run(
                    [self.train_op, self.loss, self.org_loss],
                    feed_dict={self.mat: mat, self.rhs: rhs, self.t: t, self.c: c, self.feature: feature})
                print("\r{:7d}/{}\t\tloss:{:.4e}\t\torg_loss:{:.4e}".format(
                    i, global_step, loss_val, org_loss_val), end='')
            print()

            # ------------ test ------------
            mat, rhs, t, c, feature = self.test_samples
            mat_array = coo_matrix((mat.values, mat.indices.T), shape=mat.dense_shape)
            t_array = coo_matrix((t.values, t.indices.T), shape=t.dense_shape)

            xk = c.copy()
            for k in range(self.num_layers - 1):
                xk = t_array @ xk + c
            residual = mat_array @ xk - rhs
            error = np.sqrt(np.sum(np.square(residual))).mean()

            contrast_error = sess.run(
                self.loss,
                feed_dict={self.mat: mat, self.rhs: rhs, self.t: t, self.c: c, self.feature: feature})

            print("error:{:.4e}".format(error), ",\tcontrast_error:{:.4e}".format(contrast_error))


class PoissonSolver(SparseSolver):

    @property
    def data_generator(self):
        if self._data_generator is None:

            # ------------ Redefine the `pde` function. ------------
            def poisson(w, n, num_refine):
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
                # P_1 X f
                integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
                rhs = mesh.node_mul_func(integer_tensor)
                rhs_2 = rhs[mesh.inner_node_ids]

                mat = mat_1
                rhs = rhs_2 - mat_2 @ rhs_1

                return mat, rhs

            # ------------ Define the valid `DataGenerator`. ------------
            if self.version == "jacobi":
                class PoissonDataGenerator(JacobiDataGenerator):
                    @classmethod
                    def pde(cls, w, n, num_refine=3):
                        return poisson(w, n, num_refine=num_refine)
            elif self.version == "GS":
                class PoissonDataGenerator(GSDataGenerator):
                    @classmethod
                    def pde(cls, w, n, num_refine=3):
                        return poisson(w, n, num_refine=num_refine)
            elif self.version == "SOR":
                class PoissonDataGenerator(SORDataGenerator):
                    @classmethod
                    def pde(cls, w, n, num_refine=3):
                        return poisson(w, n, num_refine=num_refine)
            else:
                raise ValueError
            self._data_generator = PoissonDataGenerator
        return self._data_generator


# ===================================================================
tf.app.flags.DEFINE_integer("n", 8, "dimension.")
tf.app.flags.DEFINE_integer("l", 16, "number of layers.")
tf.app.flags.DEFINE_integer("g", 1024, "global step.")
tf.app.flags.DEFINE_string("v", "GS", "iteration method.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    model = PoissonSolver(n=FLAGS.n, num_layers=FLAGS.l, version=FLAGS.v)
    model.train(global_step=FLAGS.g)


if __name__ == '__main__':
    tf.app.run(main)
