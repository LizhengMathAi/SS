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

        self.mat, self.rhs, self.numerical_roots, self.root = self.poisson(w, n)

        self.d = np.diag(self.mat)
        self.l = np.tril(self.mat, k=-1)
        self.u = np.triu(self.mat, k=1)
        self.rhs = self.mat@self.root

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
        numerical_root = spsolve(mat, rhs)
        root = func_u(mesh.vertices[mesh.inner_node_ids].T)

        return mat.toarray(), rhs, numerical_root, root

    def error(self, num_layers=10, display=True):
        if display:
            print('=' * 8 + "\tStart to check {}\t".format(self.__str__()) + '=' * 8)
        xk = self.c.copy()

        for k in range(num_layers - 1):
            xk = self.t@xk + self.c

        return np.linalg.norm(xk - self.root)

    @classmethod
    def plot(cls, n=8):

        def genuine_solution(variables):
            mat = np.reshape(variables[:n ** 2], [n, n])
            rhs = variables[n ** 2:]
            return np.linalg.solve(mat, rhs)

        def numerical_solution(variables, layers):
            mat = np.reshape(variables[:n ** 2], [n, n])
            rhs = variables[n ** 2:]
            roots = np.linalg.solve(mat, rhs)

            class Data(cls):
                def build(self, *args, **kwargs):
                    return mat, roots

            data = Data(n=n)
            xk = data.c.copy()
            for k in range(layers - 1):
                xk = data.t @ xk + data.c
            return xk

        input_weight_0 = np.random.rand(n*(n+1))
        input_weight_1 = np.random.rand(n*(n+1))
        output_weight = np.random.rand(n)

        x = np.linspace(0.5, 1, 25)
        y = np.linspace(0.5, 1, 25)
        xx, yy = np.meshgrid(x, y)
        xv = np.reshape(xx, -1)
        yv = np.reshape(yy, -1)

        genuine_val = []
        numerical_val_2 = []
        numerical_val_4 = []
        numerical_val_6 = []
        for x, y in zip(xv, yv):
            val = genuine_solution(x * input_weight_0 + y * input_weight_1)
            genuine_val.append(np.inner(output_weight, val))
            val = numerical_solution(x * input_weight_0 + y * input_weight_1, layers=2)
            numerical_val_2.append(np.inner(output_weight, val))
            val = numerical_solution(x * input_weight_0 + y * input_weight_1, layers=4)
            numerical_val_4.append(np.inner(output_weight, val))
            val = numerical_solution(x * input_weight_0 + y * input_weight_1, layers=6)
            numerical_val_6.append(np.inner(output_weight, val))
        genuine_val = np.reshape(genuine_val, [25, 25])
        numerical_val_2 = np.reshape(numerical_val_2, [25, 25])
        numerical_val_4 = np.reshape(numerical_val_4, [25, 25])
        numerical_val_6 = np.reshape(numerical_val_6, [25, 25])

        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 12))

        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.set_title(r"genuine solution: $z(x, y) = \omega^T (x * A_1 + y * A_2)^{-1} (x * b_1 + y * b_2)$")
        ax.plot_surface(xx, yy, genuine_val)

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_title(r"numerical solution (iter=2)")
        ax.plot_surface(xx, yy, numerical_val_2)

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_title(r"numerical solution (iter=4)")
        ax.plot_surface(xx, yy, numerical_val_4)

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.set_title(r"numerical solution (iter=6)")
        ax.plot_surface(xx, yy, numerical_val_6)

        plt.show()


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


def bn_layer(input_tensor):
    size = input_tensor.get_shape().as_list()[-1]

    mean, variance = tf.nn.moments(input_tensor, axes=[0])
    beta = tf.Variable(initial_value=tf.zeros(size, dtype=tf.float32), name="beta")
    gamma = tf.Variable(initial_value=tf.ones(size, dtype=tf.float32), name="gamma")

    return tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, 0.001)


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


class TensorflowSolver:

    def remainder(self, ts, cs):

        def block(tensor, output_channels):
            channels = tensor.get_shape().as_list()[-1]

            tensor_1 = fc_layer(tensor, channels)
            tensor_1 = bn_layer(tensor_1)
            tensor_1 = tf.nn.relu(tensor_1)

            tensor_2 = fc_layer(tensor_1, channels)
            tensor_2 = bn_layer(tensor_2)
            tensor_2 = tf.nn.relu(tensor_2)

            tensor = tensor + tensor_2

            return fc_layer(tensor, output_channels)

        with tf.name_scope("pre_processing"):
            t_flatten = tf.reshape(ts, shape=[-1, (self.n - 1) ** 4])
            x = tf.concat([t_flatten, cs], axis=1)

        for i in range(8):
            with tf.name_scope("hidden_{}".format(i)):
                x = block(x, 64)

        weights_init = tf.zeros(shape=[x.get_shape().as_list()[-1], (self.n - 1) ** 2], dtype=tf.float32)
        weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

        return tf.matmul(x, weights)

    # ---------------------- data generator ----------------------
    def __init__(self, n, num_layers):
        self.n = n
        self.num_layers = num_layers

        self.input_ts = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2, (self.n - 1) ** 2], name="t")
        self.input_cs = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2], name="c")
        self.input_roots = tf.placeholder(tf.float32, shape=[None, (self.n - 1) ** 2], name="root")

        xks = tf.identity(self.input_cs)
        for _ in range(num_layers - 1):
            xks = tf.reduce_sum(self.input_ts * tf.reshape(xks, [-1, 1, (self.n - 1) ** 2]), axis=2) + self.input_cs

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

    def tarin(self, batch_size=16, global_step=1024, version='jacobi'):
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


class KerasSolver:
    @classmethod
    def sequence(cls, batch_size, n):
        while True:
            ts = []
            cs = []
            for _ in range(batch_size):
                data = JacobiDataGenerator(n=n)
                ts.append(data.t)
                cs.append(data.c)

            ts = np.stack(ts)
            cs = np.stack(cs)
            t_labels = np.einsum("nij,njk->nik", ts, ts)
            c_labels = np.einsum("nij,nj->ni", ts, cs) + cs
            labels = np.concatenate([np.reshape(t_labels, [-1, n ** 2]), c_labels], axis=1)
            yield {"input_1": ts, "input_2": cs}, labels

    @classmethod
    def t_logits(cls):
        def polynomial(ts):
            with tf.name_scope('t_noise'):
                n = ts.get_shape().as_list()[-1]

                with tf.name_scope('pre_processing'):
                    flatten_ts = tf.reshape(ts, shape=[-1, n ** 2])

                with tf.name_scope('square_item'):
                    weights = tf.Variable(initial_value=tf.zeros([n, n, n ** 2, n ** 2], dtype=tf.float32))
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

                    square_item = tf.einsum("xyij,ni,nj->nxy", weights, flatten_ts, flatten_ts)

                with tf.name_scope('linear_item'):
                    weights = tf.Variable(initial_value=tf.zeros([n, n, n ** 2], dtype=tf.float32))
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

                    linear_item = tf.einsum("xyi,ni->nxy", weights, flatten_ts)

                t_noise = square_item + linear_item

            return ts + t_noise

        layer = keras.layers.Lambda(polynomial)
        return layer

    @classmethod
    def c_logits(cls):
        def polynomial(args):
            ts, cs = args
            with tf.name_scope('c_noise'):
                n = cs.get_shape().as_list()[-1]

                with tf.name_scope('pre_processing'):
                    flatten_ts = tf.reshape(ts, shape=[-1, n ** 2])

                with tf.name_scope('square_item'):
                    weights = tf.Variable(initial_value=tf.zeros([n, n ** 2, n], dtype=tf.float32))
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

                    square_item = tf.einsum("xij,ni,nj->nx", weights, flatten_ts, cs)

                with tf.name_scope('linear_item'):
                    weights = tf.Variable(initial_value=tf.zeros([n, n], dtype=tf.float32))
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

                    linear_item = tf.einsum("xi,ni->nx", weights, cs)

                c_noise = square_item + linear_item

            return cs + c_noise

        layer = keras.layers.Lambda(polynomial)
        return layer

    @classmethod
    def interface(cls):
        def func(args):
            t_logits, c_logits = args

            n = c_logits.get_shape().as_list()[-1]

            return tf.concat([tf.reshape(t_logits, [-1, n ** 2]), c_logits], axis=1)

        layer = keras.layers.Lambda(func)
        return layer

    def loss(self, labels, logits):
        def func(args):
            logits_, labels_ = args
            square_norm = tf.reduce_sum(tf.square(logits_ - labels_), axis=-1)

            return tf.reduce_mean(tf.sqrt(square_norm))

        return keras.layers.Lambda(func)([labels, logits])

    # ---------------------- data generator ----------------------
    def __init__(self, n):
        self.n = n
        self.input_ts = keras.layers.Input(shape=[n, n])
        self.input_cs = keras.layers.Input(shape=[n])
        self.labels = keras.layers.Input(shape=[n * (n + 1)])

        self.t_logits = self.t_logits()(self.input_ts)
        self.c_logits = self.c_logits()([self.input_ts, self.input_cs])
        logits = self.interface()([self.t_logits, self.c_logits])

        self.model = keras.models.Model(inputs=[self.input_ts, self.input_cs], outputs=logits)

        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=self.loss)

    @classmethod
    def tarin(cls, n, batch_size=2 ** 8, global_step=2 ** 15, version='jacobi'):
        model = cls(n)
        # model.model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=model.loss
        #                     # , metrics=['mse']
        #                     )

        history = model.model.fit_generator(
            cls.sequence(batch_size=batch_size, n=n), steps_per_epoch=global_step)

        # with tf.name_scope("logits"):
        #     t_logits = model.input_ts + model.t_noise(model.input_ts)
        #     c_logits = model.input_cs + model.c_noise(model.input_ts, model.input_cs)
        #
        # with tf.name_scope("labels"):
        #     t_labels = tf.einsum("nij,njk->nik", model.input_ts, model.input_ts)
        #     c_labels = tf.einsum("nij,nj->ni", model.input_ts, model.input_cs) + model.input_cs
        #
        # with tf.name_scope("loss"):
        #     t_square_norm = tf.reduce_sum(tf.square(t_logits - t_labels), axis=[1, 2])
        #     c_square_norm = tf.reduce_sum(tf.square(c_logits - c_labels), axis=-1)
        #
        #     loss = tf.reduce_mean(tf.sqrt(t_square_norm + c_square_norm))
        #
        # with tf.name_scope("train_op"):
        #     train_op = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)
        #     # train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)
        #
        # with tf.Session() as sess:
        #     tf.global_variables_initializer().run()
        #     for i in range(global_step):
        #         ts = []
        #         cs = []
        #         for _ in range(batch_size):
        #             if version == "jacobi":
        #                 data = JacobiDataGenerator(n=n)
        #             elif version == "GS":
        #                 data = GSDataGenerator(n=n)
        #             elif version == "SOR":
        #                 data = SORDataGenerator(n=n)
        #             else:
        #                 raise ValueError
        #             ts.append(data.t)
        #             cs.append(data.c)
        #         ts = np.stack(ts)
        #         cs = np.stack(cs)
        #         _, loss_val = sess.run([train_op, loss], feed_dict={model.input_ts: ts, model.input_cs: cs})
        #         if i % 2 ** 10 == 0:
        #             print("i:{},\tloss:{:.4e}".format(i, loss_val))


TensorflowSolver(n=8, num_layers=4).tarin(version='jacobi')
# KerasSolver.tarin(n=8, version='jacobi')

# print(JacobiDataGenerator(n=16).error(num_layers=10))
# print(GSDataGenerator(n=16).error(num_layers=10))
# print(SORDataGenerator(n=16).error(num_layers=10))
# JacobiDataGenerator.plot()
# GSDataGenerator.plot()
# SORDataGenerator.plot()