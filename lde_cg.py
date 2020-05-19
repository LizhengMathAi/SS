import numpy as np
import tensorflow as tf


np.set_printoptions(precision=2)


class CGDataGenerator:
    """
    A@x = b, A is a SPD matrix
    """
    def __str__(self):
        return "CG"

    def __init__(self, n):
        self.n = n

        mat = 2 * np.random.rand(n, n).astype(np.float) - 1
        self.mat = mat.T @ mat

        self.root = 2 * np.random.rand(n).astype(np.float) - 1
        self.rhs = self.mat@self.root

    def contrast_error(self, num_layers=10, preliminary=True, display=True):
        if display:
            print('=' * 8 + "\tStart to check {}\t".format(self.__str__()) + '=' * 8)

        xk = np.zeros_like(self.root)
        rk = -self.rhs
        pk = -rk

        if preliminary:
            for k in range(num_layers):
                ak = -np.inner(rk, pk) / np.inner(pk, self.mat @ pk)
                xk = xk + ak * pk
                rk1 = self.mat @ xk - self.rhs
                bk = np.inner(rk1, self.mat@pk) / np.inner(pk, self.mat@pk)
                pk = -rk1 + bk * pk
                rk = rk1
        else:
            for k in range(num_layers):
                ak = np.inner(rk, rk) / np.inner(pk, self.mat@pk)
                xk = xk + ak * pk
                rk1 = rk + ak * self.mat@pk
                bk = np.inner(rk1, rk1) / np.inner(rk, rk)
                pk = -rk1 + bk * pk
                rk = rk1

        return np.linalg.norm(xk - self.root)


class CGSolver:
    @classmethod
    def sequence(cls, batch_size, n, num_layers, preliminary):
        mats = []
        rhses = []
        roots = []
        contrast_errors = []
        for _ in range(batch_size):
            data = CGDataGenerator(n=n)
            contrast_errors.append(
                data.contrast_error(num_layers=num_layers, preliminary=preliminary, display=False))

            mats.append(data.mat)
            rhses.append(data.rhs)
            roots.append(data.root)

        contrast_error = sum(contrast_errors) / batch_size

        return np.stack(mats), np.stack(rhses), np.stack(roots), contrast_error

    def __init__(self, n, num_layers=4, preliminary=True):
        self.n = n

        self.input_mats = tf.placeholder(tf.float32, shape=[None, self.n, self.n], name="mat")
        self.input_rhses = tf.placeholder(tf.float32, shape=[None, self.n], name="rhs")
        self.input_roots = tf.placeholder(tf.float32, shape=[None, self.n], name="roots")

        self.output_xks = self.inference(num_layers=num_layers, preliminary=preliminary)

    def mat_mul(self, item_1, item_2):
        return tf.reduce_sum(item_1 * tf.reshape(item_2, shape=[-1, 1, self.n]), axis=2)

    def inner_prod(self, item_1, item_2):
        return tf.reduce_sum(item_1 * item_2, axis=-1)

    def cg_prod(self, item_1, item_2, item_3):
        return tf.reduce_sum(
            tf.reshape(item_1, shape=[-1, self.n, 1]) * item_2 * tf.reshape(item_3, shape=[-1, 1, self.n]),
            axis=[1, 2]
        )

    def l1_norm(self, tensor):
        return tf.reduce_sum(tf.abs(tensor))

    def inference(self, num_layers, preliminary):
        with tf.name_scope("xk"):
            xks = tf.zeros(shape=[1, self.n], dtype=tf.float32, name='xk')

        with tf.name_scope("rk"):
            rks = -self.input_rhses

        with tf.name_scope("pk"):
            pks = tf.identity(self.input_rhses)

        for k in range(num_layers):
            if preliminary:
                with tf.name_scope("ak"):
                    with tf.name_scope("mat_noise"):
                        mat_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(mat_noise))

                    numerator = self.inner_prod(rks, pks)
                    denominator = self.cg_prod(pks, self.input_mats + mat_noise, pks)
                    aks = -numerator / denominator

                with tf.name_scope("xk"):
                    xks = xks + tf.reshape(aks, shape=[-1, 1]) * pks

                with tf.name_scope("rk"):
                    with tf.name_scope("mat_noise"):
                        mat_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(mat_noise))
                    with tf.name_scope("rhs_noise"):
                        rhs_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(rhs_noise))
                    rks = self.mat_mul(self.input_mats + mat_noise, xks) - (self.input_rhses + rhs_noise)

                with tf.name_scope("bk"):
                    with tf.name_scope("mat_noise"):
                        mat_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(mat_noise))
                    inner_1 = self.cg_prod(rks, self.input_mats + mat_noise, pks)
                    with tf.name_scope("mat_noise"):
                        mat_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(mat_noise))

                    inner_2 = self.cg_prod(pks, self.input_mats + mat_noise, pks)
                    bks = inner_1 / inner_2

                with tf.name_scope("pk"):
                    pks = -rks + tf.reshape(bks, shape=[-1, 1]) * pks
            else:
                with tf.name_scope("ak"):
                    with tf.name_scope("mat_noise"):
                        mat_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(mat_noise))

                    aks = self.inner_prod(rks, rks) / self.cg_prod(pks, self.input_mats + mat_noise, pks)

                with tf.name_scope("xk"):
                    xks = xks + tf.reshape(aks, shape=[-1, 1]) * pks

                with tf.name_scope("rk1"):
                    with tf.name_scope("mat_noise"):
                        mat_noise = tf.Variable(initial_value=np.zeros(shape=(1, self.n, self.n)), dtype=tf.float32)
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l1_norm(mat_noise))
                    rk1s = rks + tf.reshape(aks, shape=[-1, 1]) * self.mat_mul(self.input_mats + mat_noise, pks)

                with tf.name_scope("bk"):
                    bks = self.inner_prod(rk1s, rk1s) / self.inner_prod(rks, rks)

                with tf.name_scope("pk"):
                    pks = -rk1s + tf.reshape(bks, shape=[-1, 1]) * pks

                rks = rk1s

        return xks

    @classmethod
    def example(cls, n=64, num_layers=4, preliminary=True, batch_size=1024, global_step=16):

        model = cls(n=n, num_layers=num_layers, preliminary=preliminary)
        regularizer_rate = 0.001
        lr = 0.01

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(model.input_roots - model.output_xks), axis=-1)))

        with tf.name_scope("obj_func"):
            regularizer = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            obj_func = loss + regularizer_rate * regularizer

        train_op = tf.train.AdamOptimizer(lr).minimize(obj_func)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(global_step):
                mats, rhses, roots, contrast_error = model.sequence(batch_size, n, num_layers, preliminary)
                error, _ = sess.run([loss, train_op], feed_dict={
                    model.input_mats: mats,
                    model.input_rhses: rhses,
                    model.input_roots: roots
                })
                # error = sess.run(loss, feed_dict={
                #     model.input_mats: mats,
                #     model.input_rhses: rhses,
                #     model.input_roots: roots
                # })
                print("error:{:.4e}".format(error), ",\tcontrast_error:{:.4e}".format(contrast_error))


# ===================================================================
tf.app.flags.DEFINE_integer("n", 64, "dimension.")
tf.app.flags.DEFINE_integer("num_layers", 4, "models layers.")
tf.app.flags.DEFINE_integer("global_step", 16, "global step.")
tf.app.flags.DEFINE_bool("preliminary", True, "preliminary version.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    CGSolver.example(
        n=FLAGS.n, num_layers=FLAGS.num_layers, global_step=FLAGS.global_step, preliminary=FLAGS.preliminary)


if __name__ == '__main__':
    tf.app.run(main)
