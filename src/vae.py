import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, Normal

class VAE(object):
    def __init__(self, hdims, zdim, xdim):
        x_ph = tf.placeholder(tf.float32, [None, xdim])
        batch_size = tf.shape(x_ph)[0]
        sample_size = tf.placeholder(tf.int32, [])

        # Define the generative network (p(x | z))
        with tf.variable_scope('generative'):
            z = Normal(loc=tf.zeros([batch_size, hdimz]),
                       scale=tf.ones([batch_size, hdimz]))

            hidden = tf.layers.dense(z, hdim1, activation=tf.nn.relu, name="dense1")
            loc = tf.layers.dense(hidden, xdim, name="dense2")

            x_gen = Bernoulli(logits=loc)

        # Define the inference network (q(z | x))
        with tf.variable_scope('inference'):
            hidden = tf.layers.dense(x_ph, hdim1, activation=tf.nn.relu)
            qloc = tf.layers.dense(hidden, hdimz)
            qscale = tf.layers.dense(hidden, hdimz, activation=tf.nn.softplus)
            qz = Normal(loc=qloc, scale=qscale)
            qz_sample = qz.sample(sample_size)

        # Define the generative network using posterior samples from q(z | x)
        with tf.variable_scope('generative'):
            qz_sample = tf.reshape(qz_sample, [-1, hdimz])
            hidden = tf.layers.dense(qz_sample, hdim1, activation=tf.nn.relu, reuse=True, name="dense1")
            loc = tf.layers.dense(hidden, xdim, reuse=True, name="dense2")

            x_gen_post = tf.sigmoid(loc)

        self.x_ph = x_ph
        self.x_data = tf.cast(x_ph > 0, tf.int32)
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.ops = {
            'generative': x_gen,
            'inference': qz_sample,
            'generative_post': x_gen_post
        }

    with tf.variable_scope('inference'):
        self.inference = ed.KLqp({z: qz}, data={x_gen: self.x_data})
        self.lr = tf.placeholder(tf.float32, shape=())

        optimizer = tf.train.RMSPropOptimizer(self.lr, epsilon=0.9)

        self.inference.initialize(
            optimizer=optimizer,
            n_samples=10,
            kl_scaling={z: kl_coef}
        )

        # Build elbo loss to evaluate on validation data
        self.elbo_loss, _ = inference.build_loss_and_gradients([])
