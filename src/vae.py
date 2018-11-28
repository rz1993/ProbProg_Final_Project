import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, Normal, TransformedDistribution

tfd = tf.contrib.distributions

class VAE(object):
    def __init__(self, hdims, zdim, xdim, gen_scale=1.):
        x_ph = tf.placeholder(tf.float32, [None, xdim])
        batch_size = tf.shape(x_ph)[0]
        sample_size = tf.placeholder(tf.int32, [])

        # Define the generative network (p(x | z))
        with tf.variable_scope('generative', reuse=tf.AUTO_REUSE):
            z = Normal(loc=tf.zeros([batch_size, zdim]),
                       scale=tf.ones([batch_size, zdim]))

            hidden = tf.layers.dense(z, hdims[0], activation=tf.nn.relu, name="dense1")
            loc = tf.layers.dense(hidden, xdim, name="dense2")

            x_gen = TransformedDistribution(
                distribution=tfd.Normal(loc=loc, scale=gen_scale),
                bijector=tfd.bijectors.Exp(),
                name="LogNormalTransformedDistribution"
            )
            #x_gen = Bernoulli(logits=loc)

        # Define the inference network (q(z | x))
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(x_ph, hdims[0], activation=tf.nn.relu)
            qloc = tf.layers.dense(hidden, zdim)
            qscale = tf.layers.dense(hidden, zdim, activation=tf.nn.softplus)
            qz = Normal(loc=qloc, scale=qscale)
            qz_sample = qz.sample(sample_size)

        # Define the generative network using posterior samples from q(z | x)
        with tf.variable_scope('generative'):
            qz_sample = tf.reshape(qz_sample, [-1, zdim])
            hidden = tf.layers.dense(qz_sample, hdims[0], activation=tf.nn.relu, reuse=True, name="dense1")
            loc = tf.layers.dense(hidden, xdim, reuse=True, name="dense2")

            x_gen_post = tf.exp(loc)

        self.x_ph = x_ph
        self.x_data = self.x_ph
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.ops = {
            'generative': x_gen,
            'inference': qz_sample,
            'generative_post': x_gen_post
        }

        self.kl_coef = tf.placeholder(tf.float32, ())
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            self.inference = ed.KLqp({z: qz}, data={x_gen: self.x_data})
            self.lr = tf.placeholder(tf.float32, shape=())

            optimizer = tf.train.RMSPropOptimizer(self.lr, epsilon=0.9)

            self.inference.initialize(
                optimizer=optimizer,
                n_samples=10,
                kl_scaling={z: self.kl_coef}
            )

            # Build elbo loss to evaluate on validation data
            self.eval_loss, _ = self.inference.build_loss_and_gradients([])

    def make_feed_dict_trn(self, data, epoch, n_epochs):
        return {
            self.x_ph: data,
            self.kl_coef: min(1., epoch / 4),
            self.lr: 0.001
        }

    def make_feed_dict_test(self, data, epoch, n_epochs):
        return {
            self.x_ph: data,
            self.sample_size: 10,
            self.kl_coef: 1.
        }
