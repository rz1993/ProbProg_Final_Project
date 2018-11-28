import edward as ed
import tensorflow as tf

from edward.models import (Dirichlet, InverseGamma,
                           MultivariateNormalDiag, Normal,
                           ParamMixture, Empirical)

class GMM(object):
    def __init__(self, n, xdim, n_mixtures=5, mc_samples=500):
        # Compute the shape dynamically from placeholders
        self.x_ph = tf.placeholder(tf.float32, [None, xdim])
        self.k = k = n_mixtures
        self.batch_size = n
        self.d = d = xdim
        self.sample_size = tf.placeholder(tf.int32, ())

        # Build the priors over membership probabilities and mixture parameters
        with tf.variable_scope("priors"):
            pi = Dirichlet(tf.ones(k))

            mu = Normal(tf.zeros(d), tf.ones(d), sample_shape=k)
            sigmasq = InverseGamma(tf.ones(d), tf.ones(d), sample_shape=k)

        # Build the conditional mixture model
        with tf.variable_scope("likelihood"):
            x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                             MultivariateNormalDiag,
                             sample_shape=n)
            z = x.cat

        # Build approximate posteriors as Empirical samples
        t = mc_samples
        with tf.variable_scope("posteriors_samples"):
            qpi = Empirical(tf.get_variable(
                "qpi/params", [t, k],
                initializer=tf.constant_initializer(1.0 / k)))
            qmu = Empirical(tf.get_variable(
                "qmu/params", [t, k, d],
                initializer=tf.zeros_initializer()))
            qsigmasq = Empirical(tf.get_variable(
                "qsigmasq/params", [t, k, d],
                initializer=tf.ones_initializer()))
            qz = Empirical(tf.get_variable(
                "qz/params", [t, n],
                initializer=tf.zeros_initializer(),
                dtype=tf.int32))

        # Build inference graph using Gibbs and conditionals
        with tf.variable_scope("inference"):
            self.inference = ed.Gibbs({
                pi: qpi,
                mu: qmu,
                sigmasq: qsigmasq,
                z: qz
            }, data={
                x: self.x_ph
            })
            self.inference.initialize()

        # Build predictive posterior graph by taking samples
        n_samples = self.sample_size
        with tf.variable_scope("posterior"):
            mu_smpl = qmu.sample(n_samples) # shape: [1, 100, k, d]
            sigmasq_smpl = qsigmasq.sample(n_samples)

            x_post = Normal(
                loc=tf.ones((n, 1, 1, 1)) * mu_smpl,
                scale=tf.ones((n, 1, 1, 1)) * tf.sqrt(sigmasq_smpl)
            )
            # NOTE: x_ph has shape [n, d]
            x_broadcasted = tf.tile(
                tf.reshape(self.x_ph, (n, 1, 1, d)),
                (1, n_samples, k, 1)
            )

            x_ll = x_post.log_prob(x_broadcasted)
            x_ll = tf.reduce_sum(x_ll, axis=3)
            x_ll = tf.reduce_mean(x_ll, axis=1)

        self.sample_t_ph = tf.placeholder(tf.int32, ())
        self.eval_ops = {
            'generative_post': x_post,
            'qmu': qmu,
            'qsigma': qsigma,
            'post_running_mu': tf.reduce_mean(
                qmu.params[:self.sample_t_ph],
                axis=0
            )
            'post_log_prob': xll
        }

    def make_feed_dict_trn(self, data, epoch=None, n_epochs=None):
        return {
            self.x_ph: data
        }

    def make_feed_dict_test(self, data, epoch=None, n_epochs=None, sample_size=100):
        return {
            self.x_ph: data,
            self.sample_size: sample_size
        }
