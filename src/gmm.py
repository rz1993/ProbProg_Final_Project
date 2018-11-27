import edward as ed
import tensorflow as tf

from edward.models import (Dirichlet, InverseGamma,
                           MultivariateNormalDiag, Normal)

class GMM(object):
    def __init__(self, data, n_mixtures=5, mc_samples=500):
        self.x_ph = tf.placeholder(tf.float32, [None, data.shape[1]])
        self.k = k = n_mixtures
        n, d = tf.shape(self.x_ph)

        with tf.variable_scope("priors"):
            pi = Dirichlet(tf.ones(k))

            mu = Normal(tf.zeros(d), tf.ones(d), sample_shape=k)
            sigmasq = InverseGamma(tf.ones(d), tf.ones(d), sample_shape=k)

        with tf.variable_scope("likelihood")
            x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                             MultivariateNormalDiag,
                             sample_shape=n)
            z = x.cat

        t = mc_samples
        with tf.variable_scope("posteriors"):
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
            
