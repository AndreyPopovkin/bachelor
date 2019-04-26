from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt

from IPython.core.pylabtools import figsize
figsize(11, 9)

import collections

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import debug as tf_debug

tfd = tfp.distributions

import tensorflow.contrib.eager as tfe
from tensorflow.python.eager.context import eager_mode, graph_mode
import pandas as pd
import numpy as np

# Handy snippet to reset the global graph and global session.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tf.reset_default_graph()
    try:
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession()
    
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

inv_alpha_transform = lambda y: np.log(y)  # Not using TF here.
fwd_alpha_transform = tf.exp

def _make_ps_prior(num_classes, dtype):
    raw_prior_alpha = tf.get_variable(
      name='raw_prior_alpha',
      initializer=np.array(inv_alpha_transform(5.), dtype=dtype))
    raw_prior_beta = tf.get_variable(
      name='raw_prior_beta',
      initializer=np.array(inv_alpha_transform(5.), dtype=dtype))   
    return tfp.distributions.Independent(
      tfp.distributions.Beta(
          fwd_alpha_transform(raw_prior_alpha) * tf.ones(num_classes),
          fwd_alpha_transform(raw_prior_beta) * tf.ones(num_classes)),
      reinterpreted_batch_ndims=1)

make_ps_prior = tf.make_template(name_='make_ps_prior', func_=_make_ps_prior)

beta3_5 = tfp.distributions.Beta(3,5)

beta3_5 = tfp.distributions.Beta([3,3],[5,5])

print (sess.run(beta3_5.sample()))

two_betas = tfp.distributions.Independent(tfp.distributions.Beta(3 * tf.ones(2), 5 * tf.ones(2)),  reinterpreted_batch_ndims=1)

d = tfp.distributions.Binomial(total_count=[5.0,5.0,5.0], probs=[0.3,0.4,0.99])
print(d)
print(sess.run(d.prob([1,2,5])))

d = tfp.distributions.Binomial(total_count=5.0, probs=[0.3,0.4,0.99])
print(d)
print(sess.run(tf.reduce_sum(d.log_prob([1,2,5]), axis=-1)))

def _make_ps_log_likelihood(prob, class_id, total_count):
    prob_c = tf.gather(prob, indices=tf.to_int32(class_id - 1), axis=-1)
    total_count_c = tf.gather(total_count, indices=tf.to_int32(class_id - 1), axis=-1)
    return tfp.distributions.Binomial(total_count=tf.to_float(total_count_c), probs=prob_c)

make_ps_log_likelihood = tf.make_template(name_='make_ps_log_likelihood', func_=_make_ps_log_likelihood)

res = tf.gather([0.1,0.2,0.3,0.4,0.5,0.6], indices=tf.to_int32([0,0,4,4,1,1,2,2,3,3]))
print(sess.run(res)+1)

def joint_log_prob(prob, total_count, clicks, class_id, dtype):
    num_classes = len(total_count)
    rv_prob = make_ps_prior(num_classes, dtype)
    rv_clicks = make_ps_log_likelihood(prob, class_id, total_count)
    return (rv_prob.log_prob(prob) + 
        tf.reduce_sum(rv_clicks.log_prob(clicks), axis=-1))

ps_data_arr = np.array([
    [20, 10, 1, 0.499993], [20, 3, 2, 0.230211], [20, 8, 3, 0.236831], 
    [20, 7, 4, 0.246463], [20, 6, 5, 0.370862], [20, 5, 6, 0.320656], 
    [20, 10, 7, 0.519887], [20, 12, 8, 0.52845], [20, 8, 9, 0.453077], 
    [20, 8, 10, 0.431245], [20, 10, 11, 0.499243], [20, 9, 12, 0.471968], 
    [20, 2, 13, 0.152176], [20, 14, 14, 0.48496], [20, 6, 15, 0.246193]
    ])

ps_data_pd=pd.DataFrame(data=ps_data_arr[0:, 0:],
             index=ps_data_arr[0:, 2],
             columns=["total_count", "clicks", "class_id", "true_p"],
             dtype=np.float32)

ps_data_pd['class_id'] = ps_data_pd.class_id.astype('int32')
ps_data_pd.drop_duplicates()

# Specify unnormalized posterior.

dtype = np.float32


def unnormalized_posterior_log_prob(prob):
    return joint_log_prob(
        prob=prob,
        total_count=dtype(ps_data_pd.total_count.values),
        clicks=dtype(ps_data_pd.clicks.values),
        class_id=np.int32(ps_data_pd.class_id.values),
        dtype=dtype)


# Set-up E-step.
step_size = tf.get_variable(
            'step_size',
            initializer=np.array(0.1, dtype=dtype),
            trainable=False)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_posterior_log_prob,
    num_leapfrog_steps=10,
    step_size=step_size,
    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
      num_adaptation_steps=None),
    state_gradients_are_stopped=True)

init_random_weights = tf.placeholder(dtype, shape=[len(ps_data_pd)])

posterior_random_weights, kernel_results = tfp.mcmc.sample_chain(
    num_results=3,
    num_burnin_steps=0,
    num_steps_between_results=0,
    current_state=init_random_weights,
    kernel=hmc)

# Initialize all variables.

init_op = tf.initialize_all_variables()

# Grab variable handles for diagnostic purposes.

with tf.variable_scope('make_ps_prior', reuse=True):
    prior_alpha = fwd_alpha_transform(tf.get_variable(
        name='raw_prior_alpha', dtype=dtype))
    prior_beta = fwd_alpha_transform(tf.get_variable(
        name='raw_prior_beta', dtype=dtype))

init_op.run()
w_ = np.random.beta(1, 2, size=len(ps_data_pd)).astype(dtype)# 0.5 * np.ones([len(ps_data_pd)], dtype=dtype)


print(sess.run([
    kernel_results,
    #posterior_random_weights,
    #loss,
    #train_op
], feed_dict={init_random_weights: w_}))
