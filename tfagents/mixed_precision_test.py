from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

from hanabi_learning_environment import rl_env
from hanabi_learning_environment import utility
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.utils import common
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def observation_and_action_constraint_splitter(obs):
    return obs['observations'], obs['legal_moves']



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

    
train_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
epoch_counter = tf.Variable(0, trainable=False, name='Epoch', dtype=tf.int64)
    
    
    
#TODO Performance Improvement: "When training on GPUs, make use of the TensorCore. GPU kernels use
# the TensorCore when the precision is fp16 and input/output dimensions are divisible by 8 or 16 (for int8)"
# (from https://www.tensorflow.org/guide/profiler#improve_device_performance). Maybe consider decreasing
# precision to fp16 and possibly compensating with increased model complexity to not lose performance?
# I mean if this allows us to use TensorCore then maybe it is worthwhile (computationally) to increase 
# model size and lower precision. Need to test what the impact on agent performance is.
# See https://www.tensorflow.org/guide/keras/mixed_precision for more info
# create an agent and a network 

q_network = q_network.QNetwork(tf_env.time_step_spec().observation['observations'],
                                tf_env.action_spec(),
                                fc_layer_params=(512,512))

enc_network_layers = q_network._encoder._postprocessing_layers

print(enc_network_layers)

