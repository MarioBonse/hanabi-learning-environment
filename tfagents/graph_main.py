# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3

r"""Train and Eval DQN.

To run:

```bash
tensorboard --logdir $HOME/tmp/dqn/hanabi --port 2223 &

python tfagents/DQNmain.py \
  --root_dir=$HOME/tmp/dqn/hanabi/ \
  --alsologtostderr
```

code from https://github.com/tensorflow/agents/blob/master/tf_agents/agents/dqn/examples/v1/train_eval_gym.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

from hanabi_learning_environment import rl_env
import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym

from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from tqdm import tqdm


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_bool('use_ddqn', False,
                  'If True uses the DdqnAgent instead of the DqnAgent.')
FLAGS = flags.FLAGS


def observation_and_action_constraint_splitter(obs):
    return obs['observations'], obs['legal_moves']


@gin.configurable
def train_eval(
    root_dir,
    num_iterations=100000,
    fc_layer_params=(100,),
    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1000,
    batch_size=64,
    learning_rate=1e-3,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    agent_class=dqn_agent.DqnAgent,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None,
        num_players=2):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)

    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    eval_metrics = [
        py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    # create the enviroment
    env = rl_env.make('Hanabi-Full-CardKnowledge',
                            num_players=num_players)                        
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_py_env = rl_env.make(
        'Hanabi-Full-CardKnowledge', num_players=num_players)

    # create an agent and a network 
    tf_agent_1 = agent_class(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network= q_network.QNetwork(
                                        tf_env.time_step_spec().observation['observations'],
                                        tf_env.action_spec(),
                                        fc_layer_params=fc_layer_params
                                        ),
        optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate),
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
        epsilon_greedy=epsilon_greedy,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars)

    # Second agent. we can have as many as we want
    tf_agent_2 = agent_class(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network= q_network.QNetwork(
                                        tf_env.time_step_spec().observation['observations'],
                                        tf_env.action_spec(),
                                        fc_layer_params=fc_layer_params
                                        ),
        optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate),
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
        epsilon_greedy=epsilon_greedy,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars)

    # replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent_1.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent_1.policy)

    # metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent_1,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=tf_agent_1.policy,)

    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    # replay buffer update for the driver
    replay_observer = [replay_buffer.add_batch]
    collect_time = 0
    train_time = 0
    for global_step_val in range(num_iterations):
        # the two policies we use to collect data
        collect_policy_1 = tf_agent_1.collect_policy
        collect_policy_2 = tf_agent_2.collect_policy

        # episode driver 
        start_time = time.time()
        collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            [collect_policy_1, collect_policy_2],
            observers=replay_observer + train_metrics,
            num_episodes=collect_steps_per_iteration).run()
        collect_time += time.time() - start_time
        start_time = time.time()
        print('\nFinished running the Driver\n')
        # Dataset generates trajectories with shape [Bx2x...]
        # train for the first agent
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        print('\n\n\nStarting epoch training of both Agents from Replay Buffer\nCounting Steps:')
        start_time  = time.time()
        
        losses_1, losses_2 = partial_training(dataset, tf_agent_1, tf_agent_2, n_steps=5000)
        
        print("Ended epoch training of both Agents, it took {}".format(time.time() - start_time))
        print('Mean loss for Agent 1 is: {}'.format(tf.math.reduce_mean(losses_1)))
        print('Mean loss for Agent 2 is: {}'.format(tf.math.reduce_mean(losses_2)))
        
        if global_step_val % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step = global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
            policy_checkpointer.save(global_step = global_step_val)

        if global_step_val % rb_checkpoint_interval == 0:
            rb_checkpointer.save(global_step = global_step_val)


#FIXME actually losses probably need to be dealt with differently because I think that when I redefine losses_1
# using tf.stack I create a tensor and tensor variables are all in global scope  (they actually don't even know what scope is)
# so an InaccessibleTensorError is raised... Not sure I understand everything though 
#TODO change this function so that it actually can run in batches and uses tensors as much as possible (inluding losses)
@tf.function
def partial_training(dataset, tf_agent_1, tf_agent_2, n_steps=500):
    c = 0
    losses_1 = []
    print('SEMO Dentro')
    print(losses_1)
    print(type(losses_1))
    print(type(c))
    print('ho printato')
    losses_2 = []
    for data in dataset:
        if c % 500 == 0:
            tf.print(c)
        c += 1
        if c == n_steps:
            break
        experience, _ = data
        losses_1.append(tf_agent_1.train(experience=experience).loss)
        losses_2.append(tf_agent_2.train(experience=experience).loss)
    
    losses_1 = tf.stack(losses_1)
    losses_2 = tf.stack(losses_2)
    
    return losses_1, losses_2




def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    agent_class = dqn_agent.DdqnAgent if FLAGS.use_ddqn else dqn_agent.DqnAgent
    train_eval(
        FLAGS.root_dir,
        agent_class=agent_class,
        num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
