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
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


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
    fc_layer_params=(128, 64),
    # Params for collect
    collect_episodes_per_iteration=500,
    epsilon_greedy=0.1,
    replay_buffer_capacity=200000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=50000,
    batch_size=128,
    learning_rate=1e-3,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=3,
    policy_checkpoint_interval=3,
    rb_checkpoint_interval=3,
    summaries_flush_secs=10,
    agent_class=dqn_agent.DqnAgent,
    debug_summaries=False,
    summarize_grads_and_vars=False,
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
        q_network= q_network.QNetwork(tf_env.time_step_spec().observation['observations'],
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
        q_network= q_network.QNetwork(tf_env.time_step_spec().observation['observations'],
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

    # metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent_1=tf_agent_1,
        agent_2=tf_agent_2,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy_1=tf_agent_1.policy,
        policy_2=tf_agent_2.policy)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    print('\n\n\nTrying to restore Checpoints for the agents and Replay Buffer')
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    print('\n\n')
    
    # Compiled version of training functions (much faster)
    agent_1_train_function = common.function(tf_agent_1.train)
    agent_2_train_function = common.function(tf_agent_2.train)
    
    # replay buffer update for the driver
    replay_observer = [replay_buffer.add_batch]
    for global_step_val in range(num_iterations):
        # the two policies we use to collect data
        collect_policy_1 = tf_agent_1.collect_policy
        collect_policy_2 = tf_agent_2.collect_policy
        print('EPOCH {}'.format(global_step_val + 1))
        # episode driver
        print('\nStarting to run the Driver')
        start_time = time.time()
        collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            [collect_policy_1, collect_policy_2],
            observers=replay_observer + train_metrics,
            num_episodes=collect_episodes_per_iteration).run()
        print('Finished running the Driver, it took {} seconds for {} episodes\n'.format(time.time() - start_time,
                                                                                           collect_episodes_per_iteration))
        # Dataset generates trajectories with shape [Bx2x...]
        # train for the first agent
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(5)

        print('Starting partial training of both Agents from Replay Buffer\nCounting Steps:')

        losses_1 = tf.TensorArray(tf.float32, size=train_steps_per_iteration)
        losses_2 = tf.TensorArray(tf.float32, size=train_steps_per_iteration)
        c = 0
        start_time  = time.time()
        for data in dataset:
            if c % (train_steps_per_iteration/10) == 0 and c != 0:
                print("{}% completed with {} steps done".format(int(c/train_steps_per_iteration*100), c))
            if c == train_steps_per_iteration:
                break
            experience, _ = data
            losses_1 = losses_1.write(c, agent_1_train_function(experience=experience).loss)
            losses_2 = losses_2.write(c, agent_2_train_function(experience=experience).loss)
            c += 1
        losses_1 = losses_1.stack()
        losses_2 = losses_2.stack()
        print("Ended epoch training of both Agents, it took {}".format(time.time() - start_time))
        print('Mean loss for Agent 1 is: {}'.format(tf.math.reduce_mean(losses_1)))
        print('Mean loss for Agent 2 is: {}\n\n'.format(tf.math.reduce_mean(losses_2)))
        
        
        if global_step_val % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step = global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
            policy_checkpointer.save(global_step = global_step_val)

        if global_step_val % rb_checkpoint_interval == 0:
            rb_checkpointer.save(global_step = global_step_val)





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
