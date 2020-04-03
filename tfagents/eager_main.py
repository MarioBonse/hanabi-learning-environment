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
from hanabi_learning_environment import utility
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
from tf_agents.policies import py_tf_policy
from functools import partial


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 21,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('checkpoint_interval', 5,
                     'Number of Epochs to run before checkpointing')
flags.DEFINE_integer('rb_size', 50000,
                     'Number of transitions to store in the Replay Buffer')
flags.DEFINE_float('gradient_clipping', 0.1,
                     'Numerical value to clip the norm of the gradients')
flags.DEFINE_float('learning_rate', 1e-6,
                     "Learning Rate for the agent's training process")
flags.DEFINE_bool('use_ddqn', False,
                  'If True uses the DdqnAgent instead of the DqnAgent.')
flags.DEFINE_bool('perf_tracing', False,
                  'If True uses traces computation to see on Tensorboard the utilization of computational resources')
flags.DEFINE_list('network', [512, 512],
                  'List of layers and corresponding nodes per layer')
FLAGS = flags.FLAGS


def observation_and_action_constraint_splitter(obs):
    return obs['observations'], obs['legal_moves']


def run_verbose_mode(agent_1, agent_2):
    env = rl_env.make('Hanabi-Full-CardKnowledge', num_players=2)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    
    state = tf.env.reset()



@gin.configurable
def train_eval(
    root_dir,
    num_iterations=21,
    fc_layer_params=(256, 128),
    # Params for collect
    collect_episodes_per_iteration=300,
    epsilon_greedy=0.4,
    decay_steps=200000,
    replay_buffer_capacity=50000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=25000,
    batch_size=64,
    learning_rate=1e-6,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=0.1,
    # Params for eval
    eval_interval=1000,
    num_eval_episodes=10,
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=5,
    policy_checkpoint_interval=5,
    rb_checkpoint_interval=5,
    summaries_flush_secs=10,
    agent_class=dqn_agent.DqnAgent,
    debug_summaries=False,
    perf_tracing=False,
    summarize_grads_and_vars=False,
    num_players=2):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)

    train_summary_writer.set_as_default()
    

    eval_summary_writer = tf.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    eval_metrics = [
        py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]


    # create the enviroment
    env = rl_env.make('Hanabi-Full-CardKnowledge', num_players=num_players)                        
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_py_env = rl_env.make(
        'Hanabi-Full-CardKnowledge', num_players=num_players)

    
    train_step_1 = tf.Variable(0, trainable=False, name='global_step_1', dtype=tf.int64)
    train_step_2 = tf.Variable(0, trainable=False, name='global_step_2', dtype=tf.int64)
    epoch_counter = tf.Variable(0, trainable=False, name='Epoch', dtype=tf.int64)
    
    decaying_epsilon_1 = partial(utility.decaying_epsilon, epsilon_greedy, train_step_1, decay_steps)
    decaying_epsilon_2 = partial(utility.decaying_epsilon, epsilon_greedy, train_step_2, decay_steps)
    
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
        epsilon_greedy=decaying_epsilon_1,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_1)

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
        epsilon_greedy=decaying_epsilon_2,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_2)

    # replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent_1.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    
    #FIXME we haven't really looked at how train_metrics are managed in the driver when it's running
    # in particular it is unclear whether any issues come up because of the fact that now the driver
    # is running two different policies (agents). In other words, we only modified the DynamicEpicodeDriver
    # with what was stricly necessary to make it run with two different agents. We never checked what the 
    # implications of this would be for logging, summaries and metrics.
    # metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    # checkpointer:
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent_1=tf_agent_1,
        agent_2=tf_agent_2,
        train_step_1=train_step_1,
        train_step_2=train_step_2,
        epoch_counter=epoch_counter,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy_1=tf_agent_1.policy,
        policy_2=tf_agent_2.policy)

    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    
    #FIXME I believe that this is useless because Checkpointer object will already restore latest checkpoint
    # inside it's __init__ method. Should be tested though.
    print('\n\n\nTrying to restore Checkpoints for the agents and Replay Buffer')
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    print('\n\n')
    
    # Compiled version of training functions (much faster)
    agent_1_train_function = common.function(tf_agent_1.train)
    agent_2_train_function = common.function(tf_agent_2.train)
    
    # replay buffer update for the driver
    replay_observer = [replay_buffer.add_batch]
    
    # This allows us to look at resource utilization across time
    if perf_tracing:
        tf.summary.trace_on(profiler=True)
    
    # Supposedly this is a performance improvement. According to TF devs it achieves
    # better performance by compiling stuff specialized on shape. If the shape of the stuff
    # going around changes a lot then it may actually get worse performance. To me it seems
    # that everything in our code runs with same shapes/batch sizes so I think it should be good.
    tf.config.optimizer.set_jit(True)
    for step in range(num_iterations):
        # the two policies we use to collect data
        collect_policy_1 = tf_agent_1.collect_policy
        collect_policy_2 = tf_agent_2.collect_policy
        
        assert decaying_epsilon_1().numpy() == decaying_epsilon_2().numpy()
        
        print('EPOCH {}'.format(epoch_counter.numpy()))
        print('The epsilon for the epsilon-greedy policy at this iteration is:', decaying_epsilon_1().numpy())
        
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
            num_steps=2).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print('Starting partial training of both Agents from Replay Buffer\nCounting Steps:')        

        losses_1 = tf.TensorArray(tf.float32, size=train_steps_per_iteration)
        losses_2 = tf.TensorArray(tf.float32, size=train_steps_per_iteration)
        c = 0
        start_time  = time.time()
        for data in dataset:
            if c % (train_steps_per_iteration/10) == 0 and c != 0:
                tf.summary.scalar("loss_agent_1", tf.math.reduce_mean(losses_1.stack()), step=train_step_1)
                tf.summary.scalar("loss_agent_2",  tf.math.reduce_mean(losses_1.stack()), step=train_step_2)
                print("{}% completed with {} steps done".format(int(c/train_steps_per_iteration*100), c))
            if c == train_steps_per_iteration:
                break
            experience, _ = data
            #FIXME _train and _loss functions of the two agents call the tf.summary to log the value
            # of various things. It seems though that they are writing on the same summary variables
            # because they're not build for the possibility of two agents in training. We need to change 
            # the agent class so that it can accept some agent_id string that it then uses to tf.name_scope
            # all the summaries. See line 482 in dwn_agent.py to understand what I mean by tf.name_scope  
            losses_1 = losses_1.write(c, agent_1_train_function(experience=experience).loss)
            losses_2 = losses_2.write(c, agent_2_train_function(experience=experience).loss)                
            c += 1

        losses_1 = losses_1.stack()
        losses_2 = losses_2.stack()
        print("Ended epoch training of both Agents, it took {}".format(time.time() - start_time))
        print('Mean loss for Agent 1 is: {}'.format(tf.math.reduce_mean(losses_1)))
        print('Mean loss for Agent 2 is: {}\n\n'.format(tf.math.reduce_mean(losses_2)))
                
        eval_policy_1 = tf_agent_1.policy
        eval_policy_2 = tf_agent_2.policy
        
        epoch_counter.assign_add(1)
        
        for train_metric in train_metrics:
            train_metric.tf_summaries(train_step=epoch_counter, step_metrics=train_metrics[:2])

        if epoch_counter.numpy() % train_checkpoint_interval == 1:
            train_checkpointer.save(global_step=epoch_counter.numpy() - 1)

        if epoch_counter.numpy() % policy_checkpoint_interval == 1:
            policy_checkpointer.save(global_step=epoch_counter.numpy() - 1)

        if epoch_counter.numpy() % rb_checkpoint_interval == 1:
            rb_checkpointer.save(global_step=epoch_counter.numpy() - 1)

        ''' TODO
        #FIXME compute summaries runs a PyDriver instead of DynamicEpisodeDriver, we need to
        # adapt it so that it can also accept two policies in input and run them one after the
        # other (implementing non-self-play).
        if (epoch_counter.numpy() - 1) % eval_interval == 0:
            eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent_1.policy)
            metric_utils.compute_summaries(
            eval_metrics,
            eval_py_env,
            eval_py_policy,
            num_episodes=num_eval_episodes,
            global_step=train_step_1,
            log = True
            )
        '''
    
    # This allows us to look at resource utilization across time
    if perf_tracing:
        tf.summary.trace_export(name='Performance check', step=train_step_1, profiler_outdir=train_dir)





def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    agent_class = dqn_agent.DdqnAgent if FLAGS.use_ddqn else dqn_agent.DqnAgent
    fc_layer_params = tuple([int(number) for number in FLAGS.network])
    train_eval(
        FLAGS.root_dir,
        agent_class=agent_class,
        num_iterations=FLAGS.num_iterations,
        gradient_clipping=FLAGS.gradient_clipping,
        learning_rate=FLAGS.learning_rate,
        train_checkpoint_interval=FLAGS.checkpoint_interval,
        policy_checkpoint_interval=FLAGS.checkpoint_interval,
        rb_checkpoint_interval=FLAGS.checkpoint_interval,
        replay_buffer_capacity=FLAGS.rb_size,
        fc_layer_params=fc_layer_params,
        perf_tracing=FLAGS.perf_tracing)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
