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
flags.DEFINE_integer('num_iterations', 31,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_list('network', [512, 512],
                  'List of layers and corresponding nodes per layer')
flags.DEFINE_integer('collect_episodes_per_epoch', 300,
                     'Number of Episodes to the run in the Driver for collection at each epoch')
flags.DEFINE_integer('reset_at_step', None,
                     'Epoch at which to reset the decay process of epsilon in the Epsilon-Greedy Policy')
flags.DEFINE_integer('rb_size', 50000,
                     'Number of transitions to store in the Replay Buffer')
flags.DEFINE_integer('train_steps_per_epoch', 25000,
                     'Number of calls to the training function for each epoch')
flags.DEFINE_float('learning_rate', 1e-7,
                     "Learning Rate for the agent's training process")
flags.DEFINE_float('gradient_clipping', 0.1,
                     'Numerical value to clip the norm of the gradients')
flags.DEFINE_integer('num_eval_episodes', 1000,
                     'Number of Episodes to the run in the Driver for evaluation')
flags.DEFINE_integer('checkpoint_interval', 10,
                     'Number of Epochs to run before checkpointing')
flags.DEFINE_bool('use_ddqn', False,
                  'If True uses the DdqnAgent instead of the DqnAgent.')

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
    num_iterations=31,
    fc_layer_params=(256, 128),
    # Params for collect
    collect_episodes_per_epoch=300,
    epsilon_greedy=0.4,
    decay_steps=8,
    reset_at_step=None,
    replay_buffer_capacity=50000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_epoch=25000,
    batch_size=64,
    learning_rate=1e-7,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=0.1,
    # Params for eval
    eval_interval=10,
    num_eval_episodes=1000,
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=10,
    policy_checkpoint_interval=10,
    rb_checkpoint_interval=10,
    summaries_flush_secs=10,
    agent_class=dqn_agent.DqnAgent,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    num_players=2):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    #FIXME Checkpointing doesn't synergize with tensorboard summaries, i.e. if you checkpoint
    # at some point, execute some epochs (which are not checkpointed), stop the program and run again 
    # from the last saved checkpoint; then tensorboard  will receive (and display) twice the summaries 
    # relative to the epochs that had been executed, but not checkpointed. How to solve this? No idea. 
    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)

    train_summary_writer.set_as_default()
    

    eval_summary_writer = tf.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]
    
    tf.profiler.experimental.server.start(6009)


    # create the enviroment
    env = rl_env.make('Hanabi-Full-CardKnowledge', num_players=num_players)                        
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_py_env = tf_py_environment.TFPyEnvironment(rl_env.make('Hanabi-Full-CardKnowledge', num_players=num_players))

    
    train_step_1 = tf.Variable(0, trainable=False, name='global_step_1', dtype=tf.int64)
    train_step_2 = tf.Variable(0, trainable=False, name='global_step_2', dtype=tf.int64)
    epoch_counter = tf.Variable(0, trainable=False, name='Epoch', dtype=tf.int64)
    
    #TODO current implementation of the decaying epsilon essentially requires you to pass the
    # reset_at_step argument from the command line every time after you pass it the first time
    # (if you wish for consistent decaying behaviour). Maybe implement some checkpointing of 
    # something in order to avoid this requirement... The only negative side-effect of not having 
    # this implementation is that epsilon might become very low all of a sudden if you forget to
    # pass the reset_at_step argument after you passed it once.
    decaying_epsilon_1 = partial(utility.decaying_epsilon,
                                 initial_epsilon=epsilon_greedy,
                                 train_step=epoch_counter,
                                 decay_time=decay_steps,
                                 reset_at_step=reset_at_step)
    decaying_epsilon_2 = partial(utility.decaying_epsilon,
                                 initial_epsilon=epsilon_greedy,
                                 train_step=epoch_counter,
                                 decay_time=decay_steps,
                                 reset_at_step=reset_at_step)
    
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
    # implications of this would be for logging, summaries and metrics. It seems reasonable though that all these
    # metrics effectively depend only on the environment and so are unaffected by what happens to the agent(s). 
    # We thus do not expect any surprises here, but for example the metric AverageReturnMetric will most likely
    # be considering the rewards of the two agents together; this is actually desired (for now), as it tells us 
    # how many cards they managed to put down together
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
        max_to_keep=3,
        replay_buffer=replay_buffer)

    
    # Compiled version of training functions (much faster)
    #FIXME Tensorflow documentation of tf.function (https://www.tensorflow.org/api_docs/python/tf/function)
    # states that autograph parameter should be set to True for Data-dependent control flow. What does this
    # mean? Is our training function not Data-dependent? Currently common.function (which is a wrapper on the 
    # tf.function wrapper) passes autograph=False by default.
    #TODO Maybe pass experimental_compile=True to common.function? Maybe it's not needed because 
    # later in the code we use tf.config.optimizer.set_jit(True) which enables XLA in general?
    # Who knows, test and look at performance I would say. Another thing to notice is that
    # experimental_compile=True would have the added bonus of telling us if indeed it manages
    # to compile or not since "The experimental_compile API has must-compile semantics: either 
    # the entire function is compiled with XLA, or an errors.InvalidArgumentError exception is thrown."
    # See: https://www.tensorflow.org/xla#explicit_compilation_with_tffunction
    #TODO common.function passes the parameter experimental_relax_shapes=True by default. Maybe 
    # consider instead passing it as False for efficiency... This is most likely linked to the
    # input_signature TODO that follows
    #TODO (low priority) add an input_signature parameter so that tf.function knows what to expect
    # and won't adapt to the input if something strange happens (which it really shouldn't happen) 
    agent_1_train_function = common.function(tf_agent_1.train)
    agent_2_train_function = common.function(tf_agent_2.train)
    
    # replay buffer update for the driver
    replay_observer = [replay_buffer.add_batch]
    
    # Supposedly this is a performance improvement. According to TF devs it achieves
    # better performance by compiling stuff specialized on shape. If the shape of the stuff
    # going around changes a lot then it may actually get worse performance. To me it seems
    # that everything in our code runs with same shapes/batch sizes so I think it should be good.
    tf.config.optimizer.set_jit(True)
    for _ in range(num_iterations):
        # the two policies we use to collect data
        collect_policy_1 = tf_agent_1.collect_policy
        collect_policy_2 = tf_agent_2.collect_policy
                
        print('EPOCH {}'.format(epoch_counter.numpy()))
        tf.summary.scalar(name='Epsilon', data=decaying_epsilon_1(), step=epoch_counter)        
        # episode driver
        print('\nStarting to run the Driver')
        start_time = time.time()
        collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            [collect_policy_1, collect_policy_2],
            observers=replay_observer + train_metrics,
            num_episodes=collect_episodes_per_epoch).run()
        print('Finished running the Driver, it took {} seconds for {} episodes\n'.format(time.time() - start_time,
                                                                                           collect_episodes_per_epoch))
        
        # Dataset generates trajectories with shape [Bx2x...]
        # train for the first agent
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        print('Starting partial training of both Agents from Replay Buffer\nCounting Steps:')        

        losses_1 = tf.TensorArray(tf.float32, size=train_steps_per_epoch)
        losses_2 = tf.TensorArray(tf.float32, size=train_steps_per_epoch)
        c = 0
        start_time  = time.time()
        for data in dataset:
            if c % (train_steps_per_epoch/10) == 0 and c != 0:
                tf.summary.scalar("loss_agent_1", tf.math.reduce_mean(losses_1.stack()), step=train_step_1)
                tf.summary.scalar("loss_agent_2",  tf.math.reduce_mean(losses_1.stack()), step=train_step_2)
                print("{}% completed with {} steps done".format(int(c/train_steps_per_epoch*100), c))
            if c == train_steps_per_epoch:
                break
            experience, _ = data
            #FIXME _train and _loss functions of the two agents call the tf.summary to log the value
            # of various things. It seems though that they are writing on the same summary variables
            # because they're not build for the possibility of two agents in training. We need to change 
            # the agent class so that it can accept some agent_id string that it then uses to tf.name_scope
            # all the summaries. See line 482 in dqn_agent.py to understand what I mean by tf.name_scope
            #FIXME tensorflow documentation at https://www.tensorflow.org/tensorboard/migrate states that
            # default_writers do not cross the tf.function boundary and should instead be called as default
            # inside the tf.function. For now our code works because on the first run of the training function
            # the code is run in non-graph mode and thus "sees" the writer (and can then use it even in subsequent
            # graph-mode executions). This will stop working either if we start to export the compiled functions
            # so that we don't have the first "pythonic" run of them or if for some reason we change the file_writer
            # during execution. Should the summary writer be passed to the agent training function so that it can be set 
            # as default from inside the boundary of tf.function? How does tf-agent solve this issue?
            losses_1 = losses_1.write(c, agent_1_train_function(experience=experience).loss)
            losses_2 = losses_2.write(c, agent_2_train_function(experience=experience).loss)                
            c += 1

        losses_1 = losses_1.stack()
        losses_2 = losses_2.stack()
        print("Ended epoch training of both Agents, it took {}".format(time.time() - start_time))
        print('Mean loss for Agent 1 is: {}'.format(tf.math.reduce_mean(losses_1)))
        print('Mean loss for Agent 2 is: {}\n\n'.format(tf.math.reduce_mean(losses_2)))
        
        
        epoch_counter.assign_add(1)
        
        for train_metric in train_metrics:
            train_metric.tf_summaries(train_step=epoch_counter, step_metrics=train_metrics[:2])
        
        train_summary_writer.flush()

        # Checkpointing
        if (epoch_counter.numpy() % train_checkpoint_interval == 1) and not skip_checkpointing:
            train_checkpointer.save(global_step=epoch_counter.numpy() - 1)

        if (epoch_counter.numpy() % policy_checkpoint_interval == 1) and not skip_checkpointing:
            policy_checkpointer.save(global_step=epoch_counter.numpy() - 1)

        if (epoch_counter.numpy() % rb_checkpoint_interval == 1) and not skip_checkpointing:
            rb_checkpointer.save(global_step=epoch_counter.numpy() - 1)
        
        # Evaluation Run
        if (epoch_counter.numpy() % eval_interval == 1) and not skip_checkpointing:
            eval_py_policy_1 = tf_agent_1.policy
            eval_py_policy_2 = tf_agent_2.policy
            with eval_summary_writer.as_default(): 
                metric_utils.compute_summaries(eval_metrics, 
                                               eval_py_env,
                                               [eval_py_policy_1, eval_py_policy_2],
                                               num_episodes=num_eval_episodes,
                                               global_step=epoch_counter,
                                               tf_summaries=False,
                                               log = True
                                               )
                eval_summary_writer.flush()





def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    agent_class = dqn_agent.DdqnAgent if FLAGS.use_ddqn else dqn_agent.DqnAgent
    fc_layer_params = tuple([int(number) for number in FLAGS.network])
    train_eval(
        root_dir=FLAGS.root_dir,
        num_iterations=FLAGS.num_iterations,
        fc_layer_params=fc_layer_params,
        collect_episodes_per_epoch=FLAGS.collect_episodes_per_epoch,
        reset_at_step=FLAGS.reset_at_step,
        replay_buffer_capacity=FLAGS.rb_size,
        train_steps_per_epoch=FLAGS.train_steps_per_epoch,
        learning_rate=FLAGS.learning_rate,
        gradient_clipping=FLAGS.gradient_clipping,
        num_eval_episodes=FLAGS.num_eval_episodes,
        train_checkpoint_interval=FLAGS.checkpoint_interval,
        policy_checkpoint_interval=FLAGS.checkpoint_interval,
        rb_checkpoint_interval=FLAGS.checkpoint_interval,
        eval_interval=FLAGS.checkpoint_interval,
        agent_class=agent_class)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
