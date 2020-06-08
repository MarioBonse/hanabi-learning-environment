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
import tensorflow as tf
#from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
#from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
#from tf_agents.policies import py_tf_policy
from functools import partial

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
					'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_files', [],
						  'List of paths to gin configuration files (e.g.'
						  '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string('gin_bindings', [],
						  'Gin bindings to override the values set in the config files '
						  '(e.g. "train_eval.num_iterations=100").')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
	root_dir,
	num_iterations,
	# Params for collect
	collect_episodes_per_epoch,
	# Params for decaying Epsilon
	initial_epsilon,
	decay_type,
	decay_time,
	reset_at_step,
	# Params for train
	train_steps_per_epoch,
	batch_size,
	# Params for eval
	eval_interval,
	num_eval_episodes,
	# Params for checkpoints, summaries, and logging
	train_checkpoint_interval,
	policy_checkpoint_interval,
	rb_checkpoint_interval,
	summaries_flush_secs=10,
	):
	"""A simple train and eval for DQN."""
	root_dir = os.path.expanduser(root_dir)
	train_dir = os.path.join(root_dir, 'train')
	eval_dir = os.path.join(root_dir, 'eval')


	train_summary_writer = tf.summary.create_file_writer(
		train_dir, flush_millis=summaries_flush_secs * 1000)

	train_summary_writer.set_as_default()
	
	eval_summary_writer = tf.summary.create_file_writer(
		eval_dir, flush_millis=summaries_flush_secs * 1000)

	
	# Profiler is used to trace computational resource utilization if required from tensorboard
	# Note: "To profile multiple GPUs, install CUDA® Toolkit 10.2 or later. CUDA® Toolkit 10.1 supports only
	# single GPU profiling." (from https://www.tensorflow.org/guide/profiler#install_the_profiler_and_gpu_prerequisites)
	tf.profiler.experimental.server.start(6009)


	# create the enviroment
	env = utility.create_environment()                        
	tf_env = tf_py_environment.TFPyEnvironment(env)
	eval_py_env = tf_py_environment.TFPyEnvironment(utility.create_environment())

	
	train_step_1 = tf.Variable(0, trainable=False, name='global_step_1', dtype=tf.int64)
	train_step_2 = tf.Variable(0, trainable=False, name='global_step_2', dtype=tf.int64)
	epoch_counter = tf.Variable(0, trainable=False, name='Epoch', dtype=tf.int64)

	# Epsilon implementing decaying behaviour for the two agents
	decaying_epsilon_1 = partial(utility.decaying_epsilon,
								 initial_epsilon=initial_epsilon,
								 train_step=epoch_counter,
								 decay_type=decay_type,
								 decay_time=decay_time,
								 reset_at_step=reset_at_step)
	decaying_epsilon_2 = partial(utility.decaying_epsilon,
								 initial_epsilon=initial_epsilon,
								 train_step=epoch_counter,
								 decay_type=decay_type,
								 decay_time=decay_time,
								 reset_at_step=reset_at_step)
	

	# create an agent and a network 
	tf_agent_1 = utility.create_agent(environment=tf_env,
									  decaying_epsilon=decaying_epsilon_1,
									  train_step_counter=train_step_1)
	# Second agent. we can have as many as we want
	tf_agent_2 = utility.create_agent(environment=tf_env,
									  decaying_epsilon=decaying_epsilon_2,
									  train_step_counter=train_step_2)
	# replay buffer
	replay_buffer, prb_flag = utility.create_replay_buffer(
		data_spec=tf_agent_1.collect_data_spec,
		batch_size=tf_env.batch_size)

	

	# metrics
	train_metrics = [
		tf_metrics.NumberOfEpisodes(),
		tf_metrics.EnvironmentSteps(),
		tf_metrics.HanabiAverageReturnMetric(buffer_size=collect_episodes_per_epoch),
		tf_metrics.AverageEpisodeLengthMetric(buffer_size=collect_episodes_per_epoch),
	]

	# replay buffer update for the driver
	replay_observer = [replay_buffer.add_batch]
	
	eval_metrics = [
		tf_metrics.HanabiAverageReturnMetric(buffer_size=num_eval_episodes),
		tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
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

	
	print('\n\n\n')
	print(len(tf_agent_1._q_network._encoder._postprocessing_layers))
	print('\n\n\n')
	print(tf_agent_1._q_network._encoder._postprocessing_layers)
	print('\n\n\n')
	for layer in tf_agent_1._q_network._encoder._postprocessing_layers[1:]:
		print(layer.units)
	print('\n\n\n')
	print(allo)


	# Compiled version of training functions (much faster)
	agent_1_train_function = common.function(tf_agent_1.train)
	agent_2_train_function = common.function(tf_agent_2.train)
	
	
	# Supposedly this is a performance improvement. According to TF devs it achieves
	# better performance by compiling stuff specialized on shape. If the shape of the stuff
	# going around changes a lot then it may actually get worse performance. To me it seems
	# that everything in our code runs with same shapes/batch sizes so I think it should be good.
	# Would be nice if at some point someone took the time to actually test this :)
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
		"""
		TODO Performance Optimization: TF recommends not running the metrics at every step when running/training a model, but
			instead to run them every few steps. Our problem is that to know how many cards it plays in one episode (or how long an episode is)
			we need to keep track of what's happening at every single step with no exception. I thus suggest maybe logging the metrics every 
			couple of episodes instead of every episode so that we can gain (maybe, needs to be tested) some performance improvement without 
			loosing too much info on what's happening.
		"""
		dynamic_episode_driver.DynamicEpisodeDriver(
			tf_env,
			[collect_policy_1, collect_policy_2],
			observers=replay_observer + train_metrics,
			num_episodes=collect_episodes_per_epoch).run()
		print('Finished running the Driver, it took {} seconds for {} episodes\n'.format(time.time() - start_time,
																						   collect_episodes_per_epoch))
		
		
		"""
		TODO Performance Optimization: Try out different batch sizes (TF usually recommends higher batch size) and see how this influences
			performance, keeping track of possible differences in RAM/VRAM requirements. To do this properly the variable train_steps_per_epoch
			should be changed appropriately (e.g. double the batch size --> half the train_steps), but it would be nice to also check that this
			behaves as expected and doesn't impact per-epoch-learning. Per-epoch-learning is an abstract metric I just invented that would tell you
			how much better a model got after an epoch... Essentially one should check that the agent manages to reach the same level of performance
			(measured perhaps in average_return_per_episode == number of fireworks placed) at the same epoch (more or less) even if you do this thing
			of doubling batch_size and halving train_steps_per_epoch.
		FIXME Currently the num_parallel_calls passed changes depending on whether the replay buffer is uniform or prioritized. This is because passing
			a value higher than 1 with the Prioritized Replay Buffer will make the code crash(without ever ending execution, outputting no errors, and 
			not stucked in any loop (I have never witnessed this type of crash and it's extremely hard to investigate). My best guess to why this happens
			is because multiple parallel executions of the _get_next() method end up calling the same SumTree object (which isn't expressed in tensors) 
			probably creating some problems there. If one was masochistic enough to try and fix this I would suggest to start by writing the SumTree object
			from scratch in TF 2.x compliant code; it might also be necessary to remove the tf.py_function wrappers that I created in the Prioritized Replay
			Buffer code (which force TF to execute the functions eagerly as python functions). Note that the value "3" used in the case of a Uniform Replay Buffer
			is completely arbitrary.
		"""
		# Dataset generates trajectories with shape [Bx2x...]
		dataset = replay_buffer.as_dataset(
			num_parallel_calls=1 if prb_flag else 3,
			sample_batch_size=batch_size,
			num_steps=2).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		
		print('Starting partial training of both Agents from Replay Buffer\nCounting Steps:')        
		# Commenting out losses_1/_2 (and all their relevant code) to try and see if they are responsible for an observed memory leak. 
		# No feedback available yet on whether this is the case or not
		# losses_1 = tf.TensorArray(tf.float32, size=train_steps_per_epoch)
		# losses_2 = tf.TensorArray(tf.float32, size=train_steps_per_epoch)
		c = 0
		start_time  = time.time()
		for data in dataset:
			if c % (train_steps_per_epoch/10) == 0 and c != 0:
				#tf.summary.scalar("loss_agent_1", tf.math.reduce_mean(losses_1.stack()), step=train_step_1)
				#tf.summary.scalar("loss_agent_2",  tf.math.reduce_mean(losses_1.stack()), step=train_step_2)
				print("{}% completed with {} steps done".format(int(c/train_steps_per_epoch*100), c))
			if c == train_steps_per_epoch:
				break
			experience, data_info = data
			"""
			FIXME _train and _loss functions of the two agents call the tf.summary to log the value
				of various things. It seems though that they are writing on the same summary variables
				because they're not build for the possibility of two agents in training. We need to change 
				the agent class so that it can accept some agent_id string that it then uses to tf.name_scope
				all the summaries. See line 482 in dqn_agent.py to understand what I mean by tf.name_scope
			FIXME tensorflow documentation at https://www.tensorflow.org/tensorboard/migrate states that
				default_writers do not cross the tf.function boundary and should instead be called as default
				inside the tf.function. For now our code works because on the first run of the training function
				the code is run in non-graph mode and thus "sees" the writer (and can then use it even in subsequent
				graph-mode executions). This will stop working either if we start to export the compiled functions
				so that we don't have the first "pythonic" run of them or if for some reason we change the file_writer
				during execution. Should the summary writer be passed to the agent training function so that it can be set 
				as default from inside the boundary of tf.function? How does tf-agent solve this issue?
			"""

			# losses_1 = losses_1.write(c, agent_1_train_function(experience=experience).loss)
			# losses_2 = losses_2.write(c, agent_2_train_function(experience=experience).loss)
			loss_agent_1_info = agent_1_train_function(experience=experience)
			loss_agent_2_info = agent_2_train_function(experience=experience)
			if prb_flag:
				td_error_1 = tf.math.abs(loss_agent_1_info.extra.td_error)
				td_error_2 = tf.math.abs(loss_agent_2_info.extra.td_error)
				priorities = tf.math.reduce_mean([td_error_1, td_error_2], axis=0)
				indices = data_info.ids[:, 0]
				replay_buffer.tf_set_priority(indices, priorities)
			c += 1

		# losses_1 = losses_1.stack()
		# losses_2 = losses_2.stack()
		print("Ended epoch training of both Agents, it took {}".format(time.time() - start_time))
		# print('Mean loss for Agent 1 is: {}'.format(tf.math.reduce_mean(losses_1)))
		# print('Mean loss for Agent 2 is: {}\n\n'.format(tf.math.reduce_mean(losses_2)))
		
		
		epoch_counter.assign_add(1)
		
		for train_metric in train_metrics:
			train_metric.tf_summaries(train_step=epoch_counter, step_metrics=train_metrics[:2])
		
		# Resetting average return and average episode length metrics so that the average only refers to current epoch
		# This is actually probably redundant because they have a buffer_size which only allows them to keep track of 
		# exactly the number of episodes that are run each epoch so even without getting resetted they probably would
		# overwrite old values. Since resetting doesn't really take time and is clearer, we keep it.
		train_metrics[2].reset()
		train_metrics[3].reset()
		
		train_summary_writer.flush()

		# Checkpointing
		if epoch_counter.numpy() % train_checkpoint_interval == 0:
			train_checkpointer.save(global_step=epoch_counter.numpy())

		if epoch_counter.numpy() % policy_checkpoint_interval == 0:
			policy_checkpointer.save(global_step=epoch_counter.numpy())

		if epoch_counter.numpy() % rb_checkpoint_interval == 0:
			rb_checkpointer.save(global_step=epoch_counter.numpy())
		
		# Evaluation Run
		if epoch_counter.numpy() % eval_interval == 0:
			eval_py_policy_1 = tf_agent_1.policy
			eval_py_policy_2 = tf_agent_2.policy
			metric_utils.eager_compute(eval_metrics,
									   eval_py_env,
									   [eval_py_policy_1, eval_py_policy_2],
									   num_episodes=num_eval_episodes,
									   train_step=epoch_counter,
									   summary_writer=eval_summary_writer,
									   summary_prefix='Metrics')
			eval_summary_writer.flush()
			# Resetting average return and average episode length metrics so that the average only refers to current epoch
			# This is actually probably redundant because they have a buffer_size which only allows them to keep track of 
			# exactly the number of episodes that are run each epoch so even without getting resetted they probably would
			# overwrite old values. Since resetting doesn't really take time and is clearer, we keep it.
			eval_metrics[0].reset()
			eval_metrics[1].reset()





def main(_):
	logging.set_verbosity(logging.INFO)
	utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
	train_eval(root_dir=FLAGS.root_dir,)


if __name__ == '__main__':
	flags.mark_flag_as_required('root_dir')
	app.run(main)
