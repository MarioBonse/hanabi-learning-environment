# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""Run methods for training a DQN agent on Atari.

Methods in this module are usually referenced by |train.py|.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
from hanabi_learning_environment import rl_env
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
import numpy as np
import tensorflow as tf


class ObservationStacker(object):
    """Class for stacking agent observations."""

    def __init__(self, history_size, observation_size, num_players):
        """Initializer for observation stacker.

        Args:
          history_size: int, number of time steps to stack.
          observation_size: int, size of observation vector on one time step.
          num_players: int, number of players.
        """
        self._history_size = history_size
        self._observation_size = observation_size
        self._num_players = num_players
        self._obs_stacks = list()
        for _ in range(0, self._num_players):
            self._obs_stacks.append(np.zeros(self._observation_size *
                                             self._history_size))

    def add_observation(self, observation, current_player):
        """Adds observation for the current player.

        Args:
          observation: observation vector for current player.
          current_player: int, current player id.
        """
        self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                                   -self._observation_size)
        self._obs_stacks[current_player][(self._history_size - 1) *
                                         self._observation_size:] = observation

    def get_observation_stack(self, current_player):
        """Returns the stacked observation for current player.

        Args:
          current_player: int, current player id.
        """

        return self._obs_stacks[current_player]

    def reset_stack(self):
        """Resets the observation stacks to all zero."""

        for i in range(0, self._num_players):
            self._obs_stacks[i].fill(0.0)

    @property
    def history_size(self):
        """Returns number of steps to stack."""
        return self._history_size

    def observation_size(self):
        """Returns the size of the observation vector after history stacking."""
        return self._observation_size * self._history_size


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

    Args:
      gin_files: A list of paths to the gin configuration files for this
        experiment.
      gin_bindings: List of gin parameter bindings to override the values in the
        config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


@gin.configurable
def create_environment(game_type='Hanabi-Full', num_players=2):
    """Creates the Hanabi environment.

    Args:
      game_type: Type of game to play. Currently the following are supported:
        Hanabi-Full: Regular game.
        Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.
      num_players: Int, number of players to play this game.

    Returns:
      A Hanabi environment.
    """
    return rl_env.make(
        environment_name=game_type, num_players=num_players, pyhanabi_path=None)


@gin.configurable
def create_obs_stacker(environment, history_size=4):
    """Creates an observation stacker.

    Args:
      environment: environment object.
      history_size: int, number of steps to stack.

    Returns:
      An observation stacker object.
    """

    return ObservationStacker(history_size,
                              environment.vectorized_observation_shape()[0],
                              environment.players)


@gin.configurable
def create_agent(agent_class,
                 environment,
                 learning_rate,
                 decaying_epsilon,
                 target_update_tau,
                 target_update_period,
                 gamma,
                 reward_scale_factor,
                 gradient_clipping,
                 debug_summaries,
                 summarize_grads_and_vars,
                 train_step_counter):
    """Creates the Hanabi agent.

    Args:
      agent_class: str, type of agent to construct.
      environment: The environment.
      learning_rate: The Learning Rate
      decaying_epsilon: Epsilon for Epsilon Greedy Policy
      target_update_tau: Agent parameter
      target_update_period: Agent parameter
      gamma: Agent parameter
      reward_scale_factor: Agent parameter
      gradient_clipping: Agent parameter
      debug_summaries: Agent parameter
      summarize_grads_and_vars: Agent parameter
      train_step_counter: The train step tf.Variable to be passed to agent


    Returns:
      An agent for playing Hanabi.

    Raises:
      ValueError: if an unknown agent type is requested.
    """
    if agent_class == 'DQN':
        return dqn_agent.DqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            q_network=q_network.QNetwork(environment.time_step_spec().observation['observations'],
                                         environment.action_spec()
                                         ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
    elif agent_class == 'DDQN':
        return dqn_agent.DdqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            q_network=q_network.QNetwork(environment.time_step_spec().observation['observations'],
                                         environment.action_spec(),
                                         ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
    else:
        raise ValueError(
            'Expected valid agent_type, got {}'.format(agent_class))


def format_legal_moves(legal_moves, action_dim):
    """Returns formatted legal moves.

    This function takes a list of actions and converts it into a fixed size vector
    of size action_dim. If an action is legal, its position is set to 0 and -Inf
    otherwise.
    Ex: legal_moves = [0, 1, 3], action_dim = 5
        returns [0, 0, -Inf, 0, -Inf]

    Args:
      legal_moves: list of legal actions.
      action_dim: int, number of actions.

    Returns:
      a vector of size action_dim.
    """
    new_legal_moves = np.full(action_dim, -float('inf'))
    if legal_moves:
        new_legal_moves[legal_moves] = 0
    return new_legal_moves


# Only works for Full Hanabi (5 colors, 5 cards in hand)
def transform_obs(obs):
    game_obs = [obs["current_player"], obs["life_tokens"], obs["information_tokens"],
                obs["num_players"], obs["deck_size"], obs["fireworks"]['R'],
                obs["fireworks"]['Y'], obs["fireworks"]['G'], obs["fireworks"]['W'],
                obs["fireworks"]['B']]

    color_order = ['R', 'Y', 'G', 'W', 'B']
    hands_obs = [[color_order.index(card['color']), card['rank']]
                 for hand in obs["observed_hands"] for card in hand]
    for hand in obs["observed_hands"]:
        hand_obs = []
        for card in hand:
            color = color_order.index(card['color']) if card['color'] else -1
            rank = card['rank'] if card['rank'] else -1
            hand_obs.append([color, rank])
        hands_obs.append(hand_obs)

    
    knowledge_obs = []
    for player_hints in obs["card_knowledge"]:
        hints_obs = []
        for hint in player_hints:
            color = color_order.index(hint['color']) if hint['color'] else -1
            rank = hint['rank'] if hint['rank'] else -1
            hints_obs.append([color, rank])
        knowledge_obs.append(hints_obs)
    
    assert np.array(knowledge_obs).shape == (2, 5)
    assert np.array(hands_obs).shape == (10)
    
    return [game_obs, hands_obs, knowledge_obs]


def parse_observations(observations, num_actions, obs_stacker):
    """Deconstructs the rich observation data into relevant components.

    Args:
      observations: dict, containing full observations.
      num_actions: int, The number of available actions.
      obs_stacker: Observation stacker object.

    Returns:
      current_player: int, Whose turn it is.
      legal_moves: `np.array` of floats, of length num_actions, whose elements
        are -inf for indices corresponding to illegal moves and 0, for those
        corresponding to legal moves.
      observation_vector: Vectorized observation for the current player.
    """
    current_player = observations['current_player']
    current_player_observation = (
        observations['player_observations'][current_player])

    legal_moves = current_player_observation['legal_moves_as_int']
    legal_moves = format_legal_moves(legal_moves, num_actions)

    observation_vector = current_player_observation['vectorized']
    obs_stacker.add_observation(observation_vector, current_player)
    observation_vector = obs_stacker.get_observation_stack(current_player)

    # These observations are meant for rule-based agents only. Note that they only carry information
    # about the state of the game at this timestep without any history of what happened.
    non_encoded_obs = transform_obs(current_player_observation)

    return current_player, legal_moves, observation_vector, non_encoded_obs


# TODO Implementing an automatic reset of the decaying epsilon parameter? Something maybe that looks at the variance
# of some performance metric(s) and decides based on that if it should reset the decay of epsilon or not
@gin.configurable
def decaying_epsilon(initial_epsilon, train_step, decay_time, decay_type='exponential', reset_at_step=None):
    if reset_at_step:
        # The reason why these two ifs are separated and not grouped in an *and* expression (using python short-circuit)
        # is because this function might actually get optimized by tf.function since it's called inside the agent training function
        # and I think short-circuiting doesn't work in graph mode.
        if reset_at_step <= train_step:
            # Notice that this doesn't change the train_step outside the scope of this function
            # (which is the desired behaviour)
            train_step = train_step - reset_at_step
    if decay_type == 'exponential':
        decay = 0.5 ** tf.cast((train_step // decay_time), tf.float32)
    elif decay_type == None:
        decay = 1
    return initial_epsilon*decay


def observation_and_action_constraint_splitter(obs):
    return obs['observations'], obs['legal_moves']
