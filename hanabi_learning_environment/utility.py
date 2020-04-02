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


"""
FEDE COMMENT
Questa è da modificare con i tf.agents invece di quelli di DeepMind
"""
'''
@gin.configurable
def create_agent(environment, obs_stacker, agent_type='DQN'):
  """Creates the Hanabi agent.

  Args:
    environment: The environment.
    obs_stacker: Observation stacker object.
    agent_type: str, type of agent to construct.

  Returns:
    An agent for playing Hanabi.

  Raises:
    ValueError: if an unknown agent type is requested.
  """
  if agent_type == 'DQN':
    return dqn_agent.DQNAgent(observation_size=obs_stacker.observation_size(),
                              num_actions=environment.num_moves(),
                              num_players=environment.players)
  elif agent_type == 'Rainbow':
    return rainbow_agent.RainbowAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  else:
    raise ValueError('Expected valid agent_type, got {}'.format(agent_type))

'''


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

    return current_player, legal_moves, observation_vector


def decaying_epsilon(initial_epsilon, train_step, decay_time):
    return initial_epsilon*(tf.math.pow(0.5, (tf.math.floordiv(train_step, decay_time))))