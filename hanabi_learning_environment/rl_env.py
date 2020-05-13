# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RL environment for Hanabi, using an API similar to OpenAI Gym."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.pyhanabi import color_char_to_idx

import abc
from .utility import *
import tensorflow as tf
import numpy as np
import gin.tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]

#-------------------------------------------------------------------------------
# Environment API
#-------------------------------------------------------------------------------

@gin.configurable
class HanabiEnv(py_environment.PyEnvironment):
    """RL interface to a Hanabi environment.

    ```python

    environment = rl_env.make()
    config = { 'players': 5 }
    observation = environment.reset(config)
    while not done:
            # Agent takes action
            action =  ...
            # Environment take a step
            observation, reward, done, info = environment.step(action)
    ```
    """

    def __init__(self, config, gamma=0.95, history_size=4, alternative_reward=False):
        r"""Creates an environment with the given game configuration.

        Args:
            config: dict, With parameters for the game. Config takes the following
                keys and values.
                    - colors: int, Number of colors \in [2,5].
                    - ranks: int, Number of ranks \in [2,5].
                    - players: int, Number of players \in [2,5].
                    - hand_size: int, Hand size \in [4,5].
                    - max_information_tokens: int, Number of information tokens (>=0).
                    - max_life_tokens: int, Number of life tokens (>=1).
                    - observation_type: int.
                        0: Minimal observation.
                        1: First-order common knowledge observation.
                    - seed: int, Random seed.
                    - random_start_player: bool, Random start player.
            gamma: float in [0,1], discount parameter for past rewards
            history_size: int, number of past observations to concatenate (1 means only current obs)
            alternative_reward: bool, flag for alternative reward, 
                    i.e.  reward at end of game = achieved points - max_points + 1
        """
        super().__init__()
        assert isinstance(config, dict), "Expected config to be of type dict."
        self.game = pyhanabi.HanabiGame(config)
        n_colors = config['colors']        
        self.gamma = gamma
        self.alternative_reward = alternative_reward
        self.observation_encoder = pyhanabi.ObservationEncoder(
                self.game, pyhanabi.ObservationEncoderType.CANONICAL)
        self.players = self.game.num_players()
        self.obs_stacker = create_obs_stacker(self, history_size=history_size)
        self._observation_spec = {'observations': array_spec.ArraySpec(shape=(self.obs_stacker.observation_size(),), dtype=np.float64),
                                  'legal_moves': array_spec.ArraySpec(shape=(self.num_moves(),), dtype=np.bool_),
                                  'game_obs': array_spec.ArraySpec(shape=(5+n_colors,), dtype=np.int64),
                                  'hand_obs': array_spec.ArraySpec(shape=(2, n_colors, 2), dtype=np.int64),
                                  'knowledge_obs': array_spec.ArraySpec(shape=(2, n_colors, 2), dtype=np.int64)}
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int_, minimum=0, maximum=self.num_moves() - 1)

    def observation_spec(self):
        return self._observation_spec
    
    def action_spec(self):
        return self._action_spec

    def _reset(self):
        r"""Resets the environment for a new game.

        Returns:
            observation: dict, containing the full observation about the game at the
                current step. *WARNING* This observation contains all the hands of the
                players and should not be passed to the agents.
                An example observation:
                {'current_player': 0,
                 'player_observations': [{'current_player': 0,
                                                                    'current_player_offset': 0,
                                                                    'deck_size': 40,
                                                                    'discard_pile': [],
                                                                    'fireworks': {'B': 0,
                                                                                                'G': 0,
                                                                                                'R': 0,
                                                                                                'W': 0,
                                                                                                'Y': 0},
                                                                    'information_tokens': 8,
                                                                    'legal_moves': [{'action_type': 'PLAY',
                                                                                                     'card_index': 0},
                                                                                                    {'action_type': 'PLAY',
                                                                                                     'card_index': 1},
                                                                                                    {'action_type': 'PLAY',
                                                                                                     'card_index': 2},
                                                                                                    {'action_type': 'PLAY',
                                                                                                     'card_index': 3},
                                                                                                    {'action_type': 'PLAY',
                                                                                                     'card_index': 4},
                                                                                                    {'action_type':
                                                                                                    'REVEAL_COLOR',
                                                                                                     'color': 'R',
                                                                                                     'target_offset': 1},
                                                                                                    {'action_type':
                                                                                                    'REVEAL_COLOR',
                                                                                                     'color': 'G',
                                                                                                     'target_offset': 1},
                                                                                                    {'action_type':
                                                                                                    'REVEAL_COLOR',
                                                                                                     'color': 'B',
                                                                                                     'target_offset': 1},
                                                                                                    {'action_type': 'REVEAL_RANK',
                                                                                                     'rank': 0,
                                                                                                     'target_offset': 1},
                                                                                                    {'action_type': 'REVEAL_RANK',
                                                                                                     'rank': 1,
                                                                                                     'target_offset': 1},
                                                                                                    {'action_type': 'REVEAL_RANK',
                                                                                                     'rank': 2,
                                                                                                     'target_offset': 1}],
                                                                    'life_tokens': 3,
                                                                    'observed_hands': [[{'color': None, 'rank':
                                                                    -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1}],
                                                                                                         [{'color': 'G', 'rank': 2},
                                                                                                            {'color': 'R', 'rank': 0},
                                                                                                            {'color': 'R', 'rank': 1},
                                                                                                            {'color': 'B', 'rank': 0},
                                                                                                            {'color': 'R', 'rank':
                                                                                                            1}]],
                                                                    'num_players': 2,
                                                                    'vectorized': [ 0, 0, 1, ... ]},
                                                                 {'current_player': 0,
                                                                    'current_player_offset': 1,
                                                                    'deck_size': 40,
                                                                    'discard_pile': [],
                                                                    'fireworks': {'B': 0,
                                                                                                'G': 0,
                                                                                                'R': 0,
                                                                                                'W': 0,
                                                                                                'Y': 0},
                                                                    'information_tokens': 8,
                                                                    'legal_moves': [],
                                                                    'life_tokens': 3,
                                                                    'observed_hands': [[{'color': None, 'rank':
                                                                    -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1},
                                                                                                            {'color': None, 'rank':
                                                                                                            -1}],
                                                                                                         [{'color': 'W', 'rank': 2},
                                                                                                            {'color': 'Y', 'rank': 4},
                                                                                                            {'color': 'Y', 'rank': 2},
                                                                                                            {'color': 'G', 'rank': 0},
                                                                                                            {'color': 'W', 'rank':
                                                                                                            1}]],
                                                                    'num_players': 2,
                                                                    'vectorized': [ 0, 0, 1, ... ]}]}
        """
        self.state = self.game.new_initial_state()

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        obs = self._make_observation_all_players()
        obs["current_player"] = self.state.cur_player()
        current_player, legal_moves, current_agent_obs, non_encoded_obs = parse_observations(obs, self.num_moves(), self.obs_stacker)
        
        observations_and_legal_moves = {'observations': current_agent_obs,
                                        'legal_moves': np.logical_not(legal_moves),
                                        'game_obs': non_encoded_obs[0],
                                        'hand_obs': non_encoded_obs[1],
                                        'knowledge_obs': non_encoded_obs[2]}
        return ts.restart(observations_and_legal_moves)

    def _step(self, action):
        """Take one step in the game.

        Args:
            action: dict, mapping to a legal action taken by an agent. The following
                actions are supported:
                    - { 'action_type': 'PLAY', 'card_index': int }
                    - { 'action_type': 'DISCARD', 'card_index': int }
                    - {
                            'action_type': 'REVEAL_COLOR',
                            'color': str,
                            'target_offset': int >=0
                        }
                    - {
                            'action_type': 'REVEAL_RANK',
                            'rank': str,
                            'target_offset': int >=0
                        }
                Alternatively, action may be an int in range [0, num_moves()).

        Returns:
            observation: dict, containing the full observation about the game at the
                current step. *WARNING* This observation contains all the hands of the
                players and should not be passed to the agents.
                An example observation:
                {'current_player': 0,
                 'player_observations': [{'current_player': 0,
                                                        'current_player_offset': 0,
                                                        'deck_size': 40,
                                                        'discard_pile': [],
                                                        'fireworks': {'B': 0,
                                                                            'G': 0,
                                                                            'R': 0,
                                                                            'W': 0,
                                                                            'Y': 0},
                                                        'information_tokens': 8,
                                                        'legal_moves': [{'action_type': 'PLAY',
                                                                                 'card_index': 0},
                                                                                {'action_type': 'PLAY',
                                                                                 'card_index': 1},
                                                                                {'action_type': 'PLAY',
                                                                                 'card_index': 2},
                                                                                {'action_type': 'PLAY',
                                                                                 'card_index': 3},
                                                                                {'action_type': 'PLAY',
                                                                                 'card_index': 4},
                                                                                {'action_type': 'REVEAL_COLOR',
                                                                                 'color': 'R',
                                                                                 'target_offset': 1},
                                                                                {'action_type': 'REVEAL_COLOR',
                                                                                 'color': 'G',
                                                                                 'target_offset': 1},
                                                                                {'action_type': 'REVEAL_COLOR',
                                                                                 'color': 'B',
                                                                                 'target_offset': 1},
                                                                                {'action_type': 'REVEAL_RANK',
                                                                                 'rank': 0,
                                                                                 'target_offset': 1},
                                                                                {'action_type': 'REVEAL_RANK',
                                                                                 'rank': 1,
                                                                                 'target_offset': 1},
                                                                                {'action_type': 'REVEAL_RANK',
                                                                                 'rank': 2,
                                                                                 'target_offset': 1}],
                                                        'life_tokens': 3,
                                                        'observed_hands': [[{'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1}],
                                                                                     [{'color': 'G', 'rank': 2},
                                                                                        {'color': 'R', 'rank': 0},
                                                                                        {'color': 'R', 'rank': 1},
                                                                                        {'color': 'B', 'rank': 0},
                                                                                        {'color': 'R', 'rank': 1}]],
                                                        'num_players': 2,
                                                        'vectorized': [ 0, 0, 1, ... ]},
                                                     {'current_player': 0,
                                                        'current_player_offset': 1,
                                                        'deck_size': 40,
                                                        'discard_pile': [],
                                                        'fireworks': {'B': 0,
                                                                            'G': 0,
                                                                            'R': 0,
                                                                            'W': 0,
                                                                            'Y': 0},
                                                        'information_tokens': 8,
                                                        'legal_moves': [],
                                                        'life_tokens': 3,
                                                        'observed_hands': [[{'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1},
                                                                                        {'color': None, 'rank': -1}],
                                                                                     [{'color': 'W', 'rank': 2},
                                                                                        {'color': 'Y', 'rank': 4},
                                                                                        {'color': 'Y', 'rank': 2},
                                                                                        {'color': 'G', 'rank': 0},
                                                                                        {'color': 'W', 'rank': 1}]],
                                                        'num_players': 2,
                                                        'vectorized': [ 0, 0, 1, ... ]}]}
            reward: float, Reward obtained from taking the action.
            done: bool, Whether the game is done.
            info: dict, Optional debugging information.

        Raises:
            AssertionError: When an illegal action is provided.
        """
        #action = int(np.argmax(action))
        if self._current_time_step.is_last():
            return self.reset()
        elif isinstance(action, np.ndarray) and (action.dtype == np.int64):
            # Convert int action into a Hanabi move.
            action = self.game.get_move(int(action))
        else:
            print(type(action), action.dtype)
            print(action)
            raise ValueError("Expected action as int, got: {}".format(
                    action))

        last_score = self.state.score()
        # Apply the action to the state.
        self.state.apply_move(action)

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        """
        Player turns are currently managed by both driver and environment. In the 2 player case the env is keeping track of two 
        players (PLAYER 0, PLAYER 1) and for the env PLAYER 0 always starts the game. The env will also make sure to feed
        in order and alternating the observations of PLAYER 0 and PLAYER 1. The driver on the other hand has two policies
        (and thus two agents) which we can call AGENT 1, AGENT 2. It is important to note that in any given episode AGENT 1 
        could be assigned to PLAYER 0 or PLAYER 1. This assignment isn't "really" random though; in the first episode run by the
        driver AGENT 1 will be always assigned to PLAYER 0 whilst in subsequent episodes it will be the AGENT who made the last move
        in the previous episode to be assigned to PLAYER 0. Since episodes vary in length this "kinda" insures that not always the same
        player will start the game... It is perhaps a not so nice way to do it though. Also note that the assignment AGENT i - PLAYER j 
        happens implicitly, the driver isn't aware of the existence of PLAYER 0,1 and the env isn't aware of the existence of AGENT 0,1.
        """

        obs = self._make_observation_all_players()
        current_player, legal_moves, current_agent_obs, non_encoded_obs = parse_observations(obs, self.num_moves(), self.obs_stacker)
        done = self.state.is_terminal()
        
        # Note that DeepMind implementation of the reward (alternative_reward=False) has self.state.score() = 0 at the end of a 
        # "failed" game. This means that if you reach 24/25 points at the last time-step you get -24 points. However for some wierd reason
        # if instead you fail the game by deck exhaustion (instead of life tokens = 0) then self.state.score() = fireworks placed. Therefore
        # the last reward in this case would be 0 (if you exhausted deck with discard/wrong play) or 1 (if you exhausted deck with a correct play).
        if self.alternative_reward and done:
            reward = self.state.score() - self.game.num_colors()*self.game.num_ranks() + 1
        else:
            # Reward is score differential. May be large and negative at game end.
            reward = self.state.score() - last_score

        observations_and_legal_moves = {'observations': current_agent_obs,
                                        'legal_moves': np.logical_not(legal_moves),
                                        'game_obs': non_encoded_obs[0],
                                        'hand_obs': non_encoded_obs[1],
                                        'knowledge_obs': non_encoded_obs[2]}

        if done:
            return ts.termination(observations_and_legal_moves, reward)
        else:
            return ts.transition(observations_and_legal_moves, reward, self.gamma)

    def vectorized_observation_shape(self):
        """Returns the shape of the vectorized observation.

        Returns:
            A list of integer dimensions describing the observation shape.
        """
        return self.observation_encoder.shape()

    def num_moves(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
            Integer, number of moves.
        """
        return self.game.max_moves()

    def _make_observation_all_players(self):
        """Make observation for all players.

        Returns:
            dict, containing observations for all players.
        """
        obs = {}
        player_observations = [self._extract_dict_from_backend(
                player_id, self.state.observation(player_id))
                for player_id in range(self.players)]  # pylint: disable=bad-continuation
        obs["player_observations"] = player_observations
        obs["current_player"] = self.state.cur_player()
        return obs

    def _extract_dict_from_backend(self, player_id, observation):
        """Extract a dict of features from an observation from the backend.

        Args:
            player_id: Int, player from whose perspective we generate the observation.
            observation: A `pyhanabi.HanabiObservation` object.

        Returns:
            obs_dict: dict, mapping from HanabiObservation to a dict.
        """
        obs_dict = {}
        obs_dict["current_player"] = self.state.cur_player()
        obs_dict["current_player_offset"] = observation.cur_player_offset()
        obs_dict["life_tokens"] = observation.life_tokens()
        obs_dict["information_tokens"] = observation.information_tokens()
        obs_dict["num_players"] = observation.num_players()
        obs_dict["deck_size"] = observation.deck_size()

        obs_dict["fireworks"] = {}
        fireworks = self.state.fireworks()
        for color, firework in zip(pyhanabi.COLOR_CHAR, fireworks):
            obs_dict["fireworks"][color] = firework

        obs_dict["legal_moves"] = []
        obs_dict["legal_moves_as_int"] = []
        for move in observation.legal_moves():
            obs_dict["legal_moves"].append(move.to_dict())
            obs_dict["legal_moves_as_int"].append(self.game.get_move_uid(move))

        obs_dict["observed_hands"] = []
        for player_hand in observation.observed_hands():
            cards = [card.to_dict() for card in player_hand]
            obs_dict["observed_hands"].append(cards)

        obs_dict["discard_pile"] = [
                card.to_dict() for card in observation.discard_pile()
        ]

        # Return hints received.
        obs_dict["card_knowledge"] = []
        for player_hints in observation.card_knowledge():
            player_hints_as_dicts = []
            for hint in player_hints:
                hint_d = {}
                if hint.color() is not None:
                    hint_d["color"] = pyhanabi.color_idx_to_char(hint.color())
                else:
                    hint_d["color"] = None
                hint_d["rank"] = hint.rank()
                player_hints_as_dicts.append(hint_d)
            obs_dict["card_knowledge"].append(player_hints_as_dicts)

        # ipdb.set_trace()
        obs_dict["vectorized"] = self.observation_encoder.encode(observation)
        obs_dict["pyhanabi"] = observation

        return obs_dict

    def _build_move(self, action):
        """Build a move from an action dict.

        Args:
            action: dict, mapping to a legal action taken by an agent. The following
                actions are supported:
                    - { 'action_type': 'PLAY', 'card_index': int }
                    - { 'action_type': 'DISCARD', 'card_index': int }
                    - {
                            'action_type': 'REVEAL_COLOR',
                            'color': str,
                            'target_offset': int >=0
                        }
                    - {
                            'action_type': 'REVEAL_RANK',
                            'rank': str,
                            'target_offset': int >=0
                        }

        Returns:
            move: A `HanabiMove` object constructed from action.

        Raises:
            ValueError: Unknown action type.
        """
        assert isinstance(action, dict), "Expected dict, got: {}".format(action)
        assert "action_type" in action, ("Action should contain `action_type`. "
                                                                         "action: {}").format(action)
        action_type = action["action_type"]
        assert (action_type in MOVE_TYPES), (
                "action_type: {} should be one of: {}".format(action_type, MOVE_TYPES))

        if action_type == "PLAY":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_play_move(card_index=card_index)
        elif action_type == "DISCARD":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_discard_move(card_index=card_index)
        elif action_type == "REVEAL_RANK":
            target_offset = action["target_offset"]
            rank = action["rank"]
            move = pyhanabi.HanabiMove.get_reveal_rank_move(
                    target_offset=target_offset, rank=rank)
        elif action_type == "REVEAL_COLOR":
            target_offset = action["target_offset"]
            assert isinstance(action["color"], str)
            color = color_char_to_idx(action["color"])
            move = pyhanabi.HanabiMove.get_reveal_color_move(
                    target_offset=target_offset, color=color)
        else:
            raise ValueError("Unknown action_type: {}".format(action_type))

        legal_moves = self.state.legal_moves()
        assert (str(move) in map(
                str,
                legal_moves)), "Illegal action: {}. Move should be one of : {}".format(
                        move, legal_moves)

        return move


def make(environment_name="Hanabi-Full", num_players=2, pyhanabi_path=None):
    """Make an environment.

    Args:
        environment_name: str, Name of the environment to instantiate.
        num_players: int, Number of players in this game.
        pyhanabi_path: str, absolute path to header files for c code linkage.

    Returns:
        env: An `Environment` object.

    Raises:
        ValueError: Unknown environment name.
    """

    if pyhanabi_path is not None:
        prefixes=(pyhanabi_path,)
        assert pyhanabi.try_cdef(prefixes=prefixes), "cdef failed to load"
        assert pyhanabi.try_load(prefixes=prefixes), "library failed to load"
    if (environment_name == "Hanabi-Full"):
        return HanabiEnv(
                config={
                        "colors":
                                5,
                        "ranks":
                                5,
                        "players":
                                num_players,
                        "max_information_tokens":
                                8,
                        "max_life_tokens":
                                3,
                        "observation_type":
                                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
                })
    elif environment_name == "Hanabi-Full-CardKnowledge":
            return HanabiEnv(
                config={
                        "colors":
                                5,
                        "ranks":
                                5,
                        "players":
                                num_players,
                        "max_information_tokens":
                                8,
                        "max_life_tokens":
                                3,
                        "observation_type":
                                pyhanabi.AgentObservationType.SEER.value
                })
    elif environment_name == "Hanabi-Full-Minimal":
        return HanabiEnv(
                config={
                        "colors": 5,
                        "ranks": 5,
                        "players": num_players,
                        "max_information_tokens": 8,
                        "max_life_tokens": 3,
                        "observation_type": pyhanabi.AgentObservationType.MINIMAL.value
                })
    elif environment_name == "Hanabi-Small-CardKnowledge":
        return HanabiEnv(
                config={
                        "colors":
                                3,
                        "ranks":
                                5,
                        "players":
                                num_players,
                        "hand_size":
                                3,
                        "max_information_tokens":
                                8,
                        "max_life_tokens":
                                3,
                        "observation_type":
                                pyhanabi.AgentObservationType.SEER.value
                })
    elif environment_name == "Hanabi-Very-Small":
        return HanabiEnv(
                config={
                        "colors":
                                1,
                        "ranks":
                                5,
                        "players":
                                num_players,
                        "hand_size":
                                2,
                        "max_information_tokens":
                                3,
                        "max_life_tokens":
                                1,
                        "observation_type":
                                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
                })
    else:
        raise ValueError("Unknown environment {}".format(environment_name))

