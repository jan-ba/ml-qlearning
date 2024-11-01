from collections import defaultdict
from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np
from typing import TypedDict

class GameState(TypedDict):
    player_y: int
    player_vel: int
    next_pipe_dist_to_player: float
    next_pipe_top_y: int
    next_pipe_bottom_y: int
    next_next_pipe_dist_to_player: float
    next_next_pipe_top_y: int
    next_next_pipe_bottom_y: int

# Lecture 17, slide 38 for deep q learning

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        return
    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    
    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)
        print("episode left: ", nb_episodes)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        agent.observe(state, action, reward, newState, env.game_over())

        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


class FlappyAgentV1(FlappyAgent):
    def __init__(self):
        super().__init__()
        self.DISCOUNT_RATE_GAMMA = 1
        self.LEARNING_RATE_ALPHA = 0.1
        self.E_GREEDY_EPSILON = 0.1
        self.num_intervals = 15
        self.pixels_y = 512
        self.pixels_x = 288
        self.velocity_max = 10
        self.velocity_min = -8
        self.q_table = dict([((player_int, next_pipe_int, dist_int, vel), [0, 0]) for player_int in\
                             range(self.num_intervals) for next_pipe_int in range(self.num_intervals) for dist_int in\
                                  range(self.num_intervals) for vel in range(self.velocity_min, self.velocity_max + 1)])
      
    
    def clamp_state(self, state):

        # Clamp state values between min and max (because of some bug in the environment)

        vel = int(max(min(self.velocity_max, state["player_vel"]), self.velocity_min))

        player_y = int(max(min(self.pixels_y, state["player_y"]), 0))
        x_dist_next = int(max(min(self.pixels_x, state["next_pipe_dist_to_player"]), 0))
        y_top_next = int(max(min(self.pixels_y, state["next_pipe_top_y"]), 0))

        y_bottom_next = int(max(min(self.pixels_y, state["next_pipe_bottom_y"]), 0))
        x_dist_next_next = int(max(min(self.pixels_x, state["next_next_pipe_dist_to_player"]), 0))

        y_top_next_next = int(max(min(self.pixels_y, state["next_next_pipe_top_y"]), 0))
        y_bottom_next_next = int(max(min(self.pixels_y, state["next_next_pipe_bottom_y"]), 0))
        return GameState(player_vel=vel, player_y=player_y, next_pipe_dist_to_player=x_dist_next, next_pipe_top_y=y_top_next,\
                         next_pipe_bottom_y=y_bottom_next, next_next_pipe_dist_to_player=x_dist_next_next,\
                            next_next_pipe_top_y=y_top_next_next, next_next_pipe_bottom_y=y_bottom_next_next)

    def get_binned_state(self, state: GameState):
        clamped_state = self.clamp_state(state)
        
        binned_player_vel = int(clamped_state["player_vel"])

        binned_player_y = int(clamped_state["player_y"] / (self.pixels_y / (self.num_intervals - 1)))
        binned_x_dist_next = int(clamped_state["next_pipe_dist_to_player"] / (self.pixels_x / (self.num_intervals  - 1)))
        binned_y_top_next = int(clamped_state["next_pipe_top_y"] / (self.pixels_y / (self.num_intervals - 1)))

        binned_y_bottom_next = int(clamped_state["next_pipe_bottom_y"] / (self.pixels_y / (self.num_intervals - 1)))
        binned_x_dist_next_next = int(clamped_state["next_next_pipe_dist_to_player"] / (self.pixels_x / (self.num_intervals - 1)))

        binned_y_top_next_next = int(clamped_state["next_next_pipe_top_y"] / (self.pixels_y / (self.num_intervals - 1)))
        binned_y_bottom_next_next = int(clamped_state["next_next_pipe_bottom_y"] / (self.pixels_y / (self.num_intervals - 1)))
        binned_y_bottom_next = int(clamped_state["next_pipe_bottom_y"] / (self.pixels_y / (self.num_intervals - 1)))
        return GameState(player_vel=binned_player_vel, player_y=binned_player_y, next_pipe_dist_to_player=binned_x_dist_next,\
                         next_pipe_top_y=binned_y_top_next, next_pipe_bottom_y=binned_y_bottom_next,\
                         next_next_pipe_dist_to_player=binned_x_dist_next_next, next_next_pipe_top_y=binned_y_top_next_next,\
                            next_next_pipe_bottom_y=binned_y_bottom_next_next)

    def get_q_table_key(self, state: GameState):
        bins = self.get_binned_state(state)
        return (bins["player_y"], bins["next_pipe_top_y"], bins["next_pipe_dist_to_player"], bins["player_vel"])

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        key_s1 = self.get_q_table_key(s1)
        key_s2 = self.get_q_table_key(s2)
        self.q_table[key_s1][a] = self.q_table[key_s1][a] + self.LEARNING_RATE_ALPHA * (r + self.DISCOUNT_RATE_GAMMA *\
            max(self.q_table[key_s2]) - self.q_table[key_s1][a])
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # print("state: %s" % state)
        if np.random.uniform(0, 1) > self.E_GREEDY_EPSILON:
            actions = self.q_table[self.get_q_table_key(state)]
            # print("Greedy: ", actions)
            return 0 if actions[0] >= actions[1] else 1
        else:
            return random.randint(0, 1)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
                """
        """
        player y position.
        players velocity.
        next pipe distance to player
        next pipe top y position
        next pipe bottom y position
        next next pipe distance to player
        next next pipe top y position
        next next pipe bottom y position
        """
        print(f"state: {state}")
        actions = self.q_table[self.get_q_table_key(state)]
        return 0 if actions[0] >= actions[1] else 1


# if __name__ == "__main__":
# 
    # agent = FlappyAgentV1()
    # input("Press any button to start training")
    # train(40000, agent)
    # input("Press any button to resume to playing (training mode off)")
    # run_game(10, agent)

