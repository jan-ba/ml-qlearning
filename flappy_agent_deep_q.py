from collections import defaultdict, deque
from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np
from typing import TypedDict
from sklearn.neural_network import MLPRegressor
from flappy_agent import FlappyAgent, GameState
from copy import deepcopy

# Lecture 17, slide 38 for deep q learning

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": -1.0, "tick": -0.01, "loss": -5.0, "win": 10.0}

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
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None,
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


class FlappyAgentV2(FlappyAgent):
    def __init__(self):
        super().__init__()
        self.pixels_y = 512
        self.pixels_x = 288
        self.velocity_max = 10
        self.velocity_min = -8
        self.frame_counter = 0
        
        # Hyperparameters
        self.DISCOUNT_RATE_GAMMA = 0.9
        self.INIT_LEARNING_RATE_ALPHA = 0.1
        self.E_GREEDY_EPSILON = 0.1
        self.REPLAY_BUFFER_SIZE = 1000
        self.BATCH_SIZE = 100
        self.MODEL_UPDATE_STEP = 100
    
        # Neural Network stuff
        self.model = MLPRegressor(hidden_layer_sizes=(100, 10), activation="logistic",\
                                  learning_rate_init=self.INIT_LEARNING_RATE_ALPHA, warm_start=True,\
                                  max_iter=1)
        self.target_model = MLPRegressor(hidden_layer_sizes=(100, 10), activation="logistic",\
                                         learning_rate_init=self.INIT_LEARNING_RATE_ALPHA, warm_start=True,\
                                         max_iter=1)

        # Fit network to dummies so as to get desired number for in- and output neurons
        dummy_input = np.zeros((1, 8))  # number of state attributes
        dummy_output = np.zeros((1, 2))  # for 0 (jump) and 1 (do nothing)
        self.model.fit(dummy_input, dummy_output)
        self.target_model.fit(dummy_input, dummy_output)

        # init replay buffer
        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)
    
    def get_normalised_state_values(self, state: GameState) -> tuple:

        # Clamp state values between min and max (because of some bug in the environment)
        # also normalises each value to [1, -1]

        vel = int(max(min(self.velocity_max, state["player_vel"]), self.velocity_min))

        player_y = int(max(min(self.pixels_y, state["player_y"]), 0))
        x_dist_next = int(max(min(self.pixels_x, state["next_pipe_dist_to_player"]), 0))
        y_top_next = int(max(min(self.pixels_y, state["next_pipe_top_y"]), 0))

        y_bottom_next = int(max(min(self.pixels_y, state["next_pipe_bottom_y"]), 0))
        x_dist_next_next = int(max(min(self.pixels_x, state["next_next_pipe_dist_to_player"]), 0))

        y_top_next_next = int(max(min(self.pixels_y, state["next_next_pipe_top_y"]), 0))
        y_bottom_next_next = int(max(min(self.pixels_y, state["next_next_pipe_bottom_y"]), 0))

        # Normalize to [-1, 1]
        vel = 2 * (vel - self.velocity_min) / (self.velocity_max - self.velocity_min) - 1
        player_y = 2 * player_y / self.pixels_y - 1
        x_dist_next = 2 * x_dist_next / self.pixels_x - 1
        y_top_next = 2 * y_top_next / self.pixels_y - 1
        y_bottom_next = 2 * y_bottom_next / self.pixels_y - 1
        x_dist_next_next = 2 * x_dist_next_next / self.pixels_x - 1
        y_top_next_next = 2 * y_top_next_next / self.pixels_y - 1
        y_bottom_next_next = 2 * y_bottom_next_next / self.pixels_y - 1
        # return GameState(player_vel=vel, player_y=player_y, next_pipe_dist_to_player=x_dist_next, next_pipe_top_y=y_top_next,\
        #                  next_pipe_bottom_y=y_bottom_next, next_next_pipe_dist_to_player=x_dist_next_next,\
        #                     next_next_pipe_top_y=y_top_next_next, next_next_pipe_bottom_y=y_bottom_next_next)
        return (vel, player_y, x_dist_next, y_top_next, y_bottom_next, x_dist_next_next, y_top_next_next, y_bottom_next_next)

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # Normalize state 1 and state 2
        s1_norm = self.get_normalised_state_values(s1)
        s2_norm = self.get_normalised_state_values(s2)
        # Append the state, action, reward, state 2 and end in bottom
        self.replay_buffer.append((s1_norm, a, r, s2_norm, end))

        print("len replay_buffer: ", len(self.replay_buffer))
        if len(self.replay_buffer) >= self.BATCH_SIZE:
            # Draw random sample from the replay buffer
            batch = random.sample(self.replay_buffer, self.BATCH_SIZE)
            s1_batch, a_batch, r_batch, s2_batch, end_batch = zip(*batch)


            # Convert to Numpy for efficiency
            s1_batch = np.array(s1_batch)
            a_batch = np.array(a_batch)
            r_batch = np.array(r_batch)#.reshape(-1, 1)  # gets an array of arrays, supposedly required for fitting function
            s2_batch = np.array(s2_batch)
            end_batch = np.array(end_batch)#.reshape(-1, 1)  
            
            q_s1 = self.model.predict(s1_batch)
            q_s2 = self.target_model.predict(s2_batch)
            # print("q_s1 (batch): ", q_s1)


            # target_q_values = q_s1.copy()
            # Taking the maximum expected benefit of future state (Q learning), also ensuring the right format for the for loop later
            max_q_values_s2 = np.max(q_s2, axis=1)#.reshape(-1,1)

            for i in range(self.BATCH_SIZE):
                if end_batch[i]:  # if state is terminal, use reward only
                    q_s1[i, a_batch[i]] = r_batch[i]
                else:
                    q_s1[i, a_batch[i]] = r_batch[i] + self.DISCOUNT_RATE_GAMMA * max_q_values_s2[i]
                    # print("UPDATE TARGET Q VALUES: ", target_q_values[i, a_batch[i]])
            # Partially fitting the model because its more memory efficient
            self.model.partial_fit(s1_batch, q_s1)
            print("JUST PARTIALLY FIT THE MODEL")
            print("Min Q-value in q_s1:", np.min(q_s1), "Max Q-value in q_s1:", np.max(q_s1))
            print("Min Q-value in q_s2:", np.min(q_s2), "Max Q-value in q_s2:", np.max(q_s2))

       
        print("self.frame_counter: ", self.frame_counter)
        # count the frames, so as to move the model into the target model after x steps
        if self.frame_counter >= self.MODEL_UPDATE_STEP:
            print(f"UPDATE TARGET MODEL, frame counter is {self.frame_counter}")
            self.update_target_model()
            self.frame_counter = 0
        else: self.frame_counter += 1

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # Eploiting
        if np.random.uniform(0, 1) > self.E_GREEDY_EPSILON:
            # Normalizing the state
            state_normalized = np.array([self.get_normalised_state_values(state)])
            # Finding all of the q values
            q_values = self.model.predict(state_normalized)
            # Returning the index of the best action e.g. Q-values=[0.3, 0.8] -> not jump 
            return np.argmax(q_values)
        # Exploring
        else:
            return random.randint(0, 1)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        
            Selects the best action based on the trained model during evaluation.
        """
        # Normalizing the state
        state_normalized = np.array([self.get_normalised_state_values(state)])
        # Finding all of the q values
        q_values = self.model.predict(state_normalized)
        # Returning the index of the best action e.g. Q-values=[0.3, 0.8] -> not jump 
        return np.argmax(q_values)

    def update_target_model(self):
        # self.target_model.coefs_ = deepcopy(self.model.coefs_)
        # self.target_model.intercepts_ = deepcopy(self.model.intercepts_)
        print("std. coeffs: ", self.model.coefs_[0][0][:5])
        print("target coeffs: ", self.target_model.coefs_[0][0][:5])
        self.target_model.coefs_ = [np.copy(coef) for coef in self.model.coefs_]
        self.target_model.intercepts_ = [np.copy(intercept) for intercept in self.model.intercepts_]
        print("std. coeffs UPDTATE: ", self.model.coefs_[0][0][:5])
        print("target coeffs UPDATE: ", self.target_model.coefs_[0][0][:5])

if __name__ == "__main__":
    agent = FlappyAgentV2()
    train(1000, agent)
    input("Press any button to resume to playing (training mode off)")
    run_game(10, agent)

