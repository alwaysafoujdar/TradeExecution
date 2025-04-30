#importing the use ful libraries
import random
import numpy as np
import pandas as pd
import yfinance as yf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



def scale_data(data_series):
    """
    Scales a numerical series to the range [0, 1].

    Args:
        data_series (array-like): The input numerical series.

    Returns:
        array-like: The scaled series.
    """
    min_val = np.min(data_series)
    max_val = np.max(data_series)
    return (data_series - min_val) / (max_val - min_val)


#here are some of the hyperparatemters I used in my code below
EXECUTION_EPISODES = 1000  
REPLAY_MEMORY_SIZE = 100000  # Maximum size of the experience replay buffer
INITIAL_SHARES = 21  # Initial number of shares to trade
SCALED_INITIAL_SHARES = 1 
EXECUTION_WINDOW = 11  # Number of time steps for execution
PENALTY_FACTOR = 0.01  # Penalty factor for large trades in reward function


IS_TRAINING = True  # Flag for training mode
USE_BOUNDARIES = False 
LOAD_PRETRAINED = False


USE_PNL_REWARD = True  
EXPLORATION_STRATEGY = "Binomial"  # Epsilon-greedy exploration strategy
NETWORK_ARCHITECTURE = "NN=3_NN"  
OPTIMIZATION_ALGORITHM = 'Adam' 
FILE_PREFIX = "{}ep_{}_{}A={}Actions={}TimeConstr={}Opt={}REW=".format(
    EXECUTION_EPISODES, NETWORK_ARCHITECTURE, EXPLORATION_STRATEGY, PENALTY_FACTOR, INITIAL_SHARES,
    EXECUTION_WINDOW, OPTIMIZATION_ALGORITHM, str(USE_PNL_REWARD)
)
MINIBATCH_SIZE = 128  # Size of the training batch
STATE_DIMENSION = 2  # Dimension of the state space (time, inventory)




#we will now prepare our csv data to be used by our model
raw_price_data = pd.read_csv('./Data_Sets/APPL10minTickData.csv', header=0)  
scaled_execution_times = scale_data(np.array(range(EXECUTION_WINDOW)))  
MAX_EXECUTION_TIME_SCALED = np.max(scaled_execution_times)
NUM_EXECUTION_STEPS = len(scaled_execution_times)
TIME_INCREMENT = 1 / (NUM_EXECUTION_STEPS - 1)


# 70% for training
TRAIN_SIZE = int(0.7 * len(raw_price_data)) 


if IS_TRAINING:
    trading_data = raw_price_data.iloc[:TRAIN_SIZE, ]
    current_time_index = 0
    final_time_index = len(trading_data)
else:
    trading_data = raw_price_data.iloc[TRAIN_SIZE:, ]
    current_time_index = TRAIN_SIZE
    final_time_index = TRAIN_SIZE + len(trading_data)
    EXECUTION_EPISODES = 34

price_series = scale_data(trading_data['close'].to_numpy()) 




class TradingState:
    """
    Represents the state of the trading environment at a given time.
    """

    def __init__(self, current_time, remaining_inventory):
        """
        Initializes the trading state.

        Args:
            current_time (float): The current time step within the episode.
            remaining_inventory (float): The number of shares remaining to be executed.
        """
        self.time = current_time  # Time period in an episode
        self.inventory = remaining_inventory  # Number of shares yet to be executed

    def to_list(self):
        """
        Returns the state as a list.

        Returns:
            list: A list representing the state [time, inventory].
        """
        return [self.time, self.inventory]
    


class TWAPStrategy:
    """
    Implements the Time-Weighted Average Price (TWAP) strategy.
    """

    def __init__(self, initial_inventory, num_execution_steps):
        """
        Initializes the TWAP strategy.

        Args:
            initial_inventory (float): The initial number of shares.
            num_execution_steps (int): The number of execution steps.
        """
        self.initial_inventory = initial_inventory
        self.num_execution_steps = num_execution_steps

    def execute_action(self):
        """
        Calculates the action (number of shares to trade) at each time step.

        Returns:
            float: The number of shares to trade in this step.
        """
        action = self.initial_inventory / self.num_execution_steps
        return action
    

class TradingEnvironment:
    """
    Simulates the trading environment where the agent interacts.
    """

    def __init__(self, initial_state, price_data, execution_window, num_execution_steps, start_time_index, end_time_index):
        """
        Initializes the trading environment.

        Args:
            initial_state (TradingState): The initial state of the environment.
            price_data (numpy.ndarray): Array of price data.
            execution_window (float): The total execution time window.
            num_execution_steps (int): Number of discrete time points.
            start_time_index (int): Start index in the full price data.
            end_time_index (int): End index in the full price data.
        """
        self.state = initial_state
        self.prices = price_data
        self.execution_window = execution_window
        self.num_execution_steps = num_execution_steps
        self.current_time_index = current_time_index
        self.start_time_index = start_time_index  # Time period along the whole data set
        self.end_time_index = end_time_index  # Final time of our dataset

    def reset(self):
        """
        Resets the environment to the initial state for a new episode.

        Returns:
            TradingState: The initial state.
        """
        self.state.inventory = SCALED_INITIAL_SHARES
        self.state.time = 0.0
        return self.state

    def step(self, action):
        """
        Executes an action in the environment.

        Args:
            action (float): The number of shares to trade.

        Returns:
            tuple: (next_state, reward, done)
                next_state (TradingState): The next state after taking the action.
                reward (float): The reward received for taking the action.
                done (bool): Whether the episode is finished.
        """
        reward = self.calculate_reward(self.state.inventory, action)
        self.state.time = round(self.state.time + TIME_INCREMENT, 2)
        self.current_time_index = (self.current_time_index + 1) if (self.current_time_index + 1) % self.end_time_index != 0 else self.start_time_index
        self.state.inventory = round(self.state.inventory - action, 2)
        return self.state, reward, self.is_episode_finished()

    def is_episode_finished(self):
        """
        Checks if the episode has finished.

        Returns:
            bool: True if the episode is finished, False otherwise.
        """
        return self.state.time == self.execution_window

    def get_current_price(self):
        """
        Gets the current price.

        Returns:
            float: The current price.
        """
        return self.prices[self.current_time_index - self.start_time_index]

    def calculate_reward(self, remaining_inventory, action):
        """
        Calculates the reward for a given action.

        Args:
            remaining_inventory (float): The remaining inventory.
            action (float): The action taken (number of shares traded).

        Returns:
            float: The reward.
        """
        if USE_PNL_REWARD:
            return action * self.get_current_price() - 2.5 * (action ** 2)
        return remaining_inventory * (self.get_current_price() - self.get_current_price()) - PENALTY_FACTOR * (action ** 2)

    def calculate_pnl(self, action):
        """
        Calculates the Profit and Loss for a given action.

        Args:
            action (float): The number of shares traded.

        Returns:
            float: The Profit and Loss.
        """
        return action * self.get_current_price() - PENALTY_FACTOR * (action ** 2)


class DQNAgent:
    """
    DQN Agent for learning the optimal trading strategy.
    """

    def __init__(self, state_dimension, action_space_size, is_training):
        """
        Initializes the DQN agent.

        Args:
            state_dimension (int): Dimension of the state space.
            action_space_size (int): Number of possible actions.
            is_training (bool): Whether the agent is being trained.
        """
        self.state_dimension = state_dimension
        self.action_space_size = action_space_size
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.discount_factor = 0.99  # gamma)
        self.exploration_rate = 1.0 if is_training else 0.0  # Exploration rate (epsilon)
        self.min_exploration_rate = 0.01 if is_training else 0.0  
        self.exploration_decay_rate = 0.995  
        self.learning_rate = 0.001
        self.model = self._build_q_network()

    def _build_q_network(self):
        """
        Builds the Q-network (neural network).

        Returns:
            tensorflow.keras.models.Sequential: The Q-network model.
        """
        q_network = Sequential()
        q_network.add(Dense(20, input_dim=self.state_dimension, activation='relu'))
        q_network.add(Dense(20, activation='relu'))
        q_network.add(Dense(20, activation='relu'))
        q_network.add(Dense(self.action_space_size, activation='linear'))
        q_network.compile(loss='mse', optimizer=OPTIMIZATION_ALGORITHM)
        return q_network

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state (numpy.ndarray): The current state.
            action (float): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, current_time, execution_window):
        """
        Selects an action based on the current state and exploration rate.

        Args:
            state (numpy.ndarray): The current state.
            current_time (float): The current time step.
            execution_window (float): The total execution time window.

        Returns:
            float: The action to take (number of shares to trade).
        """
        remaining_inventory = state[0][1]

        if current_time == (execution_window - TIME_INCREMENT):
            action = remaining_inventory
            print("last action is " + str(action))
        elif remaining_inventory == 0:
            action = 0  # To be time consistent with TWAP
        elif np.random.rand() <= self.exploration_rate:
            if EXPLORATION_STRATEGY == 'Binomial':
                n = remaining_inventory * INITIAL_SHARES
                p = TIME_INCREMENT / (execution_window - current_time)
                action = np.random.binomial(n, p)
                action = np.linspace(0, 1, INITIAL_SHARES)[action]  # Scale back the action
                print("E-greedy action " + str(action))
            elif EXPLORATION_STRATEGY == 'Uniform':
                action = random.randrange(self.action_space_size)
                action = np.linspace(0, 1, INITIAL_SHARES)[action]
                print("E-greedy action " + str(action))
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0]) / (INITIAL_SHARES - 1)
            print("Optimal action " + str(action))

        if action > remaining_inventory:
            action = remaining_inventory
            print("action>intentory action is " + str(action))
        return round(action, 2)

    def select_boundary_action(self, state, current_time, execution_window, index):
        """
        Selects action based on boundary conditions.

        Args:
            state: Current state
            current_time: Current time
            execution_window: Total execution window
            index: boundary condition index

        Returns:
            float: Action
        """
        if index == 0:
            if current_time == (execution_window - TIME_INCREMENT):
                action = SCALED_INITIAL_SHARES
            else:
                action = 0
        elif index == 1:
            if current_time == 0:
                action = SCALED_INITIAL_SHARES
            else:
                action = 0

        return round(action, 2)

    def replay(self, batch_size):
        """
        Trains the Q-network using a minibatch of experiences from the replay memory.

        Args:
            batch_size (int): The size of the minibatch.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_q_values = self.model.predict(state)
            action_range = np.linspace(0, 1, INITIAL_SHARES)
            action_index = [i for i in range(len(action_range)) if round(action_range[i].item(), 2) == action][0]
            target_q_values[0][action_index] = target
            self.model.fit(state, target_q_values, epochs=1, verbose=0)
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay_rate

    def load_model_weights(self, filename):
        """
        Loads the weights of a pre-trained model.

        Args:
            filename (str): The name of the file containing the model weights.
        """
        self.model.load_weights(filename)

    def save_model_data(self, filename, pnl_vs_twap, agent_pnl, twap_pnl):
        """
        Saves the trained model's weights and the performance metrics to files.

        Args:
            filename (str): The base filename for saving.
             pnl_vs_twap (np.array): Array containing P&L vs TWAP values.
            agent_pnl (np.array): Array containing agent P&L values
            twap_pnl (np.array): Array containing TWAP P&L values.
        """
        self.model.save_weights(filename)
        state_time_series = np.array([item[0][0][0] for item in list(self.memory)])
        state_inventory_series = np.array([item[0][0][1] for item in list(self.memory)])
        action_series = np.array([item[1] for item in list(self.memory)])
        reward_series = np.array([item[2] for item in list(self.memory)])
        next_state_time_series = np.array([item[3][0][0] for item in list(self.memory)])
        next_state_inventory_series = np.array([item[3][0][1] for item in list(self.memory)])
        done_series = np.array([item[4] for item in list(self.memory)])

        memory_data = pd.DataFrame({
            'state_time': state_time_series,
            'state_inventory': state_inventory_series,
            'action': action_series,
            'reward': reward_series,
            'next_state - Inventory': next_state_time_series,
            'next_state - Time': next_state_inventory_series,
            'done': done_series
        })
        pnl_data = pd.DataFrame({'PandL_vs_TWAP': pnl_vs_twap, 'PandL_agent': agent_pnl, 'PandL_TWAP': twap_pnl})
        print(memory_data)
        memory_data.to_csv('Memory_{}Train={}.csv'.format(EXECUTION_EPISODES, IS_TRAINING))
        pnl_data.to_csv('P&L_{}Train={}.csv'.format(EXECUTION_EPISODES, IS_TRAINING))

def main():
    """
    Main function to run the trading simulation.
    """
    initial_state = TradingState(current_time=0.0, remaining_inventory=SCALED_INITIAL_SHARES)
    trading_env = TradingEnvironment(
        initial_state=initial_state,
        price_data=price_series,
        execution_window=MAX_EXECUTION_TIME_SCALED,
        num_execution_steps=NUM_EXECUTION_STEPS,
        start_time_index=current_time_index,
        end_time_index=final_time_index
    )

    agent = DQNAgent(state_dimension=STATE_DIMENSION, action_space_size=INITIAL_SHARES, is_training=IS_TRAINING)

    if not IS_TRAINING:
        agent.load_model_weights("{}.weights.h5".format(FILE_PREFIX))
    if IS_TRAINING and LOAD_PRETRAINED:
        agent.load_model_weights("100ep_NN=3_NN_UniformA=0.01Actions=21TimeConstr=11Opt=RMSpropREW=.weights.h5")

    twap_strategy = TWAPStrategy(initial_inventory=SCALED_INITIAL_SHARES, num_execution_steps=NUM_EXECUTION_STEPS)

    agent_pnl_history = np.array([])
    twap_pnl_history = np.array([])
    pnl_comparison_history = np.array([])
    episode_rewards = np.array([])
    average_episode_rewards = np.array([])

    for episode in range(EXECUTION_EPISODES):
        state = trading_env.reset()
        state = np.reshape(state.to_list(), [1, STATE_DIMENSION])
        print("REAL TIME is: " + str(trading_env.start_time_index))
        print("start episode:" + str(episode))
        for current_time in scaled_execution_times:
            print("inventory is: " + str(trading_env.state.inventory))
            print("time is: " + str(trading_env.state.time))
            if USE_BOUNDARIES:
                if current_time == 0:
                    boundary_index = np.random.binomial(1, 1 / 2)
                action = agent.select_boundary_action(state, current_time, trading_env.execution_window, boundary_index)
            else:
                action = agent.select_action(state, current_time, trading_env.execution_window)
            next_state, reward, done = trading_env.step(action)
            next_state = np.reshape(next_state.to_list(), [1, STATE_DIMENSION])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent_pnl_history = np.append(agent_pnl_history, trading_env.calculate_pnl(action))
            twap_pnl_history = np.append(twap_pnl_history, trading_env.calculate_pnl(twap_strategy.execute_action()))
            pnl_comparison_history = np.append(pnl_comparison_history,
                                             ((agent_pnl_history[trading_env.start_time_index - 1 - current_time_index] -
                                               twap_pnl_history[trading_env.start_time_index - 1 - current_time_index]) /
                                              twap_pnl_history[trading_env.start_time_index - 1 - current_time_index]) * 100)
            episode_rewards = np.append(episode_rewards, reward)

            if done:
                print("DONE")
                print("episode: {}/{}, P&L_vs_TWAP: {}%, time: {}, e: {:.2}".format(
                    episode, EXECUTION_EPISODES, pnl_comparison_history[trading_env.start_time_index - 1 - current_time_index], current_time,
                    agent.exploration_rate))
                average_episode_rewards = np.append(average_episode_rewards, np.mean(episode_rewards))
                episode_rewards = np.array([])
                break
            if len(agent.memory) > MINIBATCH_SIZE and IS_TRAINING:
                agent.replay(MINIBATCH_SIZE)

        total_agent_pnl = np.sum(agent_pnl_history)
        total_twap_pnl = np.sum(twap_pnl_history)
        final_pnl_comparison = ((total_agent_pnl - total_twap_pnl) / total_twap_pnl) * 100
        print("PandL_vs_TWAP is {}%".format(final_pnl_comparison))
        avg_rewards_df = pd.DataFrame({'avg_rewards': average_episode_rewards})
        print("Avg_rewards are {}".format(average_episode_rewards))
        avg_rewards_df.to_csv('avg_rewards_{}Train={}NN={}REW=.csv'.format(EXECUTION_EPISODES, IS_TRAINING,
                                                                          NETWORK_ARCHITECTURE, str(USE_PNL_REWARD)))

        agent.save_model_data(FILE_PREFIX + ".weights.h5", pnl_comparison_history, agent_pnl_history, twap_pnl_history)

if __name__ == "__main__":
    main()