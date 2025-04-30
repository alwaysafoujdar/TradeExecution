# Deep Reinforcement Learning for Optimal Order Execution

## Overview

This project implements a deep reinforcement learning (DRL) agent for optimizing the execution of large-volume orders in a financial market. The agent learns to trade a given number of shares over a specified time window, aiming to maximize profit and minimize market impact.  The code utilizes TensorFlow/Keras for building the neural network that approximates the optimal trading strategy.

## Key Features

* **DRL Agent:** A Deep Q-Network (DQN) agent learns to execute trades.
* **Trading Environment:** A simulated market environment provides price data and tracks the agent's inventory and time.
* **TWAP Baseline:** The Time-Weighted Average Price (TWAP) strategy is implemented as a benchmark for comparison.
* **Reward Function:** The agent is rewarded based on its profit and loss (PNL), with a penalty for large trades.
* **Exploration Strategy:** The agent uses an epsilon-greedy strategy with a binomial distribution to balance exploration and exploitation.
* **Replay Memory:** Experience replay is used to stabilize training.
* **Hyperparameter Tuning:** Key parameters, such as the number of episodes, memory size, and network architecture, can be configured.
* **Training and Evaluation:** The code supports both training and evaluation modes.
* **Data Handling:** The code reads price data from a CSV file.
* **State Representation**: The state is represented by the time and remaining inventory.

## Dependencies

* Python 3
* numpy
* pandas
* yfinance  # For downloading financial data (if needed)
* tensorflow (>=2.0)
* Keras

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```
2.  **Install the required packages:**
    ```bash
    pip install numpy pandas tensorflow yfinance
    ```
    *It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.*

## Data Preparation

* **Price Data:** The code expects price data in a CSV file named `APPL10minTickData.csv` in the `./Data_Sets/` directory.  The CSV file should contain a 'close' column representing the closing price.
* **Data Splitting:** The data is split into training and testing sets (70% for training, 30% for testing).

## Usage

1.  **Prepare Data:** Ensure your price data is in the correct format and location.
2.  **Run the `main.py` script:**
    ```bash
    python main.py
    ```

## Configuration

The following hyperparameters can be configured within the `main.py` script:

* `EXECUTION_EPISODES`:  The number of training/evaluation episodes.
* `REPLAY_MEMORY_SIZE`:  The maximum size of the experience replay memory.
* `INITIAL_SHARES`:  The initial number of shares to trade.
* `SCALED_INITIAL_SHARES`: Initial number of shares scaled.
* `EXECUTION_WINDOW`:  The time window for order execution.
* `PENALTY_FACTOR`:  Penalty for large trades in the reward function.
* `IS_TRAINING`:  Flag to switch between training (True) and evaluation (False) modes.
* `USE_BOUNDARIES`: Flag to use boundary conditions.
* `LOAD_PRETRAINED`: Flag to load a pre-trained model.
* `USE_PNL_REWARD`: Flag to use PNL in the reward function.
* `EXPLORATION_STRATEGY`:  Exploration strategy ('Binomial' or 'Uniform').
* `NETWORK_ARCHITECTURE`:  Nework architecture.
* `OPTIMIZATION_ALGORITHM`: Optimization algorithm.
* `FILE_PREFIX`:  Prefix for saved files.
* `MINIBATCH_SIZE`:  Size of the training batch.
* `STATE_DIMENSION`: Dimension of the state space.

## Code Description

### Imports

The code imports the following libraries:

* `random`:  For random number generation (used for exploration).
* `numpy`:  For numerical computations.
* `pandas`:  For data manipulation (reading CSV data).
* `yfinance`: For downloading financial data.
* `collections.deque`:  For implementing the replay memory.
* `tensorflow.keras`:  For building and training the neural network.

### Functions

* `scale_data(data_series)`:  Scales a numerical series to the range \[0, 1].

### Classes

* `TradingState`:  Represents the current state of the trading environment (time and remaining inventory).
* `TWAPStrategy`:  Implements the TWAP trading strategy as a baseline.
* `TradingEnvironment`:  Simulates the market environment and manages the trading process.
* `DQNAgent`:  Implements the DQN agent with methods for selecting actions, storing experiences, and training the Q-network.

### Main Function

* `main()`:  The main function initializes the environment, agent, and TWAP strategy.  It then runs the training or evaluation loop, collects data, and saves the results.

## Training

* The agent interacts with the trading environment over a series of episodes.
* During each episode, the agent observes the current state, selects an action (number of shares to trade), receives a reward, and transitions to the next state.
* The agent's experiences are stored in the replay memory.
* During training, the agent samples a minibatch of experiences from the replay memory and uses it to update the Q-network.
* The exploration rate decays over time, gradually shifting the agent from exploration to exploitation.

## Evaluation

* In evaluation mode (`IS_TRAINING = False`), the agent loads a pre-trained model and uses it to execute trades in the market environment.
* The agent's performance is compared to the TWAP baseline.
* The script prints the final PNL comparison and saves the results to CSV files.

## Output

The script saves the following files:

* `Memory_{EXECUTION_EPISODES}Train={IS_TRAINING}.csv`:  Contains the agent's experience replay memory.
* `P&L_{EXECUTION_EPISODES}Train={IS_TRAINING}.csv`:  Contains the PNL of the agent and the TWAP strategy.
* `avg_rewards_{EXECUTION_EPISODES}Train={IS_TRAINING}NN={NETWORK_ARCHITECTURE}REW={USE_PNL_REWARD}.csv`: Contains the average rewards per episode.
* `{FILE_PREFIX}.weights.h5`:  The weights of the trained neural network.

## Performance

The performance of the DRL agent can be evaluated by comparing its PNL to the TWAP baseline.  The `main.py` script calculates and prints the percentage difference in PNL.  The saved CSV files can be used for further analysis and visualization.

## Future Improvements

* Implement more sophisticated reward functions.
* Explore different neural network architectures.
* Incorporate more market features into the state representation.
* Use more advanced DRL algorithms, such as PPO or A3C.
* Add visualization of the training process and results.
* Implement portfolio optimization.

##  Disclaimer

This code is for educational and research purposes only.  It is not intended to be used for real-world financial trading.  Trading in financial markets involves significant risk of loss.  The user is solely responsible for any decisions made based on the use of this code.

## References
We took a lot inspiration from:
* Ning, B., Ho, F., Ling, T. and Jaimungal, S. (2018). Double Deep Q-Learning for Optimal Execution
* Reinforcement Learning For Optimal Trade Execution paper by 
