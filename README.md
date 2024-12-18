# Reinforcement Learning Project: Q-Learning and Deep Q-Learning

This project implements Q-Learning and Deep Q-Learning algorithms to train an agent in a 2D simulation environment. The environment presents dynamic obstacles, and the agent must navigate through them while optimizing performance.

## Project Overview

- **Goal**: Train an agent using reinforcement learning techniques to make optimal decisions in a dynamic environment.  
- **Algorithms**:  
  - **Q-Learning**: Uses a tabular approach to learn optimal actions for discrete states.  
  - **Deep Q-Learning**: Utilizes a neural network to approximate the Q-values for continuous states.  

## Features

- **State Representation**: Includes environment parameters like agent position, obstacle distances, and velocity.  
- **Action Space**: Limited to two actions - "Move" or "Stay."  
- **Learning Algorithms**:  
  - Q-Learning with discretized state spaces.  
  - Deep Q-Learning using neural networks for approximation.

## Requirements

The project uses Python and the following libraries:

```bash
seaborn
sklearn
numpy
matplotlib
pygame
```

Install the requirements using the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

### Setup Instructions

1. Create a Virtual Environment:

```bash
python -m venv .venv
```

2. Activate the Virtual Environment:

```bash
source .venv/bin/activate  # on macOS/Linux:
.venv\Scripts\activate  # on Windows
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Project:

```bash
# Run the Q-Learning and Deep Q-Learning scripts
python Q_Learning.py
python DeepQ_Learning.py
```

## Project Structure

- Q_Learning.py: Implementation of Q-Learning algorithm.
- DeepQ_Learning.py: Implementation of Deep Q-Learning algorithm using a neural network.
- requirements.txt: Lists required Python libraries.
- .gitignore: Excludes unnecessary files (e.g., virtual environments).

## Results

The project includes:

- Learning curve analysis comparing Q-Learning and Deep Q-Learning.
- Hyperparameter tuning for optimal agent performance.
- Visualization of agent performance over training episodes.