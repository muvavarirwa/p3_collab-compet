## COLLAB-COMPETE

### Overview of Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

1. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

2. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Environment Setup

#### 0. Installing Dependencies

Next, we will start the environment! Before running the code cell below, change the file_name parameter to match the location of the Unity environment that you downloaded.

    Mac: "path/to/Tennis.app"
    Windows (x86): "path/to/Tennis_Windows_x86/Tennis.exe"
    Windows (x86_64): "path/to/Tennis_Windows_x86_64/Tennis.exe"
    Linux (x86): "path/to/Tennis_Linux/Tennis.x86"
    Linux (x86_64): "path/to/Tennis_Linux/Tennis.x86_64"
    Linux (x86, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86"
    Linux (x86_64, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86_64"

For instance, if you are using a Linux machine, then you downloaded Tennis.x86_64. If this file is in the same folder as the notebook, then the line below should appear as follows:

env = UnityEnvironment(file_name="/home/ubuntu/udacity/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64", worker_id=162, no_graphics=True)


In this notebook, the **Directory** structure below is used:


#### 1. Directory Structure

Project assume the following home directory:   /home/ubuntu

Project files are installed in the directory:  /home/ubuntu/udaciy/deep-reinforcement-learning/p3_collab-compet

Unity Agent ("Tennis") is in the directory:   /home/ubuntu/udaciy/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux


#### 2. Testing the environment

##### Get the Default Brain

brain_name = env.brain_names[0]

brain = env.brains[brain_name]


#### 3. Examine the state and action spaces

##### Reset the environment

env_info = env.reset(train_mode=True)[brain_name]

##### Get the number of agents

num_agents = len(env_info.agents)

print('Number of agents:', num_agents)

##### Get the size of each action

action_size = brain.vector_action_space_size

print('Size of each action:', action_size)

##### Examine the state space 

states = env_info.vector_observations

state_size = states.shape[1]

print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

print('The state for the first agent looks like:', states[0])


#### 4. Running the Training sessions

1. Launch the "Tennis-64x512.v_0.2.ipynb" notebook

2. From menu, select Kernel -> Reset & Run All




