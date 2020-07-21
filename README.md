## CONTINOUS_NAVIGATION

### Overview of Environment

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.

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

For instance, if you are using a Mac, then you downloaded Tennis.app. If this file is in the same folder as the notebook, then the line below should appear as follows:

env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64", worker_id=162, no_graphics=True)


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
