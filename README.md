[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: results/images/Tennis_Final_Agent_03.gif "Preview Trained Agent"


# Collaboration and Competition - Let's learn to continously play Tennis with two individual Agents

This project implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with probabilistic noise to train two individual agents to play as long as possible tennis togehter. The agent interacts with the Unity ML-Agents Tennis environment. 

In the upcoming section [Introduction](#introduction), an introduction into the project and the environment follows. Further, one can get familiar with the necessary dependencies to run this project on your own local device in the [Getting Started](#getting-started) section. The previously mentioned sections are mainly authored by Udacity and were copied into the project. Some minor adjustments were made to these sections. In the section [Preview of the trained Agent](#preview-of-the-trained-agent) one can already get a peak in the trained agents in action. After sucessfully downloading and installing the prerequisites, in the section [Instructions](#instructions) one can get to know how to set the training configurations, run the training for the MADDPG, and how to watch the trained agent. The Training can be executed via terminal. Finally, one can get a glimpse at the structure of the repository in [Structure of the Repository](#structure-of-the-repository). A detailed report of this implementation can be found in the report markdown `Report.md`.

## Table of Contents

<!-- TOC -->

- [Collaboration and Competition - Let's learn to continously play Tennis with two individual Agents](#collaboration-and-competition---lets-learn-to-continously-play-tennis-with-two-individual-agents)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Preview of the Trained Agent](#preview-of-the-trained-agent)
  - [Getting Started](#getting-started)
    - [Python Dependencies](#python-dependencies)
    - [Getting the Tennis Environment](#getting-the-tennis-environment)
  - [Instructions](#instructions)
    - [Setting the Training Configurations](#setting-the-training-configurations)
    - [Running the Training](#running-the-training)
    - [Watching the trained Agent](#watching-the-trained-agent)
  - [Structure of the Repository](#structure-of-the-repository)

<!-- /TOC -->

## Introduction

For this project,  we work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Preview of the Trained Agent

![Trained Agent][image3]

## Getting Started

### Python Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6. In case you do not have conda installed, ensure to install anaconda on your system: https://www.anaconda.com/docs/getting-started/anaconda/install 

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/sifesch/tennis_collab_comp
cd tennis_collab_comp/python
pip install .
```

4. (Optional, in case new Python Notebooks are created) Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. (Optional, in case new Python Notebooks are created) Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

### Getting the Tennis Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `tennis_coolab_comp/` folder, and unzip (or decompress) the file. 

## Instructions

First, one has to set the configurations for the general settings, the hyperparameters and the noise. Then one can run the training with the defined configurations. Afterwards one can watch the trained agent. To watch the trained agent, the configurations need to be set to the same ones of the respective trained agent (see therefore `results/configurations/`).

### Setting the Training Configurations

1. Open the `config/train_config.yaml` file.
2. Define the configurations for the Training. View the following yaml snippet to get an understanding what possibilties there are 

```yaml
HYPERPARAMETERS:
  TRIAL_NAME: 'Test'            # Defines name logic for saving (checkpoint weights, training plot, configuration file)
  N_EPISODES: 6000              # Defines how many episodes the agent will be trained   
  MAX_T: 1000                   # Defines the maximum number of time steps the agent can take per episode
  BUFFER_SIZE: 1000000          # replay buffer size, defines the max number of experiences the buffer to hold
  BATCH_SIZE: 32                # minibatch size, defines the number of experiences sampled from the buffer to train the agent on each learning update
  GAMMA: 0.99                   # discount factor, defines how important future vs. immedieate rewards are. Close to 1 means agent careas a lot about future rewards, zero means the agent only cares about immediate rewards
  TAU: 0.001                    # for soft update of target parameters, defines how much to update the target networks each step, makes learning stable by slowly updating target networks
  LEARN_FREQ: 100               # defines after how many steps the agent should learn, until then the agent acts and collects experiences
  GRADIENT_UPDATES: 1           # Decision how many time the gradients are updated during each step
  REWARD_SCALING: False         # Defines if the reward should be scaled -> DISABLED
  SCALE_FACTOR_REWARD: 0.01     # Defines by how much the rewards should be scaled
  ACTOR_PARAMS:                 # Parameters for the Actor Network
    BATCH_NORMILIZATION: True   # Defines wheter to include Batch Normalization Layers in the Network for more stable learning
    LR_ACTOR: 0.0001            # learning rate of the actor 
    FC1_UNITS: 32               # Units of the first linear layer
    FC2_UNITS: 64               # Units of the second linear layer
  CRITIC_PARAMS:                # Parameters for the Critic Network
    BATCH_NORMILIZATION: True   # Defines wheter to include Batch Normalization Layers in the Network for more stable learning
    LR_CRITIC: 0.001            # learning rate of the critic
    FC1_UNITS: 64               # Units of the first linear layer
    FC2_UNITS: 128               # Units of the second linear layer
    WEIGHT_DECAY: 0.0           # L2 weight decay to regularize the critic to prevent overfitting. Makes critic more stable and general by preventing noisy Q-valuse from the replay buffer (usually )

NOISE:                          # Noise Settings
  GENERAL:                      # General Settings for decision of Noise Type and including noise decay
    PROB_NOISE_OR_OU: 'prob'    # choice of 'prob' or 'ou' noise.
    ACT_NOISE_DECAY: FALSE      # Decision to activate a decaying noise rate
    ADD_NOISE: False            # Not in use anymore -> DISABLED
    STOP_NOISE: 30000           # Step when during training the noise is turned off and exploring stops fully
  OUNoise_Config:               # Configurations for Ornstein-Uhlenbeck-Prozess
    MU: 0.0                     
    THETA: 0.15
    SIGMA: 0.15                 
  ProbNoise_Config:             # Configurations for Probabilistic Noise
    NOISE_INIT: 0.99            # Initialization of noise factor (If no decay this will be always the noise factor)
    NOISE_DECAY: 0.95           # Decaying rate for probabilistc noise
    NOISE_MIN: 0.01             # Min Value for Decaying Noise Rate
```

### Running the Training

1. Navigate to directory `tennis_collab_comp` in your command terminal, ensure the dependencies are installed properly and the respective conda environment is activated. Then run the training by running the following command:
 ```bash
 python3 src/train.py
 ```

### Watching the trained Agent

1. Open the `src/visualize_agent.py` file.
2. Define in section `if __name__ == '__main__':` the path of the weights of interests for the actor and critic network, indicating the individual trained agent you want to review. In addition, define the file name of the Tennis environment (depending on which environment you are using). Finally, you can decide on how many steps you want to watch the agent.

```python
if __name__ == "__main__":
    visualizer = TennisAgentsVisualizer(file_path = 'Tennis_Linux/Tennis.x86_64',
                                        path_agents='models/Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_03',
                 seed = 2)
    visualizer.run(max_t = 2000)
```

3. Ensure that the configurations in `config/train_config.yaml` are set like the ones of the agent you want to watch. To get the respective configurations of the trained agent one can check the used configurations in `results/configurations/`.
4. Navigate to the directory `tennis_collab_comp` in your command terminal, ensure the dependencies are installed properly and the respective conda environment is activated. Then run the following command to observe what the trained agent learned:
 ```bash
 python3 src/visualize_agent.py
 ```

## Structure of the Repository

```
├── config                      # Folder containing the configuration for hyperparameters and noise
│   └── train_config.yaml       # yaml file containg the Hyperparameter and noise configuration
├── models                      # Folder containing folders of trained actor and critic checkpoints of each indivudal agent.
├── python                      # Python setup (Needs to be put in this folder, see Getting Started Section) 
├── results                     # result folder containing the configurations for the run and training scores visualization.
│   ├── configurations          # folder containing the hyperparameter configurations of the trained agents
│   ├── images                  # folder containing a gif of the trained agent and the console of the training and test scores
│   └── training_scores         # folder containing the scores as numpy files and the 
├── src                         # Main files for the Actor Critic Network, the DDPG Agent, the Training
│   ├── actor_critic.py         # Containing the Actor and Critic Model Architecture
│   ├── config_loader.py        # script to load the configurations from the config/train_config.yaml file.
│   ├── ddpg_agent.py           # Containing the implementation of the DDPG Agent
│   ├── maddpg_agent.py         # script containg the logic to manage multiple DDPG agents 
│   ├── noise.py                # script containg the Ornstein Uhlenbeck Noise class
│   ├── replay_buffer.py        # script containg the replay buffer for the MADDPG
│   ├── train.py                # main training script
│   ├── utils.py                # Contains helper functions to create plots
│   └── visualize_agent.py      # Contains the logic to visualize a trained agent.
├── Tennis_Linux                # Tennis Environment (Name could vary depending on your OS, needs to be put into this folder, see Getting Started Section)
├── README.md                   # README you are currently reading
└── Report.md                   # Report about the learning algorithm, reward plots and future ideas for project improvements
```

