import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from maddpg_agent import MultiAgentDDPG
from config_loader import Configurations
from utils import create_simple_training_plot, create_complex_training_plot

class MADDPGTrainer:
    def __init__(self, config: Configurations, file_path: str = 'Tennis_Linux/Tennis.x86_64'):
        self.file_path = file_path
        self.noise_choice = config.noise_config.GENERAL.PROB_NOISE_OR_OU
        self.noise_decay_choice = config.noise_config.GENERAL.ACT_NOISE_DECAY
        self.n_episodes = config.hyperparameters.N_EPISODES
        self.max_t = config.hyperparameters.MAX_T
        self.trialname = config.hyperparameters.TRIAL_NAME
        self.config = config

        self.initialize_unity()
        self.print_environment_information()
        self.multiagent = MultiAgentDDPG(state_size=self.state_size, 
                                         action_size=self.action_size,
                                         num_agents=self.num_agents, 
                                         random_seed=2)

    def initialize_unity(self):
        self.env = UnityEnvironment(file_name=self.file_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        self.states = env_info.vector_observations
        self.state_size = self.states.shape[1]

    def maddpg_train(self, print_every=100, target_score=0.5):
        scores_deque = deque(maxlen=100)
        scores = []

        for i_episode in range(1, self.n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            self.multiagent.reset()
            episode_scores = np.zeros(self.num_agents)

            for t in range(self.max_t):
                actions = self.multiagent.act(states, noise=True)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.multiagent.step(states, actions, rewards, next_states, dones, t_step=t)
                states = next_states
                episode_scores += rewards

                if np.any(dones):
                    if self.noise_decay_choice and self.noise_choice == 'prob':
                        self.multiagent.update_noise()
                    elif self.noise_decay_choice and self.noise_choice == 'ou':
                        self.multiagent.decay_ou_noise()
                    break

            max_score = np.max(episode_scores)
            scores.append(max_score)
            scores_deque.append(max_score)

            print(f'\rEpisode {i_episode}\tScore: {max_score:.2f}', end="")

            if i_episode % print_every == 0:
                avg_score = np.mean(scores_deque)
                print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}')
                self.multiagent.save_models(trialname=self.trialname)

            if np.mean(scores_deque) >= target_score:
                print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
                self.multiagent.save_models(trialname=self.trialname)
                break

        return scores

    def print_environment_information(self):
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('Each agent observes a state of length:', self.state_size)
        print('Sample state:', self.states[0])

    def generate_result_plots(self, scores):
        create_simple_training_plot(scores, trialname=self.trialname)
        create_complex_training_plot(scores, trialname=self.trialname, window_size=100)
        self.config.save_config(output_path=f'results/configurations/trial_configurations_{self.trialname}.yaml')

    def close(self):
        self.env.close()

if __name__ == '__main__':
    config = Configurations(config_path='config/train_config.yaml')
    trainer = MADDPGTrainer(config=config)
    scores = trainer.maddpg_train()
    trainer.generate_result_plots(scores)
    trainer.close()



'''
env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

multiagent = MultiAgent(state_size=24, action_size=2, num_agents=2, random_seed=2)

# Instantiate the environment
for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    multiagent.reset()                                     # reset noise in agents
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)

    while True:
        actions = multiagent.act(states, add_noise=False)  # get actions from trained policy (no noise for evaluation)
        env_info = env.step(actions)[brain_name]           # send all actions to the environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # check if episode finished

        scores += rewards                                  # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if any agent is done
            break

    print('Score (max over agents) from episode {}: {:.2f}'.format(i, np.max(scores)))

'''