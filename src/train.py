import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from maddpg_agent import MultiAgentDDPG
from config_loader import Configurations
from utils import create_simple_training_plot, create_complex_training_plot
import os

from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import os

class MADDPGTrainer:
    def __init__(self, config: Configurations, file_path: str = 'Tennis_Linux/Tennis.x86_64'):
        self.config = config
        self.file_path = file_path
        self.noise_choice = config.noise_config.GENERAL.PROB_NOISE_OR_OU
        self.noise_decay_choice = config.noise_config.GENERAL.ACT_NOISE_DECAY
        self.n_episodes = config.hyperparameters.N_EPISODES
        self.max_t = config.hyperparameters.MAX_T
        self.trialname = config.hyperparameters.TRIAL_NAME

        self.initialize_unity()
        self.print_environment_information()

        # Instantiate MultiAgentDDPG
        self.multiagent = MultiAgentDDPG(
            num_agents=self.num_agents,
            state_size=self.state_size,
            action_size=self.action_size,
            random_seed=2
        )

    def initialize_unity(self):
        self.env = UnityEnvironment(file_name=self.file_path)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_size = brain.vector_action_space_size
        self.states = env_info.vector_observations
        self.state_size = self.states.shape[1]

    def maddpg_train(self, print_every=100, target_score=0.5):
        scores_deque = deque(maxlen=print_every)
        scores = []
        all_scores_deque = deque(maxlen=print_every)
        for i_episode in range(1, self.n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            self.multiagent.reset()
            episode_scores = np.zeros(self.num_agents)

            for t in range(self.max_t):
                actions = self.multiagent.act(states, add_noise=True)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.multiagent.step(states, actions, rewards, next_states, dones)
                states = next_states
                episode_scores += rewards

                if np.any(dones):
                    break

            # Noise Decay
            if self.noise_decay_choice:
                if self.noise_choice == 'prob':
                    for agent in self.multiagent.agents:
                        agent.noise_update()
                elif self.noise_choice == 'ou':
                    for agent in self.multiagent.agents:
                        agent.noise_update()

            max_score = np.max(episode_scores)
            scores.append(max_score)
            scores_deque.append(max_score)
            all_scores_deque.append(episode_scores)

            print(f'\rEpisode {i_episode}\tScore: {max_score:.2f}\tAgent 0: {episode_scores[0]:.2f}\tAgent 1: {episode_scores[1]:.2f}', end="")

            if i_episode % print_every == 0:
                avg_score = np.mean(scores_deque)
                    # New: store individual agent scores from last N episodes
                agent0_avg = np.mean([s[0] for s in all_scores_deque])
                agent1_avg = np.mean([s[1] for s in all_scores_deque])
                
                print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tAgent 0: {agent0_avg:.2f}\tAgent 1: {agent1_avg:.2f}')
                self.save_models()

            if np.mean(scores_deque) >= target_score:
                print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
                self.save_models()
                break

        return scores

    def save_models(self):
        """Save all agent models."""
        os.makedirs(f'models/{self.trialname}', exist_ok=True)
        for i, agent in enumerate(self.multiagent.agents):
            torch.save(agent.actor_local.state_dict(), f'models/{self.trialname}/agent_{i}_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'models/{self.trialname}/agent_{i}_critic.pth')

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


if __name__ == "__main__":
    config = Configurations()
    trainer = MADDPGTrainer(config=config, file_path="Tennis_Linux/Tennis.x86_64")
    scores = trainer.maddpg_train(print_every=100, target_score=1)
    trainer.generate_result_plots(scores)
    trainer.close()
