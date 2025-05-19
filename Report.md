[//]: # (Image References)

[image1]: results/training_scores/training_scores_trial_Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_03.png "Reward Plot Final Agent"
[image2]: results/training_scores/training_scores_trial_Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_08.png "Reward Plot Final Agent Long Learning"
[image3]: results/images/Tennis_Final_Agent_03.gif "Trained Agent"
[image4]: results/images/Training_Console_Agent_Tennis_Final.png "Console Solved"
[image5]: results/images/Training_Console_Agent_Tennis_Final_08.png "Console Score Long Learning"
[image6]: results/images/Training_Console_Agent_Tennis_Final_03_TestScores.png "Console Score Test"
[image7]: https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-04_at_10.11.20_PM.png "TrainingMA_Paper"
[image8]: results/training_scores/simple_training_plot_trial_Prob_Noise_01.png "First Trial"
[image9]: results/training_scores/simple_training_plot_trial_Prob_Noise_DeepNet_02.png "Second Trial"
[image10]: results/training_scores/simple_training_plot_trial_Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_03_first_version.png "Third Trial"

# Report Introduction

This report consists of three main sections. In the first section [Learning Algorithm](#learning-algorithm) the technicalities are explained. In this section the Model Architecture, Multi-Agent Deep Deterministic Policy Gradient (MADDPG), the Noise Process and the Hyperparameters used for this project are introduced and explained. The second section [Plot of Rewards](#plot-of-rewards) contains visualizations of the scores achieved during several successful training runs. In the third section [Ideas for Future Work](#ideas-for-future-work) multiple approaches to include in future work on this project are proposed.


# Learning Algorithm

For this environment a Deep Deterministic Policy Gradient (DDPG) is not right away suited to fit a multi agent setting. Thus, we adapt the DDPG to the Multi-Agent DDPG (MADDPG).

The DDPG, introduced by Lillicrap et al. (2015), consists of a replay buffer, an acotr-critic architecture, and a deterministic policy. It operates by first using a policy (actor) network to choose an action, which leads to a reward and a transition to a new state. The resulting experience, which compromises the current state, action, reward, and next state, is stored in a replay buffer. During training, random batches of these stored experiences are sampled. The critic network is updated using these samples to better estimate the value of state-action pairs. Subsequently, the actor (policy) network is refined using gradients derived from the critic, improving the action-selection policy over time (Zhou, Huang, and Fränti, 2021).

Lowe et al. (2017) extend the DDPG into a multi-agent policy gradient algorithm. Each agent maintains its own actor (policy) network for decentralized decision-making but leverages a centralized critic during training. This critic has access to the full state and actions of all agents, enabling it to model interactions and learn coordinated strategies. By combining centralized training with decentralized execution, MADDPG allows agents to learn in a cooperative or competitive setting while remaining scalable and robust to multi-agent dynamics (Lowe et al., 2017). The figure below, taken from Lowe et al. (2017), depicts a centralized training with decentralized execution.

![TrainingExecutionMADDPG][image7]

## Model Architecture Actor-Critic Network

The Actor-Critic Network, which we already implemented earlier in this course, was utilized as model for solving the tennis problem. Some adapations were necessary to fit a multi agent setup in the critic network.

In the script `actor_critic.py` one can find the implementation of the Actor and Critic Network. The following code snippet shows the Model architecture. The implementation of the Actor Critic network was taken from the previous exercises of the pendulum from the Udacity course (https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py). Additionaly, for a faster and more stable training, batch normalization layers were added to both the actor and critic network. Ioffe & Szegedy (2015) introduced the concept of batch normalization to make the training process more efficent. Also Lillicrap et al. (2015) suggest to utilize batch normalization layers for the actor-critic network. Furthermore, to fit the Multi-Agent setup, we expand the critics input to allow for centralized training. The critic now takes in all agents observations and actions. This leads to richer information during training and coordinates training.

The Actor (Policy) Model consists of three linear layers, ReLU activation functions, and batch normalization layers. The forward pass computes the action (or policy) for a given state. It uses two fully connected layers, (optionally) with batch normalization and ReLU activations, to process the state input and generate the action. The final action is transformed using the tanh function to ensure it's within the range [-1, 1]. For the final agent 256 units for the first and 256 units for the second fully connected layer were selected.

The Critic (Value) Model consists of three linear layer, a batch nomarlization layer and ReLU activation functions. The forward pass processes the state and action pairs to output a Q-value. The state and actions are concatenates and passed through a fully connected layer, and if batch normalization is enabled, normalization is applied. The state and action are then concatenated and passed through another fully connected layer before producing the final Q-value. For the final agent 128 units for the first and 128 units for the second fully connected layer were selected. 

```python
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=128, batch_norm=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units) 
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units) 
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.batch_norm = batch_norm
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.batch_norm:
            x = self.bn0(state)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, num_agents, seed, fcs1_units=64, fc2_units=128, batch_norm = True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        input_size = (state_size + action_size) *num_agents
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(input_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.b1 = nn.BatchNorm1d(fcs1_units)
        self.batch_norm = batch_norm
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1) 
        x = self.fcs1(x)                       
        if self.batch_norm:
            x = self.b1(x)                     
        x = F.relu(x)                          
        x = F.relu(self.fc2(x))                
        return self.fc3(x)         

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
```

## Experiements that lead to solving the environment

Throughout the project many different hyperparameters were tried. At first, a realtively small buffer size (100000) with less frequent learning updates (10), a batch size of 128 and a relative simple actor-critic network with units below of 64 and 32 for the actor and 64 and 128 for the critic was selected. After 10.000 Episodes there was no learning. Below one can see the first trial of the training plot. As one can see, the agents did not learn anything significant.

![First Trial][image8]

 Thus, multiple hyperparameters were adjusted. The first idea was to increase the complexity of the actor and critic networks and increase the batch size. The units were all adjusted to 256 aswell as the batch size. Below one can see the training scores throughout the first 10.000 episodes. However, the agents still did not seem to learn anything meaningful.

![Second Trial][image9]

Next, the learning frequency was reduced from 10 to 2 to see wheter the agents need more frequent learning updates. Additionaly, the buffer size was increased by a factor of 10 to 1000000, as to allow more memory. Additionaly, batch normalization layers was enabled and the units of the critic network were reduced to 128 units for both layers. Also the batch size was reduced again to 64. The adjustments of the parameters seemed to introduced better learning. However, after 6000 epsiodes the target score of +0.5 over 100 episodes still was not reached. 

![Third Trial][image10]

More trials were made with reducing the amounts of learning updates (LEARN_FREQ), however learning appeared to get worse again. Thus, one key finding is to often times update the learning by keeping the paramater LEARN_FREQ low. Furthermore, the batch normalization was removed and finally a maximum exploitation step was introduced. With these adjustments, the environment was considered solved after about 3600 episodes. Therefore we refer to the next section.

The final set of Hyperparameters can be found in the next section
## Chosen Hyperparameters & brief Explanation of Hyperparameters

Many different Hyperparameters were tested after the final MADDGP agent was implemented. The most impactful parameters, which changed the performance of the agent drastically, were the learning frequence, the gradient updates and the units of the fully connected layers. 

The following set of Hyperparameters resulted in solving the environment.

| Hyperparameter         | Value             | Description |
|:------------------------|:------------------|:------------|
| TRIAL_NAME              | Final_Agent        | Name for saving checkpoints, plots, config files. |
| N_EPISODES              | 6000               | Number of episodes to train the agent. |
| MAX_T                   | 1000               | Max time steps per episode. |
| BUFFER_SIZE             | 1000000            | Replay buffer size. Improves sample diversity. |
| BATCH_SIZE              | 64                 | Number of experiences per training update. |
| GAMMA                   | 0.99               | Discount factor. Higher values emphasize future rewards. |
| TAU                     | 0.001              | Soft update factor for target networks. Smaller results in slower, more stable updates. |
| LEARN_FREQ              | 2                  | Defines after how many steps the agent should learn. |
| GRADIENT_UPDATES        | 4                  | Number of optimization steps per learning trigger. |
| REWARD_SCALING          | false              | Decision whether rewards are scaled. |
| SCALE_FACTOR_REWARD     | 0.01               | Scaling factor applied to rewards (if enabled). |
| **Actor Parameters**    |                    |  |
| LR_ACTOR                | 0.0001             | Learning rate for the actor network. |
| FC1_UNITS (Actor)       | 256                | Units in the first fully connected actor layer. |
| FC2_UNITS (Actor)       | 256                | Units in the second fully connected actor layer. |
| BATCH_NORMALIZATION (Actor) | true           | Whether BatchNorm Layer is applied in actor network. |
| **Critic Parameters**   |                    |  |
| LR_CRITIC               | 0.001              | Learning rate for the critic network. |
| FC1_UNITS (Critic)      | 128                | Units in the first fully connected critic layer. |
| FC2_UNITS (Critic)      | 128                | Units in the second fully connected critic layer. |
| BATCH_NORMALIZATION (Critic) | true          | Whether BatchNorm Layer is applied in critic network. |
| WEIGHT_DECAY (Critic)   | 0.0                | L2 regularization strength for critic network. |
| **Noise Parameters**    |                    |  |
| PROB_NOISE_OR_OU        | 'prob'             | Choice between probabilistic ('prob') or OU noise ('ou'). |
| ACT_NOISE_DECAY         | false              | Decision whether noise decays should be included. |
| STOP_NOISE              | 30000              | Stop noise after these amount of steps to interrupt exploration. |
| **OU Noise Config - Was not used for final Agent**     |                    |  |
| MU                     | 0.0                 | Mean of OU process. |
| THETA                  | 0.15                | Mean reversion rate in OU process. |
| SIGMA                  | 0.15                | Volatility (noise level) in OU process. |
| **Probabilistic Noise Config - Was not used for final Agent** |             |  |
| NOISE_INIT             | 0.99                | Initial noise scale. |
| NOISE_DECAY            | 0.95                | Decay rate of probabilistic noise. |
| NOISE_MIN              | 0.01                | Minimum allowed noise level. |

# Plot of Rewards

Two models were trained which reached the target score. In the first attempt, the training was stopped right after reaching the target score. In the second attempt, the target score was lifted to 1 to see how fast it is possible to achieve better scores after reaching the inital target score.

![Plot Solved 1][image1]

![Plot Solved 1_1][image4]

One can see from the plot and the console output above that the target score was achieved after 3664 Episodes.

![Plot Solved 2][image2]

![Plot Solved 2_1][image5]

Another trial showed that a lot higher scores can be achieved quickly after reaching the target score. One can see from the plot and the console output above that the target score of an average score of +0.5 over 100 episodes was achieved after 3664 Episodes. However, if one keeps training the agents afterward reaching the target score, it just takes an extra 285 Episodes to reach an average score of +1 over 100 consecutive episodes. Another noticebale thing is that the average score of 100 episodes drops below the target score of 0.5 at in the 3700-3800 episode.

# Ideas for Future Work

As seen in this project the MADDPG had difficulties with the exploration of the high-dimensional state action space. Chen et al. (2020) also stress these challenges and propose a novel technique called Experience Augmentation. This enables time-efficent and boosted learning. In their work they combine this technique with a MADDPG in heterogenous and homogenous environments. They show that experience augementation is curical for accelerating the training process and boosting the convergence. Another promising area would be to further tune the Hyperparameters. So far only a trial and error approach was performed. However, one could utilize a more structured approach such as a random or grid search. Alternatively one could also think of bayesian optimization to identify the optimal hyperparamters and thus reduce training time and signficiantly improve the performance of the agents. Iqbal & Sha (2018) propose a Actor-attention-critic for multi-agent reinforcement learning and show in their experiemental results that the Multi-Actor-Attention-Critic (MAAC) reaches faster higher mean average rewards for certain environments and for other environments competitive results. Finally, we could utilize the Distributed distributional deterministic policy gradients (D4PG) proposed by Barth-Maron et al. (2018) and expand it to a Mutli-Agent setting.

# References

1. Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W., Horgan, D., Tb, D., … Lillicrap, T. (2018). Distributed distributional deterministic policy gradients. Retrieved from http://arxiv.org/abs/1804.08617
2. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating deep network training by reducing internal covariate shift. doi:10.48550/ARXIV.1502.03167 (https://arxiv.org/pdf/1502.03167)
3. Iqbal, S., & Sha, F. (2018). Actor-attention-critic for multi-agent reinforcement learning. Retrieved from http://arxiv.org/abs/1810.02912
4. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., … Wierstra, D. (2015). Continuous control with deep reinforcement learning. doi:10.48550/ARXIV.1509.02971 (https://arxiv.org/pdf/1509.02971)
5. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. doi:10.48550/ARXIV.1706.02275
6. Ye, Z., Chen, Y., Song, G., Yang, B., & Fan, S. (2020). Experience Augmentation: Boosting and accelerating off-policy multi-agent Reinforcement Learning. Retrieved from http://arxiv.org/abs/2005.09453
7. Zhou, C., Huang, B., & Fränti, P. (2022). A review of motion planning algorithms for intelligent robots. Journal of Intelligent Manufacturing, 33(2), 387–424. doi:10.1007/s10845-021-01867-z (https://doi.org/10.1007/s10845-021-01867-z)