[//]: # (Image References)

[image1]: results/training_scores/training_scores_trial_Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_03.png "Reward Plot Final Agent"
[image2]: results/training_scores/training_scores_trial_Prob_Noise_DeepNet_BufferPlus_Learn_FreqMin_BatchNorm_08.png "Reward Plot Final Agent Long Learning"
[image3]: results/images/Tennis_Final_Agent_03.gif "Trained Agent"
[image4]: results/images/Training_Console_Agent_Tennis_Final.png "Console Solved"
[image5]: results/images/Training_Console_Agent_Tennis_Final_08.png "Console Score Long Learning"
[image6]: results/images/Training_Console_Agent_Tennis_Final_03_TestScores.png "Console Score Test"
[image7]: results/images/OverviewOfMultiAgentdecentralizeActor_CentralizedCritic.png "TrainingMA_Paper"

# Report Introduction

This report consists of three main sections. In the first section [Learning Algorithm](#learning-algorithm) the technicalities are explained. In this section the Model Architecture, Multi-Agent Deep Deterministic Policy Gradient (MADDPG), the Noise Process and the Hyperparameters used for this project are introduced and explained. The second section [Plot of Rewards](#plot-of-rewards) contains visualizations of the scores achieved during several successful training runs. In the third section [Ideas for Future Work](#ideas-for-future-work) multiple approaches to include in future work on this project are proposed.


# Learning Algorithm

For this environment a Deep Deterministic Policy Gradient (DDPG) is not right away suited to fit a multi agent setting. Thus, we adapt the DDPG to the Multi-Agent DDPG (MADDPG). Lowe et al. (2017) extend DDPG into a multi-agent policy gradient algorithm. 

The figure below, taken from Lowe et al. (2017), depicts a centralized training with decentralized execution.

![TrainingExecutionMADDPG][image7]

In the Execution phase, each agent has its own policy network, observes local observations and produces an action independently (and thus decentralized). Policies act without centralized information during the execution. During training, access to more information is allowed (Thus it is centralized) Each agent's critic network is trained using global state/action information even from other agents. Thus allowing the agent to learn coordinated stratgies even though they act independently at execution time.

## Model Architecture Actor-Critic Network

The Actor-Critic Network, which we already implemented earlier in this course, was utilized as model for solving the tennis problem.

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

# Plot of Rewards

Two models were trained which reached the target score. In the first attempt, the training was stopped right after reaching the target score. In the second attempt, the target score was lifted to 1 to see how fast it is possible to achieve better scores after reaching the inital target score.

![Plot Solved 1][image1]

![Plot Solved 1_1][image4]

One can see from the plot and the console output above that the target score was achieved after 3664 Episodes.

![Plot Solved 2][image2]

![Plot Solved 2_1][image5]

Another trial showed that a lot higher scores can be achieved quickly after reaching the target score. One can see from the plot and the console output above that the target score of an average score of +0.5 over 100 episodes was achieved after 3664 Episodes. However, if one keeps training the agents afterward reaching the target score, it just takes an extra 285 Episodes to reach an average score of +1 over 100 consecutive episodes. Another noticebale thing is that the average score of 100 episodes drops below the target score of 0.5 at in the 3700-3800 episode.

# Ideas for Future Work


# References

[1] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. doi:10.48550/ARXIV.1706.02275