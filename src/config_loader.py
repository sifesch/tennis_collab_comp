import yaml
from dataclasses import asdict, dataclass

class Configurations:
    """
    Loads and organizes configuration values from a YAML file into structured dataclass objects.

    Attributes:
        hyperparameters (Hyperparameters): Training and model-related hyperparameters.
        noise_config (NoiseConfig): Settings related to action noise for exploration.
    """
    def __init__(self, config_path: str = 'config/train_config.yaml'):
        config_dict = self.read_yaml(config_path)

        # Hyperparameters
        hp = config_dict['HYPERPARAMETERS']
        self.hyperparameters = Hyperparameters(
            N_EPISODES = hp['N_EPISODES'],
            MAX_T = hp['MAX_T'],
            TRIAL_NAME = hp['TRIAL_NAME'],
            BUFFER_SIZE = hp['BUFFER_SIZE'],
            BATCH_SIZE = hp['BATCH_SIZE'],
            BUFFER_SIZE_MADDPG = hp['BUFFER_SIZE_MADDPG'],
            BATCH_SIZE_MADDPG = hp['BATCH_SIZE_MADDPG'],
            GAMMA = hp['GAMMA'],
            TAU = hp['TAU'],
            LEARN_FREQ = hp['LEARN_FREQ'],
            GRADIENT_UPDATES = hp['GRADIENT_UPDATES'],
            SCALE_FACTOR_REWARD = hp['SCALE_FACTOR_REWARD'],
            REWARD_SCALING = hp['REWARD_SCALING'],
            ACTOR_PARAMS = ActorParams(**hp['ACTOR_PARAMS']),
            CRITIC_PARAMS = CriticParams(**hp['CRITIC_PARAMS']),
        )

        noise = config_dict['NOISE']
        self.noise_config = NoiseConfig(
            GENERAL = GeneralNoiseConfig(**noise['GENERAL']),
            OUNoise_Config = OUNoiseConfig(**noise['OUNoise_Config']),
            ProbNoise_Config = ProbNoiseConfig(**noise['ProbNoise_Config']),
        )

    def save_config(config_obj, output_path: str):
        """
        Saves a dataclass-based configuration object to a YAML file.

        Args:
            config_obj (Configurations): The main config object.
            output_path (str): Path to the output YAML file.
        """
        # Convert all nested dataclasses to dictionaries
        config_dict = {
            'HYPERPARAMETERS': asdict(config_obj.hyperparameters),
            'NOISE': asdict(config_obj.noise_config),
        }

        with open(output_path, 'w') as file:
            yaml.dump(config_dict, file, sort_keys=False)

    @staticmethod
    def read_yaml(file_path:str = 'config/train_config.yaml') -> dict:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

@dataclass
class ActorParams:
    """
    Configuration for the actor network.

    Attributes:
        LR_ACTOR (float): Learning rate for the actor network.
        FC1_UNITS (int): Units in the first fully connected layer.
        FC2_UNITS (int): Units in the second fully connected layer.
        BATCH_NORMILIZATION (bool): Whether to apply batch normalization.
    """
    LR_ACTOR: float
    FC1_UNITS: int
    FC2_UNITS: int
    BATCH_NORMILIZATION: bool

@dataclass
class CriticParams:
    """
    Configuration for the critic network.

    Attributes:
        LR_CRITIC (float): Learning rate for the critic network.
        FC1_UNITS (int): Units in the first fully connected layer.
        FC2_UNITS (int): Units in the second fully connected layer.
        BATCH_NORMILIZATION (bool): Whether to apply batch normalization.
        WEIGHT_DECAY (float): L2 regularization strength.
    """
    LR_CRITIC: float
    FC1_UNITS: int
    FC2_UNITS: int
    BATCH_NORMILIZATION: bool
    WEIGHT_DECAY: float

@dataclass
class Hyperparameters:
    """
    Contains core training and optimization hyperparameters.

    Attributes:
        TRIAL_NAME (str): For storing and identifying results
        BUFFER_SIZE (int): Replay buffer size.
        BATCH_SIZE (int): Batch size for training.
        GAMMA (float): Discount factor for future rewards.
        TAU (float): Soft update factor for target networks.
        LEARN_FREQ (int): Frequency of learning steps.
        GRADIENT_UPDATES (int): Number of gradient updates per learning step.
        ACTOR_PARAMS (ActorParams): Actor network configuration.
        CRITIC_PARAMS (CriticParams): Critic network configuration.
    """
    TRIAL_NAME: str
    N_EPISODES: int
    MAX_T: int
    BUFFER_SIZE: int
    BATCH_SIZE: int
    GAMMA: float
    TAU: float
    LEARN_FREQ: int
    GRADIENT_UPDATES: int
    REWARD_SCALING: bool
    SCALE_FACTOR_REWARD: int
    ACTOR_PARAMS: ActorParams
    CRITIC_PARAMS: CriticParams

@dataclass
class OUNoiseConfig:
    """
    Parameters for Ornstein-Uhlenbeck noise used in continuous action spaces.

    Attributes:
        MU (float): Long-running mean of the process.
        THETA (float): Speed of mean reversion.
        SIGMA (float): Volatility or randomness in the noise.
    """
    MU: float
    THETA: float
    SIGMA: float

@dataclass
class ProbNoiseConfig:
    """
    Parameters for probabilistic noise used in exploration strategies.

    Attributes:
        NOISE_DECAY (float): Rate at which the noise decays over time.
        NOISE_INIT (float): Initial noise scale.
        NOISE_MIN (float): Minimum noise value after decay.
    """
    NOISE_DECAY: float
    NOISE_INIT: float
    NOISE_MIN: float

@dataclass
class GeneralNoiseConfig:
    """
    General noise settings for choosing and controlling exploration strategies.

    Attributes:
        PROB_NOISE_OR_OU (str): Choice between 'prob' or 'ou' noise.
        ACT_NOISE_DECAY (float): Decay rate of the applied action noise.
    """
    PROB_NOISE_OR_OU: str
    ACT_NOISE_DECAY: float

@dataclass
class NoiseConfig:
    """
    Top-level container for noise configuration used during training.

    Attributes:
        GENERAL (GeneralNoiseConfig): General configuration for noise strategy.
        OUNoise_Config (OUNoiseConfig): Config for Ornstein-Uhlenbeck noise.
        ProbNoise_Config (ProbNoiseConfig): Config for probabilistic noise.
    """
    GENERAL: GeneralNoiseConfig
    OUNoise_Config: OUNoiseConfig
    ProbNoise_Config: ProbNoiseConfig

