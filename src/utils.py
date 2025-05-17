import numpy as np
import matplotlib.pyplot as plt

def create_complex_training_plot(scores: list, trialname: str = '01', window_size: int = 10):
    """
    function to create a training plot and save it.

    Params:
    ======
    scores: scores from training run
    trial_num: string to better organize result files. Indicates the trial number.
    """
    # Compute rolling average (moving average) using numpy's convolv function
    rolling_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

    # Plot scores and rolling average
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(scores)), scores, label="Scores", alpha=0.5)
    plt.plot(np.arange(window_size-1, len(scores)), rolling_avg, label=f"{window_size}-Episode Rolling Avg", color='orange', linewidth=2)
    plt.axhline(y=30, color='r', linestyle='--', linewidth=2, label="Target Score")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(f'results/training_scores/training_scores_trial_{trialname}.png', bbox_inches='tight', transparent=True, facecolor='white')
    plt.show()

def create_simple_training_plot(scores,trialname):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'results/training_scores/simple_training_plot_trial_{trialname}')