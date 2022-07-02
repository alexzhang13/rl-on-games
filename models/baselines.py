import os
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback function (control frequency of saving)
    """
    def __init__(self, frequency, path, verbose=1):
        super().__init__(verbose)
        self.frequency = frequency
        self.path = path

    def _init_callback(self):
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            save_path = os.path.join(self.path, 'checkpoint_{}'.format(self.num_calls))
            self.model.save(save_path)
        
        return True

def get_baseline(name):
    if name == 'PPO':
        return PPO