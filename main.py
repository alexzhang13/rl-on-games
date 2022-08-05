import argparse
import retro
import yaml
import numpy as np
import random
import datetime
import torch
from pathlib import Path

import train.mario as ENV
from models import (
    baselines,
    agent
)
from utils import (
    logger
)
import time
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="Location of config file to run", 
                    default='experiments/mario/train_mario_v0.yaml')
parser.add_argument('--evaluate', action='store_true', help="Evaluate model")
args = parser.parse_args()
SEED = 7

def retro_test():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
            break
    env.close()


def test_environment(train=True):
    env = ENV.env_init()
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space.shape)

    # Load in configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if train:
        model = baselines.get_baseline('PPO')('CnnPolicy', env, verbose=1, tensorboard_log=config['LOG_DIR'], learning_rate=1e-6, n_steps=512)
        callback = baselines.TrainAndLoggingCallback(frequency=10000, path=config['SAVE_PATH'])
        model.learn(total_timesteps=1e6, callback=callback)
        ENV.train(env=env, agent=model, callback=callback)
    else:
        model = baselines.get_baseline('PPO').load('./output/checkpoints/checkpoint_340000')
        ENV.evaluate(env=env, agent=model, render=True)


def main():
    # Load in configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # TODO: Configure with YAML
    env = ENV.env_init(n_stack=config['environment']['n_stack'], seed=SEED)
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space.shape)
    
    # check CUDA config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {use_cuda}")
    print()
    
    # configure tensorboard
    # run_name = f"mario__{int(time.time())}"
    # writer = SummaryWriter(f"runs/{run_name}")
    
    # set up logging and saving
    save_dir = Path("output/") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    log = logger.MetricLogger(save_dir)
    
    if args.evaluate:
        pass
    else:
        model = agent.DQNAgent(num_frames=config['environment']['n_stack'], 
                               num_actions=env.action_space.n,
                               device=device,
                               lr=config['model']['learning_rate'],
                               save_dir=save_dir,
                               warmup=100)
        ENV.train(env=env, 
                  agent=model, 
                  logger=log, 
                  episodes=config['environment']['episodes'],
                  render=False)
        

if __name__ == "__main__":
	np.random.seed(SEED)
	random.seed(SEED)
	main()
