import argparse
import retro
import gym
import yaml
import numpy as np
import random
import datetime
import torch
from pathlib import Path
from distutils.util import strtobool
import torch.multiprocessing as _mp

import train.mario as ENV

from environments.mario_env import MultipleEnvironments

from models import (
    baselines,
    agent,
)

from models.networks import (
    PPO,
    DQN
)

from utils import (
    logger
)
import time
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="Location of config file to run", 
                    default='experiments/mario/train_mario_v0_ppo.yaml')
parser.add_argument('--model_weights_path', type=str, help="Location of model weights to load", 
                    default='saved/')
parser.add_argument('--evaluate', action='store_true', help="Evaluate model")
parser.add_argument('--cpu', type=lambda x:bool(strtobool(x)), 
                    default=False, nargs="?", 
                    const=True, help="Choose to not use CUDA by default.")
parser.add_argument('--wandb', type=lambda x:bool(strtobool(x)), 
                    default=False, nargs="?",
                    const=True, help="Toggle to enable weights and biases tracking.")
parser.add_argument('--wandb-project-name', type=str, default="marioRL", help="wandb project name")
parser.add_argument('--wandb-entity', type=str, default=None, help="specify an entity (team) for the wandb project")
parser.add_argument('--capture-video',type=lambda x:bool(strtobool(x)), 
                    default=False, nargs="?",
                    const=True, help="Toggle to save videos of agent")

args = parser.parse_args()

def train(config, env, device, mp=None):
    # set up logging and saving
    save_dir = Path("output/") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # configure wandb
    if args.wandb:
        import wandb
        
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name="mario_rl",
            monitor_gym=True,
            save_code=True,
        )
    
    # configure tensorboard
    run_name = f"mario__{int(time.time())}"
    writer = SummaryWriter(str(save_dir) + f"/runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}||" for key,value in vars(args).items()])),
                    )
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}||" for key,value in config.items()])),
                    )
    
    log = logger.MetricLogger(save_dir, writer)
    
    # TODO: Change model config selection to be more general
    if config['model']['model_name'] == 'DQN':
        model = DQN.DQNAgent(num_frames=config['environment']['n_stack'], 
                            num_actions=env.action_space.n,
                            device=device,
                            evaluate=False,
                            lr=config['model']['learning_rate'],
                            save_dir=save_dir,
                            warmup=100)
    elif config['model']['model_name'] == 'PPO':
        model = PPO.PPOAgent(obs_shape=env.num_states,
                            num_actions=env.num_actions,
                            lr=config['model']['learning_rate'],
                            num_envs=config['environment']['num_envs'],
                            num_rollout_steps=config['model']['rollout_steps'],
                            gamma=config['model']['gamma'],
                            gae_lambda=config['model']['gae_lambda'],
                            clip_coeff = config['model']['clip_coeff'],
                            entropy_coeff = config['model']['entropy_coeff'],
                            value_loss_coeff = config['model']['value_loss_coeff'],
                            max_grad_norm = config['model']['max_grad_norm'],
                            target_kl = config['model']['target_kl'],
                            device=device)
    
    if config['environment']['sync_vector_env']:
        ENV.train_vectorized(env,
                            agent=model,
                            writer=writer,
                            mp=mp,
                            config=config,
                            num_envs=config['environment']['num_envs'],
                            batch_size=config['model']['batch_size'],
                            num_steps=config['environment']['max_step'],
                            device=device,
                            render=False)
    else:
        ENV.train(env=env, 
                agent=model, 
                logger=log, 
                episodes=config['environment']['episodes'],
                render=False)

def evaluate(config, env, device):
    if config['model']['model_name'] == 'DQN':
            model = DQN.DQNAgent(num_frames=config['environment']['n_stack'], 
                                num_actions=env.action_space.n,
                                device=device,
                                evaluate=True,
                                lr=config['model']['learning_rate'],
                                warmup=0)
    elif config['model']['model_name'] == 'PPO':
        model = PPO.PPOAgent(obs_shape=np.array(env.num_states),
                                num_actions=env.num_actions,
                                lr=config['model']['learning_rate'],
                                num_envs=config['environment']['num_envs'],
                                num_rollout_steps=config['model']['rollout_steps'],
                                gamma=config['model']['gamma'],
                                gae_lambda=config['model']['gae_lambda'],
                                clip_coeff = config['model']['clip_coeff'],
                                device=device)
    model.load(args.model_weights_path, device)
    ENV.evaluate(env=env, agent=model, render=True)

def main():
    # Load in configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set seeds
    np.random.seed(config['SEED'])
    random.seed(config['SEED'])
    torch.manual_seed(config['SEED'])
    torch.backends.cudnn.deterministic = True
    
    # Set up vectorized (multiple) or single environment
    if config['environment']['sync_vector_env']:
        # open multi-processing
        mp = _mp.get_context("spawn")
        env = MultipleEnvironments(config['environment']['world'], 
                                   config['environment']['stage'], 
                                   config['environment']['action_type'], 
                                   config['environment']['num_envs'])
        
        print(dir(env.envs[0]))
        assert isinstance(env.single_action_space, gym.spaces.Discrete) # assert gym env
        
        print('Action Space:', env.single_action_space)
        print('Observation Space:', env.single_observation_space.shape)
        print('Config Model Keys', config['model'].keys())
        # env = gym.vector.SyncVectorEnv([ENV.env_init(stacked_frames=False, seed=config['SEED'] + i) \
        #                                 for i in range(config['environment']['num_envs'])])
    else:
        env = ENV.env_init(n_stack=config['environment']['n_stack'], seed=config['SEED'])()
    
    # check CUDA config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {use_cuda}")
    print()
    
    if args.evaluate:
        evaluate(config, env, device)
    else:
        train(config, env, device, mp=mp)
        

if __name__ == "__main__":
    main()
