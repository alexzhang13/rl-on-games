import argparse
import retro
import yaml

import train.mario as ENV
from stable_baselines3 import PPO 
import models.baselines as baselines

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="Location of config file to run", 
                    default='experiments/mario/train_mario_v0.yaml')
args = parser.parse_args()

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


def main():
    # TODO: Configure with YAML
    env = ENV.env_init()
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space.shape)

    # Load in configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=config['LOG_DIR'], learning_rate=1e-6, n_steps=512)
    callback = baselines.TrainAndLoggingCallback(frequency=10000, path=config['SAVE_PATH'])
    model.learn(total_timesteps=1e6, callback=callback)
    # ENV.train(env=env, agent=model, callback=callback)


if __name__ == "__main__":
    main()