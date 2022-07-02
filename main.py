import argparse
import retro
import train.mario as ENV

parser = argparse.ArgumentParser()

def retro_test():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


def main():
    # TODO: Configure with YAML
    env = ENV.env_init()
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space.shape)

    ENV.train(env)


if __name__ == "__main__":
    main()