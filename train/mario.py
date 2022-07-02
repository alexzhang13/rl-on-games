import gym_super_mario_bros

# Joypad Wrapper
from nes_py.wrappers import JoypadSpace

# Simplified controls for mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import FrameStack, GrayScaleObservation

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def train(env):
    done = True
    for step in range(100000):
        if done:
            env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

def env_init(environment_name='SuperMarioBros-v0',
            grayscale=True, 
            stacked_frames=True):
    """
    Initialize gym environment for mario game.
    args: 
        environment_name: name of environment to run
        grayscale: whether to grayscale frames
        stacked_frames: whether to stack frames
    """
    # create base mario environment
    env = gym_super_mario_bros.make(environment_name)
    # simplify action space
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # grayscale images
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    # wrap in vectorized frames 
    if stacked_frames:
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

    return env


