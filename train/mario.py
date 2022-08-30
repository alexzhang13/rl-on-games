import gym_super_mario_bros
import torch

# Joypad Wrapper
from nes_py.wrappers import JoypadSpace

# Simplified controls for mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def train_vectorized(envs, agent, writer, num_envs, 
                     device, batch_size, num_steps, render=False):
    
    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = num_steps // batch_size
    
    print('obs shape', next_obs.shape)
    
    # Loop through updates, PPO handles train loop internally and rollouts
    for update in range(1, num_updates + 1):
        global_step = agent.train(envs, batch_size, next_obs, next_done, update, num_updates, global_step, writer)
    
    writer.close()
    envs.close()

def train(env, agent, episodes, logger, render=False):
    for ep in range(episodes):
        state = env.reset()
        while True:
            
            # agent step and learn
            action = [agent.action(state)]
            
            new_state, reward, done, info = env.step(action)
            agent.cache(state, action, reward, new_state, done)
            q, loss = agent.optimize()
            
            logger.log_step(reward, loss, q)
            
            state = new_state
            
            if done or info[0]["flag_get"]:
                break
            
            if render:
                env.render()
                
        # log episode info
        logger.log_episode()
        if ep % 5 == 0:
            logger.record(episode=ep, epsilon=agent._epsilon, step=agent._curr_step)
    env.close()
    
    
def evaluate(env, agent, render=False):
    done = True
    state = env.reset()
    for step in range(100000):
        if done:
            env.reset()
        print(type(state))
        action = [agent.action(state)]
        new_state, reward, done, info = env.step(action)
        state = new_state
        if render:
            env.render()
    env.close()


# -=-=-=-=-=-=-= ENVIRONMENT BEHAVIOR -=-=-=-=-=-=-=-=- #

def env_init(environment_name='SuperMarioBros-v0',
            n_stack=4,
            grayscale=True, 
            stacked_frames=True,
            seed=7):
    """
    Initialize gym environment for mario game.
    args: 
        environment_name: name of environment to run
        grayscale: whether to grayscale frames
        stacked_frames: whether to stack frames
    """
    def make_env():
        # create base mario environment
        env = gym_super_mario_bros.make(environment_name)
        
        # set seed
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # simplify action space
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        # grayscale images
        if grayscale:
            env = GrayScaleObservation(env, keep_dim=True)

        # resize to 84x84 
        env = ResizeObservation(env, shape=84)
        # env = FrameStack(env, 4)

        # wrap in vectorized frames 
        if stacked_frames:
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack, channels_order='last')

        return env
    return make_env





