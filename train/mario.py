import gym_super_mario_bros

# Joypad Wrapper
from nes_py.wrappers import JoypadSpace

# Simplified controls for mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def train(env, agent, episodes, logger, render=False):
    for ep in range(episodes):
        print(f"episode {ep+1}")
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
        action, _ = agent.predict(state)
        state, reward, done, info = env.step(action)
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
    # create base mario environment
    env = gym_super_mario_bros.make(environment_name)
    
    # set seed
    env.seed(seed)
    
    # simplify action space
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # grayscale images
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)

	# resize to 84x84 
    env = ResizeObservation(env, shape=84)

    # wrap in vectorized frames 
    if stacked_frames:
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack, channels_order='last')

    return env


