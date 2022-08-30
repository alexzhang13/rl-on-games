import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..agent import Agent

def layer_init (layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO (nn.Module):
    '''
    Defines PPO value network and policy networks for training.
    '''
    def __init__(self, obs_shape, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self._initialize_weights()
        
        self.critic_linear = layer_init(nn.Linear(512, 1), std=1.0)
        self.actor_linear = layer_init(nn.Linear(512, num_actions), std=0.01)
        
    def _initialize_weights (self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_init(module)
    
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.reshape(x.size(0), -1)) / 255.0 # flatten
        return self.actor_linear(x), self.critic_linear(x)

class PPOAgent (Agent):
    '''
    General Agent PPO class, handles all training details as well as
    any non-deep model calculations.
    '''
    def __init__(self, 
                 obs_shape, 
                 num_actions, 
                 num_envs, 
                 num_rollout_steps, 
                 device, 
                 update_epochs=4,
                 num_minibatches=4,
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_coeff=0.2,
                 entropy_coeff=0.01,
                 value_loss_coeff=0.5,
                 max_grad_norm=0.5,
                 target_kl=None,
                 lr=1e-4):
        super(Agent, self).__init__()
        self.PPO = PPO(obs_shape, num_actions).to(device)
        self.optim = torch.optim.Adam(self.PPO.parameters(), lr=lr, eps=1e-5)
        self.lr = lr
        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coeff = clip_coeff
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
    
    def get_action_and_value (self, obs, action=None):
        obs = obs.permute(0, 3, 1, 2) # N H W C -> N C H W
        logits, values = self.PPO(obs)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), values
    
    def train (self, envs, batch_size, next_obs, next_done, step, max_steps, global_step, writer):
        # learning rate annealing
        frac = 1.0 - (step - 1.0) / max_steps
        lr = frac * self.lr
        self.optim.param_groups[0]["lr"] = lr
        
        # define rollout Tensors
        obs = torch.zeros((self.num_rollout_steps, self.num_envs) + envs.single_observation_space.shape).to(self.device)
        
        actions = torch.zeros((self.num_rollout_steps, self.num_envs) + envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        
        # rollout loop
        for step in range(0, self.num_rollout_steps):
            global_step += 1 * self.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = self.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = torch.Tensor(next_obs).to(self.device) 
            next_done = torch.Tensor(next_done).to(self.device)
             
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                    writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)
                    break
        
        # compute GAE by bootstrapping 
        with torch.no_grad():
            _,_,_,next_value = self.get_action_and_value(next_obs)
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_rollout_steps)):
                # unravel rollout steps that were sampled
                if t == self.num_rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            returns = advantages + values
        
        # flatten the batches for training
        minibatch_size = batch_size // self.num_minibatches
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # optimizing policy and value net
        b_inds = np.arange(batch_size)
        clipfracs = [] # how many times clip is triggered
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
            
                _, newlogprob, entropy, new_values = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                # advantage normalization
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # estimate kl divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coeff).float().mean()]
                
                # policy loss
                pg_loss = -mb_advantages * ratio
                clipped_pg_loss = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
                pg_loss = torch.max(pg_loss, clipped_pg_loss).mean()
                
                # value loss clipping
                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -self.clip_coeff,
                    -self.clip_coeff
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()   
                
                # entropy loss
                entropy_loss = entropy.mean()
                loss = pg_loss - self.entropy_coeff * entropy_loss + self.value_loss_coeff * v_loss
                
                # backprop computed loss
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.PPO.parameters(), self.max_grad_norm)
                self.optim.step()
        
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break
            
        # debug view for variances
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var([y_true, y_pred]) / var_y
        
        # write to tensorboard
        writer.add_scalar("charts/learning_rate", self.optim.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipped_frac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        return global_step