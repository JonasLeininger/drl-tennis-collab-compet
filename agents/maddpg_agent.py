import torch
import torch.nn.functional as F
import numpy as np
import random

from models.maddpg_actor import MADDPGActor
from models.maddpg_critic import MADDPGCritic
from models.noise import Noise
from models.gaussian_noise import GaussianNoise
from replay_buffer import ReplayBuffer


class MADDPGAgent():

    def __init__(self, config, id):
        self.config = config
        self.id = id
        self.checkpoint_path_actor = "checkpoints/maddpg/cp-actor-{id:02d}-{epoch:04d}.pt"
        self.checkpoint_path_critic = "checkpoints/maddpg/cp-critic-{id:02d}-{epoch:04d}.pt"
        self.episodes = 2
        self.env_info = None
        self.env_agents = None
        self.states = None
        self.dones = None
        self.loss = None
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = self.config.config['BatchesSizeMADDPG']
        self.memory = ReplayBuffer(self.config.config['BufferSize'], self.batch_size)
        self.learn_every = 20
        self.num_learn = 20
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.actor_local = MADDPGActor(config)
        self.actor_target = MADDPGActor(config)
        self.critic_local = MADDPGCritic(config)
        self.critic_target = MADDPGCritic(config)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=float(self.config.config['LearningRateActor']))
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=float(self.config.config['LearningRateCritic']),
                                                 weight_decay=0.0
                                                 )
        self.seed = random.seed(48)
        self.noise = Noise(2, self.seed)
        # self.noise = GaussianNoise(2)
        self.scores = []
        self.scores_agent_mean = []

    def run_agent(self):
        self.run_training()

    def act(self, states, add_noise=True):
        states = torch.tensor(states, dtype=torch.float, device=self.actor_local.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experience):
        state1, state2, actions, pred_actions, rewards, dones, next_state1, next_state2, target_next_action = experience
        full_nextstates = torch.zeros((state1.shape[0], state1.shape[1]*2)).to(self.device)
        full_nextstates[..., :24] = next_state1.clone()
        full_nextstates[..., 24:] = next_state2.clone()

        full_states = torch.zeros((state1.shape[0], state1.shape[1] * 2)).to(self.device)
        full_states[..., :24] = state1.clone()
        full_states[..., 24:] = state2.clone()

        q_targets_next = self.critic_target(full_nextstates, target_next_action.to(self.device))
        q_targets = rewards[..., self.id].view(rewards.shape[0], 1) + (self.gamma * q_targets_next * (1 - dones[...,self.id].view(rewards.shape[0], 1)))
        q_expexted = self.critic_local(full_states, actions.to(self.device))
        critic_loss = F.mse_loss(q_expexted, q_targets)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.optimizer_critic.step()

        actor_loss = -self.critic_local(full_states, pred_actions.to(self.device)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.soft_update_targets()

    def soft_update_targets(self):
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update_all(self):
        self.hard_update(self.critic_local, self.critic_target)
        self.hard_update(self.actor_local, self.actor_target)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_checkpoint(self, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer_actor.state_dict()
        }, self.checkpoint_path_actor.format(id=self.id, epoch=epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.critic_target.state_dict(),
            'optimizer_state_dict': self.optimizer_critic.state_dict()
        }, self.checkpoint_path_critic.format(id=self.id, epoch=epoch))