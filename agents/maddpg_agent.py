import torch
import torch.nn.functional as F
import numpy as np
import random

from models.maddpg_actor import MADDPGActor
from models.maddpg_critic import MADDPGCritic
from models.noise import Noise
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
        self.gamma = 0.99
        self.tau = 0.001
        self.batch_size = self.config.config['BatchesSizeMADDPG']
        self.memory = ReplayBuffer(100000, self.batch_size)
        self.learn_every = 20
        self.num_learn = 20
        self.actor_local = MADDPGActor(config)
        self.actor_target = MADDPGActor(config)
        self.critic_local = MADDPGCritic(config)
        self.critic_target = MADDPGCritic(config)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=float(self.config.config['LearningRateActor']))
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=float(self.config.config['LearningRateCritic']),
                                                 # weight_decay=0.0000
                                                 )
        self.seed = random.seed(16)
        self.noise = Noise(2, self.seed)
        self.scores = []
        self.scores_agent_mean = []

    def run_agent(self):
        self.run_training()

    def train(self):
        pass

    def act(self, states, add_noise=True):
        states = torch.tensor(states, dtype=torch.float, device=self.actor_local.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self):
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save_checkpoint(self, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer_actor.state_dict()
        }, self.checkpoint_path_actor.format(epoch=epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.critic_target.state_dict(),
            'optimizer_state_dict': self.optimizer_critic.state_dict()
        }, self.checkpoint_path_critic.format(epoch=epoch))