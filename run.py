import numpy as np

from config.config import Config
from agents.maddpg_agent import MADDPGAgent
from replay_buffer import ReplayBuffer

def main():
    config = Config()
    agent1 = MADDPGAgent(config, 1)
    agent2 = MADDPGAgent(config, 2)
    replay = ReplayBuffer(config.buffer_size, config.batch_size)
    print_env_information(config)
    run_random_env(config)
    run_training(config, agent1, agent2, replay)


def print_env_information(config):
    config.env_info = config.env.reset(train_mode=False)[config.brain_name]
    config.num_agents = len(config.env_info.agents)
    print('Number of agents:', config.num_agents)
    print('Size of each action:', config.action_dim)
    config.states = config.env_info.vector_observations
    print('There are {} agents. Each observes a state with length: {}'.format(config.states.shape[0], config.state_dim))
    print('The state for the first agent looks like:', config.states[0])


def run_random_env(config):
    env_info = config.env.reset(train_mode=False)[config.brain_name]
    states = env_info.vector_observations
    scores = np.zeros(config.num_agents)
    steps = 1
    while True:
        actions = np.random.randn(config.num_agents, config.action_dim)
        actions = np.clip(actions, -1, 1)
        print(actions)
        env_info = config.env.step(actions)[config.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            print('Scores:')
            print(scores)
            # print('Environment done after {} steps'.format(t))
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def run_training(config, agent1, agent2, replay):
    l_scores = []
    for episode in range(config.num_episodes):
        print("Episonde {}/{}".format(episode, config.num_episodes))
        env_info = config.env.reset(train_mode=True)[config.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(config.num_agents)
        while True:
            actions = np.random.randn(config.num_agents, config.action_dim)
            actions = np.clip(actions, -1, 1)
            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards

            replay.add(states[0], states[0],
                       actions[0], actions[1],
                       rewards,
                       next_states[0], next_states[1],
                       dones)
            if len(replay.memory) > config.batch_size:
                train_agents(agent1, agent2, replay)

            states = next_states

            if np.any(dones):
                break
        print(scores)
        l_scores.append(scores)
        # print("Average score from 20 agents: >> {:.2f} <<".format(scores_agent_mean[-1]))
        # if (step+1)%10==0:
        #     self.save_checkpoint(step+1)
        #     np.save(file="checkpoints/maddpg/maddpg_save_dump.npy", arr=np.asarray(self.scores))
        #
        # if (step + 1) >= 100:
        #     self.mean_of_mean = np.mean(self.scores_agent_mean[-100:])
        #     print("Mean of the last 100 episodes: {:.2f}".format(self.mean_of_mean))
        #     if self.mean_of_mean>=0.5:
        #         print("Solved the environment after {} episodes with a mean of {:.2f}".format(step, self.mean_of_mean))
        #         np.save(file="checkpoints/maddpg/maddpg_final.npy", arr=np.asarray(self.scores))
        #         self.save_checkpoint(step+1)
        #         break

def train_agents(agent1, agent2, replay):
    experience = replay.sample()
    state1, state2, action1, action2, rewards, next_state1, next_state2, dones = experience
    # agent1.train()
    # agent2.train()

if __name__=='__main__':
    main()