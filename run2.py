import numpy as np
import torch

from config.config import Config
from agents.maddpg_agent import MADDPGAgent
from replay_buffer import ReplayBuffer

def main():
    config = Config()
    agent1 = MADDPGAgent(config, 0)
    agent2 = MADDPGAgent(config, 1)
    agent1.hard_update_all()
    agent2.hard_update_all()
    replay = ReplayBuffer(config.buffer_size, config.batch_size)
    print_env_information(config)
    # run_random_env(config)
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
        print(actions.shape)
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
    l_max_score = []
    for episode in range(config.num_episodes):
        print("Episonde {}/{}".format(episode, config.num_episodes))
        env_info = config.env.reset(train_mode=True)[config.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(config.num_agents)
        while True:
            action1 = agent1.act(states[0])
            action2 = agent2.act(states[1])
            actions = np.vstack((action1, action2))
            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores = np.add(scores, np.asarray(rewards))

            replay.add(states[0], states[1],
                       actions[0], actions[1],
                       rewards,
                       next_states[0], next_states[1],
                       dones)
            if (len(replay.memory) > config.batch_size): # and (episode > 300):
                for _ in range(1):
                    train_agents(agent1, agent2, replay)

            states = next_states

            if np.any(dones):
                break
        agent1.noise.reset()
        agent2.noise.reset()
        print(scores)
        l_scores.append(scores)
        l_max_score.append(scores.max())
        print("Max score this episode from 2 agents: >> {:.2f} <<".format(l_max_score[-1]))
        if (episode)%20==0:
            agent1.save_checkpoint(episode)
            agent2.save_checkpoint(episode)
            np.save(file="checkpoints/maddpg/maddpg_save_dump.npy", arr=np.asarray(l_scores))

        if (episode) >= 100:
            mean_of_max = np.mean(l_max_score[-100:])
            print("Mean of the max for the last 100 episodes: {:.2f}".format(mean_of_max))
            if mean_of_max >= 0.5:
                print("Solved the environment after {} episodes with a mean of {:.2f}".format(episode, mean_of_max))
                np.save(file="checkpoints/maddpg/maddpg_final.npy", arr=np.asarray(l_scores))
                agent1.save_checkpoint(episode)
                agent2.save_checkpoint(episode)
                break

def train_agents(agent1, agent2, replay):
    experience = replay.sample()
    state1, state2, action1, action2, rewards, next_state1, next_state2, dones = experience
    target_next_action = torch.zeros((action1.shape[0], action1.shape[1]*2))
    target_next_action[..., :2] = agent1.actor_target(next_state1)
    target_next_action[..., 2:] = agent2.actor_target(next_state2)

    pred_actions = torch.zeros((action1.shape[0], action1.shape[1]*2)).to(agent1.device)
    pred_actions[..., :2] = agent1.actor_local(state1)
    pred_actions[..., 2:] = action2

    actions = torch.zeros((action1.shape[0], action1.shape[1] * 2)).to(agent1.device)
    actions[..., :2] = action1.clone()
    actions[..., 2:] = action2.clone()

    exp = (state1, state2, actions, pred_actions, rewards, dones, next_state1, next_state2, target_next_action)
    agent1.learn2(exp)

    experience = replay.sample()
    state1, state2, action1, action2, rewards, next_state1, next_state2, dones = experience
    target_next_action = torch.zeros((action2.shape[0], action2.shape[1] * 2))
    target_next_action[..., :2] = agent1.actor_target(next_state1)
    target_next_action[..., 2:] = agent2.actor_target(next_state2)

    pred_actions = torch.zeros((action1.shape[0], action1.shape[1] * 2)).to(agent2.device)
    pred_actions[..., :2] = action1
    pred_actions[..., 2:] = agent2.actor_local(state2)

    actions = torch.zeros((action1.shape[0], action1.shape[1] * 2)).to(agent2.device)
    actions[..., :2] = action1.clone()
    actions[..., 2:] = action2.clone()

    exp = (state1, state2, actions, pred_actions, rewards, dones, next_state1, next_state2, target_next_action)
    agent2.learn2(exp)

if __name__=='__main__':
    main()