from modularl.agents import SAC, TD3, DDPG
from modularl.policies import GaussianPolicy, DeterministicPolicy
from modularl.q_functions import SAQNetwork
from modularl.replay_buffers import ReplayBuffer
import torch.optim as optim


class CreateAgent:
    def __init__(
        self,
    ):
        pass

    def sac(self, train_opt, writer):
        actor = GaussianPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
        )
        qf1 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        qf2 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(
            buffer_size=train_opt["buffer_size"],
        )
        qf_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )
        agent = SAC(
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            entropy_lr=train_opt["q_lr"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            target_network_frequency=train_opt["target_network_frequency"],
            policy_frequency=train_opt["policy_frequency"],
            target_entropy=-train_opt["action_shape"],
            writer=writer,
        )
        agent.init()
        return agent

    def td3(self, train_opt, writer):
        actor = DeterministicPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
        )
        qf1 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        qf2 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(
            buffer_size=train_opt["buffer_size"],
        )
        qf_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )
        agent = TD3(
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            exploration_noise=0,
            policy_frequency=train_opt["policy_frequency"],
            writer=writer,
        )
        agent.init()
        return agent

    def ddpg(self, train_opt, writer):
        actor = DeterministicPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
        )
        qf = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(
            buffer_size=train_opt["buffer_size"],
        )
        qf_optimizer = optim.Adam(
            list(qf.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )
        agent = DDPG(
            actor=actor,
            qf=qf,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            exploration_noise=0,
            policy_frequency=train_opt["policy_frequency"],
            writer=writer,
        )
        agent.init()
        return agent
