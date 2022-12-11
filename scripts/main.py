import sys
sys.path.insert(0, "../")
import os.path
import argparse
import gym
import numpy as np
import gym_BSS
import gym_ERSLE  # noqa: F401
from psddAgent.agent import psddAgent
from time import time
import random
from envWrappers.wrappers import MMDPObsStackWrapper, ERStoMMDPWrapper, BSStoMMDPWrapper, InfeasibleActionDetectionWrapper


def run(env, agent, no_episodes, no_exploration_episodes, mode, training_mode, no_env_steps, train_every = 2, env_name = "BSS95-760", log_name = "", \
    log_dir = "/DATA2/moritz/", run = 0, save_model_every = 1000, reward_scaling_factor = 1):
    if (mode == 'test') | (mode == 'validate'):
        agent.epsilon = 0
        if agent.param_noise_sigma > 0:
            agent.adaptive_param_noise.turn_off(agent.actor, agent.noisy)
        action_distribution = {}


    Rs = []
    step = 0
    total_reset_time = 0
    total_sampling_time = 0
    for ep in range(no_episodes):
        epstart = time()
        R = 0
        blip_reward = 0
        done = False
        obs = env.reset()
        reset_time = time()
        agent.reset()
        total_reset_time += (time() - reset_time)
        sampling_time = 0
        training_time = 0
        env_time = 0
        
        while not done:
            startsample = time()
            psdd_action, env_action, argmax_env_action, argmax_psdd_action, argmax_actionHeads = agent.getAction(obs)
            stopsample = time()
            sampling_time += (stopsample - startsample)
            if (mode == "test") | (mode == "validate"):
                if str(env_action) not in action_distribution.keys():
                    action_distribution[str(env_action)] = 0
                action_distribution[str(env_action)] += 1

            envstart = time()
            new_obs, r, done, info = env.step(env_action)
            envstop = time()
            env_time += (envstop - envstart)
            R += r
            if 'ERS' in env_name:
                blip_reward += info["blip_reward"]

            if mode == "train":
                agent.storeSample(obs, env_action, argmax_env_action, argmax_psdd_action, argmax_actionHeads, r/reward_scaling_factor, done)

                if (training_mode == "step") & (ep >= no_exploration_episodes) & (step % train_every == 0):
                    start_training = time()
                    agent.train()
                    stop_training = time()
                    training_time += (stop_training - start_training)

            obs = new_obs
            step += 1

        if mode == "train":
            agent.update_and_log_hyperparameters()
            if (training_mode == "episode") & (ep % train_every == 0) & (ep >= no_exploration_episodes):
                start_training = time()
                agent.train()
                stop_training = time()
                training_time += (stop_training - start_training)
            if (ep + 1) % save_model_every == 0:
                filename = os.path.join(log_dir+'log/'+env_name+'/'+log_name+'/run_'+ str(run), str(ep))
                agent.save_model(filename)
            obs_no = random.randrange(no_env_steps - 1)
            no_moved = sum([abs(agent.obs_list[obs_no].tolist()[agent.demand_size:-1][i] - \
                agent.obs_list[obs_no + 1].tolist()[agent.demand_size:-1][i]) for i in range(len(agent.obs_list[obs_no].tolist()[agent.demand_size:-1]))])/2
            agent.log(ep, {"Reward":R, "ActionStats/env-actions/resources_moved":no_moved})


        print({
            'episode': ep,
            'return': R,
            "blip_reward": blip_reward,
        })
        total_sampling_time += sampling_time
        Rs.append(R)
        endep = time()
        ep_total_time = endep- epstart
        print('took, rest, sampling, training, env', ep_total_time, ep_total_time - training_time - sampling_time - env_time, sampling_time, training_time, env_time)

    print('')
    print('---------------------------')
    if (mode == 'test') | (mode == 'validate'):
        if mode == "test":
            fname = 'test_log.txt'
        else:
            fname = "validate_log.txt"
        filename = os.path.join(log_dir+'log/'+env_name+'/'+log_name+'/run_'+ str(run), fname)
        with open(filename, "w") as file:
            file.write('Average reward per episode: ' + str(np.average(Rs)) + '\n')
            file.write('total time spend on policy execution: ' + str(total_reset_time + total_sampling_time) + '\n')
            for item in action_distribution.items():
                file.write(str(item) + '\n')
        print(action_distribution)
    print('Average reward per episode:', np.average(Rs))
    print('total time spend on policy execution: ', total_reset_time + total_sampling_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'PSDD Agent')

    parser.add_argument('--log_name', help = 'name under which to log the experiment', required = True)
    parser.add_argument('--log_dir', help = 'path under which to log the experiment', required = True)
    parser.add_argument('--save_model_every', help = 'save model every _ episodes', type=int, required = True)
    parser.add_argument('--seed', help = 'seed', type = int, required = True)
    parser.add_argument('--no_zones', help = 'number of zones', type = int, required = True)
    parser.add_argument('--no_resources', help = 'number of resources', type = int, required = True)
    parser.add_argument('--env_name', help='environment to run on', required = True)
    parser.add_argument('--no_train_episodes', help = 'number of training episodes per run', type = int, required = True)
    parser.add_argument('--no_test_episodes', help = 'number of test episodes per run', type = int, required = True)
    parser.add_argument('--no_exploration_episodes', help = 'number of exploration episodes per run', type = int, default = 10)
    parser.add_argument('--env_test_seed', help='environment to run on', type = int, default = 42)
    parser.add_argument('--no_stack', help = 'number of observations that are stacked together as state description', type = int, default = 3)
    parser.add_argument('--NN', \
        help = 'neural net architecture [shared_layers, actor_layers, critic_layers] where critic_layers: [state_layers, action_layers, joint_layers]; layers e.g. [32, 32]', \
        default = "[[75], [75], [[75, 75], [75], []]]")
    parser.add_argument('--epsilon', help = 'exploration rate at the start', type = float, default = 0.9995)
    parser.add_argument('--epsilon_decay', help = 'exploration rate decay rate', type = float, default = 0.9995)
    parser.add_argument('--epsilon_min', help = 'exploration rate min', type = float, default = 0.05)
    parser.add_argument('--param_noise_sigma', help = 'sigma for param noise, if 0 no param noise', type = float, default = 0)
    parser.add_argument('--M_start', help = 'number of actions that get sampled according to uniform distribution', type = int, default = 1000)
    parser.add_argument('--M_decay', help = 'decay rate of M', type = float, default = 0.099)
    parser.add_argument('--N', help = 'number of actions that get sampled according to proposal distribution', type = int, default = 10)
    parser.add_argument('--gamma', help = 'discount factor of environment', type = float, default = 1)
    parser.add_argument('--coeff_entrop', help = 'weight of entropy added to proposal network loss', type = float, default = 0.01)
    parser.add_argument('--lr_actor', help = 'learning rate actor', type = float, default = 0.0001)
    parser.add_argument('--lr_critic', help = 'learning rate critic', type = float, default = 0.0005)
    parser.add_argument('--buffer_size', help = 'buffer size, default is no buffer', type = int, default = 0)
    parser.add_argument('--mode', help = 'training and buffer (if exists) can be "episode"- or "step"-based', default = "episode")
    parser.add_argument('--buffer_in_order', help = 'whether to replay experiences from the buffer in the order they were encountered', action = 'store_true')
    parser.add_argument('--per_buffer', help = 'whether to use a prioritized experience replay or not', action = 'store_true')
    parser.add_argument('--per_buffer_beta', help = 'beta parameter of prioritized experience replay', type = float, default = 1)
    parser.add_argument('--per_buffer_alpha', help = 'alpha parameter of prioritized experience replay', type = float, default = 0.6)
    parser.add_argument('--direct_training', help = 'whether to train directly from last episode (only for episode mode)', action = 'store_true')
    parser.add_argument('--replay_training', help = 'whether to train from replay', action = 'store_true')
    parser.add_argument('--train_every', help = 'train every train_every episodes/steps', type = int, default = 1)
    parser.add_argument('--batch_size', help = 'batch size of training batch sampled from replay buffer if buffer exists', type = int, default = 12)
    parser.add_argument('--replay_ratio', help = 'number of batches sampled from replay buffer per training', type = int, default = 0)
    parser.add_argument('--action_normalization', help = 'whether to normalize actions (3 options', default = 'unnormalized')
    parser.add_argument('--state_normalization', help = 'how to normalize states (3 options)', default = "unnormalized")
    parser.add_argument('--layer_normalization', help = 'whether use layer normalization', action = 'store_true')
    parser.add_argument('--target_net_update_every', help = 'target network update frequency', type = int, default = 0)
    parser.add_argument('--tau', help = 'updates target network with tau percent of critic', type = float, default = 0)
    parser.add_argument('--uniform_action_sampling', help = 'number of presampled lists of uniform actions', type = int, default = 1)
    parser.add_argument('--pooling', help = 'number of resources pooled together', type = int, default = 1)
    parser.add_argument('--device', help = 'run on cpu or cuda', default = "cpu")
    parser.add_argument('--no_threads', help = 'number of threads used for training if device is cpu', type = int, default = 1)
    parser.add_argument('--reward_scaling_factor', help = 'factor by which to divide reward', type = int, default = 1)
    parser.add_argument('--no_presampled_actions_per_file', help = 'to balance memory and speed', type = int, default = 10000)
    parser.add_argument('--load_model_path', help = 'path to load a preexisting model from', default='no model loading')

    args = parser.parse_args()
    net_arch = eval(args.NN)

    starttotal = time()

    env = gym.make(args.env_name)  # gym.Env
    if 'BSS' in args.env_name:
        env = BSStoMMDPWrapper(env)
        no_env_steps = 12
    elif 'ERS' in args.env_name:
        env = ERStoMMDPWrapper(env)
        no_env_steps = 48
    else:
        NotImplementedError('this env is not supported')
    env = InfeasibleActionDetectionWrapper(env)
    env = MMDPObsStackWrapper(env, args.no_stack)

    agent = psddAgent(state_size=env.observation_space.shape[0], action_size=args.no_zones, env_name = args.env_name, timesteps = no_env_steps, \
        log_name = args.log_name, run = args.seed, log_dir = args.log_dir, capacity_limits = env.action_space.high, nresources = args.no_resources, \
        constraints = env.metadata['constraints'], \
        N = args.N, M_start = args.M_start, M_decay = args.M_decay, lr_actor = args.lr_actor, lr_critic = args.lr_critic, epsilon = args.epsilon, \
        epsilon_decay = args.epsilon_decay, epsilon_min = args.epsilon_min, param_noise_sigma = args.param_noise_sigma, gamma = args.gamma, net_arch = net_arch, \
        coeff_entrop = args.coeff_entrop, batch_size = args.batch_size, direct_training = args.direct_training, replay_training = args.replay_training, \
        replay_ratio = args.replay_ratio, action_normalization = args.action_normalization, state_normalization = args.state_normalization, \
        layer_normalization = args.layer_normalization, buffer_size = args.buffer_size, per_buffer = args.per_buffer, per_buffer_beta = args.per_buffer_beta, \
        per_buffer_alpha = args.per_buffer_alpha, mode = args.mode, buffer_in_order = args.buffer_in_order, target_net_update_every = args.target_net_update_every, \
        tau = args.tau, uniform_action_sampling = args.uniform_action_sampling, no_presampled_actions_per_file = args.no_presampled_actions_per_file, \
        pooling = args.pooling, device = args.device, no_threads = args.no_threads, total_training_episodes = args.no_train_episodes)

    if args.load_model_path != 'no model loading':
        agent.load_model(args.load_model_path)

    env.seed(args.seed)
    run(env, agent, args.no_train_episodes, args.no_exploration_episodes, "train", training_mode = args.mode, no_env_steps = no_env_steps, \
        train_every = args.train_every, log_name = args.log_name, log_dir = args.log_dir, run = args.seed, env_name = args.env_name, \
        save_model_every = args.save_model_every, reward_scaling_factor = args.reward_scaling_factor)
    env.seed(args.env_test_seed)
    run(env, agent, args.no_test_episodes, 0, "test", training_mode = args.mode, no_env_steps = no_env_steps, log_name = args.log_name, \
        log_dir = args.log_dir, run = args.seed, env_name = args.env_name, \
        save_model_every = args.save_model_every, reward_scaling_factor = args.reward_scaling_factor)

    """
    # tests:
    log_name = "testERS"
    i = 0

    env = gym.make('SgERSEnv-ca-dynamic-cap6-30-v6')  # gym.Env
    env = MMDPObsStackWrapper(env, 3)
    agent = psddAgent(state_size=env.observation_space.shape[0], action_size=env.metadata['nbases'], env_name = "SgERSEnv-Poisson", \
        log_name = log_name + str(i), capacity_limits = np.asarray([6] * 25), M_start = 1000, M_decay = 0.099, lr_actor = 1e-4, \
        lr_critic = 1e-3, state_normalization = "average", target_net_update_every = 1, batch_size = 64, \
        action_normalization = False, layer_norm = True, hidden_size = 175, buffer_size = 2**7, buffer_mode = "step", \
        direct_training_batch = 0, replay_ratio = 1, epsilon = 0.9995, epsilon_decay = 0.9995,
        pooling = 1, tau = 0.2, per_buffer = True, device = 'cpu')
    env.seed(i)
    run(env, agent, 10, "train", log_name = log_name + str(i), env_name = "SgERSEnv-Poisson", save_model_every = 10)
    env.seed(42)
    run(env, agent, 10, "test", log_name = log_name + str(i), env_name = "SgERSEnv-Poisson")
    """
    print((time() - starttotal) /60)