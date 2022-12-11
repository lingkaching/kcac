import math
import os
import numpy as np
import torch as tc
import torch.nn as nn
from torch.distributions import categorical
from tensorboardX import SummaryWriter
import gc
import pickle
from time import time, sleep
import shutil
import random



class actor(nn.Module):
    def __init__(self, nodes, num_decom_nodes, seed, state_size, net_arch, layer_norm, device, no_threads):
        super(actor, self).__init__()
        if device == tc.device("cpu"):
            tc.set_num_threads(no_threads)
        self.num_decom_nodes = num_decom_nodes
        self.layer_norm = layer_norm

        self.fc_joint = nn.ModuleList([])
        self.relu_joint = nn.ModuleList([])
        self.norm_joint = nn.ModuleList([])
        self.fc = nn.ModuleList([])
        self.relu = nn.ModuleList([])
        self.norm = nn.ModuleList([])

        # joint layers with critic
        if len(net_arch[0]) > 0:
            self.fc_joint.append(nn.Linear(state_size, net_arch[0][0], bias=True))
            self.relu_joint.append(nn.ReLU())
            if self.layer_norm:
                self.norm_joint.append(nn.LayerNorm(net_arch[0][0]))
            for layer in range(len(net_arch[0][1:])):
                self.fc_joint.append(nn.Linear(net_arch[0][layer], net_arch[0][layer + 1], bias=True))
                self.relu_joint.append(nn.ReLU())
                if self.layer_norm:
                    self.norm_joint.append(nn.LayerNorm(net_arch[0][layer + 1]))

        # actor layers
        if len(net_arch[1]) > 0:
            if len(net_arch[0]) == 0:
                self.fc.append(nn.Linear(state_size, net_arch[1][0], bias=True))
            else:
                self.fc.append(nn.Linear(net_arch[0][-1], net_arch[1][0], bias=True))
            self.relu.append(nn.ReLU())
            if self.layer_norm:
                self.norm.append(nn.LayerNorm(net_arch[1][0]))
            for layer in range(len(net_arch[1][1:])):
                self.fc.append(nn.Linear(net_arch[1][layer], net_arch[1][layer + 1], bias=True))
                if self.layer_norm:
                    self.norm.append(nn.LayerNorm(net_arch[1][layer + 1]))
                self.relu.append(nn.ReLU())
            self.heads = nn.Linear(net_arch[1][-1], self.num_decom_nodes * 2)
        elif len(net_arch[0]) > 0:
            self.heads = nn.Linear(net_arch[0][-1], self.num_decom_nodes * 2)
        else:
            self.heads = nn.Linear(state_size, self.num_decom_nodes * 2)

    def forward(self, x):

        #common features
        for layer in range(len(self.fc_joint)):
            x = self.fc_joint[layer](x)
            if self.layer_norm:
                x = self.norm_joint[layer](x)
            x = self.relu_joint[layer](x)

        # actor-specific layers
        for layer in range(len(self.fc)):
            x = self.fc[layer](x)
            if self.layer_norm:
                x = self.norm[layer](x)
            x = self.relu[layer](x)        

        output = self.heads(x).reshape(-1, self.num_decom_nodes, 2).softmax(dim = -1) # OBDD VERSION 2/4
        return output
    

class critic(nn.Module):
    def __init__(self, seed, state_size, action_size, net_arch, layer_norm, device, no_threads):
        super(critic, self).__init__()
        if device == tc.device("cpu"):
            tc.set_num_threads(no_threads)
        self.layer_norm = layer_norm

        self.fc_joint = nn.ModuleList([])
        self.relu_joint = nn.ModuleList([])
        self.norm_joint = nn.ModuleList([])

        # joint layers with actor (only for state input)
        if len(net_arch[0]) > 0:
            self.fc_joint.append(nn.Linear(state_size, net_arch[0][0], bias=True))
            self.relu_joint.append(nn.ReLU())
            if self.layer_norm:
                self.norm_joint.append(nn.LayerNorm(net_arch[0][0]))
            for layer in range(len(net_arch[0][1:])):
                self.fc_joint.append(nn.Linear(net_arch[0][layer], net_arch[0][layer + 1], bias=True))
                self.relu_joint.append(nn.ReLU())
                if self.layer_norm:
                    self.norm_joint.append(nn.LayerNorm(net_arch[0][layer + 1]))

        # critic layers - state
        self.fc_state = nn.ModuleList([])
        self.relu_state = nn.ModuleList([])
        self.norm_state = nn.ModuleList([])

        if len(net_arch[2][0]) > 0:
            if len(net_arch[0]) == 0:
                self.fc_state.append(nn.Linear(state_size, net_arch[2][0][0], bias=True))
            else:
                self.fc_state.append(nn.Linear(net_arch[0][-1], net_arch[2][0][0], bias=True))
            self.relu_state.append(nn.ReLU())
            if self.layer_norm:
                self.norm_state.append(nn.LayerNorm(net_arch[2][0][0]))

            for layer in range(len(net_arch[2][0][1:])):
                self.fc_state.append(nn.Linear(net_arch[2][0][layer], net_arch[2][0][layer + 1], bias=True))
                if self.layer_norm:
                    self.norm_state.append(nn.LayerNorm(net_arch[2][0][layer + 1]))
                self.relu_state.append(nn.ReLU())

        # critic layers - action
        self.fc_action = nn.ModuleList([])
        self.relu_action = nn.ModuleList([])
        self.norm_action = nn.ModuleList([])

        if len(net_arch[2][1]) > 0:
            self.fc_action.append(nn.Linear(action_size, net_arch[2][1][0], bias=True))
            self.relu_action.append(nn.ReLU())
            if self.layer_norm:
                self.norm_action.append(nn.LayerNorm(net_arch[2][1][0]))

            for layer in range(len(net_arch[2][1][1:])):
                self.fc_action.append(nn.Linear(net_arch[2][1][layer], net_arch[2][1][layer + 1], bias=True))
                if self.layer_norm:
                    self.norm_action.append(nn.LayerNorm(net_arch[2][1][layer + 1]))
                self.relu_action.append(nn.ReLU())

        # critic layers - concateneated layers
        self.fc = nn.ModuleList([])
        self.relu = nn.ModuleList([])
        self.norm = nn.ModuleList([])

        state_side_size = state_size
        state_raw_input = True
        if len(net_arch[2][0]) > 0:
            state_side_size = net_arch[2][0][-1]
            state_raw_input = False
        elif len(net_arch[0]) > 0:
            state_side_size = net_arch[0][-1]
            state_raw_input = False

        action_side_size = action_size
        action_raw_input = True
        if len(net_arch[2][1]) > 0:
            action_side_size = net_arch[2][1][-1]
            action_raw_input = False

        if len(net_arch[2][2]) > 0:
            self.fc.append(nn.Linear(state_side_size + action_side_size, net_arch[2][2][0], bias=True))
            self.relu.append(nn.ReLU())
            if self.layer_norm:
                self.norm.append(nn.LayerNorm(net_arch[2][2][0]))
            for layer in range(len(net_arch[2][2][1:])):
                self.fc.append(nn.Linear(net_arch[2][2][layer], net_arch[2][2][layer + 1], bias=True))
                self.relu.append(nn.ReLU())
                if self.layer_norm:
                    self.norm.append(nn.LayerNorm(net_arch[2][2][layer + 1]))
            self.head = nn.Linear(net_arch[2][2][-1], 1)
        else:
            self.head = nn.Linear(action_side_size + state_side_size, 1)


    def forward(self, x_state, x_action):

        # state side:
        for layer in range(len(self.fc_joint)):
            x_state = self.fc_joint[layer](x_state)
            if self.layer_norm:
                x_state = self.norm_joint[layer](x_state)
            x_state = self.relu_joint[layer](x_state)

        for layer in range(len(self.fc_state)):
            x_state = self.fc_state[layer](x_state)
            if self.layer_norm:
                x_state = self.norm_state[layer](x_state)
            x_state = self.relu_state[layer](x_state)

        # action side:
        for layer in range(len(self.fc_action)):
            x_action = self.fc_action[layer](x_action)
            if self.layer_norm:
                x_action = self.norm_action[layer](x_action)
            x_action = self.relu_action[layer](x_action)

        # concatenation:
        x = tc.cat((x_state, x_action), dim=1)

        # final layers
        for layer in range(len(self.fc)):
            x = self.fc[layer](x)
            if self.layer_norm:
                x = self.norm[layer](x)
            x = self.relu[layer](x)  

        output = self.head(x)  
        
        return output


class RunningStats:
    def __init__(self, size):
        self.size = size
        self.avg = np.zeros(size)
        self.var = np.zeros(size)
        self.std = np.full(size, 1e-1)
        self.min = np.full(size, 1e-2)
        self.num_step = 0

    def update(self, sample):
        self.num_step += 1
        avg_old = self.avg
        self.avg = self.avg + (sample - self.avg) / self.num_step
        self.var = self.var + (sample - avg_old) * (sample - self.avg)
        self.var = np.maximum(self.var, self.min)
        if self.num_step > 1:
            self.std = np.sqrt(self.var / (self.num_step - 1))     


class UniformEnvActionsBuffer:
    def __init__(self, fileName):
        self.fileName = fileName
        self.load_buffer()

    def load_buffer(self):
        file = open(self.fileName, "rb")
        self.buffer = pickle.load(file)
        file.close()
        self.buffer_size = len(self.buffer)


    def get_sample_indices(self, sample_size):
        return random.sample(range(self.buffer_size), sample_size)


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_in_order):
        self.buffer_size = buffer_size
        self.buffer = []
        self.size = 0
        self.buffer_in_order = buffer_in_order

    def add(self, sample):
        self.buffer.append(sample)
        self.size += 1
        if self.size > self.buffer_size:
            self.buffer.pop(0)
            self.size -= 1           

    def sample(self, sample_size):
        if self.buffer_in_order == True:
            samples = self.buffer[:sample_size]
        else:
            samples = random.choices(self.buffer, k = sample_size)
        return samples


class PEReplayBuffer:
    # from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py
    """
    ## Buffer for Prioritized Experience Replay
    [Prioritized experience replay](https://papers.labml.ai/paper/1511.05952)
     samples important transitions more frequently.
    The transitions are prioritized by the Temporal Difference error (td error), $/delta$.
    We sample transition $i$ with probability,
    $$P(i) = /frac{p_i^/alpha}{/sum_k p_k^/alpha}$$
    where $/alpha$ is a hyper-parameter that determines how much
    prioritization is used, with $/alpha = 0$ corresponding to uniform case.
    $p_i$ is the priority.
    We use proportional prioritization $p_i = |/delta_i| + /epsilon$ where
    $/delta_i$ is the temporal difference for transition $i$.
    We correct the bias introduced by prioritized replay using
     importance-sampling (IS) weights
    $$w_i = \bigg(\frac{1}{N} /frac{1}{P(i)}/bigg)^/beta$$ in the loss function.
    This fully compensates when $/beta = 1$.
    We normalize weights by $/frac{1}{/max_i w_i}$ for stability.
    Unbiased nature is most important towards the convergence at end of training.
    Therefore we increase $/beta$ towards end of training.
    ### Binary Segment Tree
    We use a binary segment tree to efficiently calculate
    $/sum_k^i p_k^/alpha$, the cumulative probability,
    which is needed to sample.
    We also use a binary segment tree to find $/min p_i^/alpha$,
    which is needed for $/frac{1}{/max_i w_i}$.
    We can also use a min-heap for this.
    Binary Segment Tree lets us calculate these in $/mathcal{O}(/log n)$
    time, which is way more efficient that the naive $/mathcal{O}(n)$
    approach.
    This is how a binary segment tree works for sum;
    it is similar for minimum.
    Let $x_i$ be the list of $N$ values we want to represent.
    Let $b_{i,j}$ be the $j^{/mathop{th}}$ node of the $i^{/mathop{th}}$ row
     in the binary tree.
    That is two children of node $b_{i,j}$ are $b_{i+1,2j}$ and $b_{i+1,2j + 1}$.
    The leaf nodes on row $D = /left/lceil {1 + /log_2 N} /right/rceil$
     will have values of $x$.
    Every node keeps the sum of the two child nodes.
    That is, the root node keeps the sum of the entire array of values.
    The left and right children of the root node keep
     the sum of the first half of the array and
     the sum of the second half of the array, respectively.
    And so on...
    $$b_{i,j} = /sum_{k = (j -1) * 2^{D - i} + 1}^{j * 2^{D - i}} x_k$$
    Number of nodes in row $i$,
    $$N_i = /left/lceil{/frac{N}{D - i + 1}} /right/rceil$$
    This is equal to the sum of nodes in all rows above $i$.
    So we can use a single array $a$ to store the tree, where,
    $$b_{i,j} /rightarrow a_{N_i + j}$$
    Then child nodes of $a_i$ are $a_{2i}$ and $a_{2i + 1}$.
    That is,
    $$a_i = a_{2i} + a_{2i + 1}$$
    This way of maintaining binary trees is very easy to program.
    *Note that we are indexing starting from 1*.
    We use the same structure to compute the minimum.
    """

    def __init__(self, capacity, alpha):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.buffer = [None for i in range(self.capacity)]
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add(self, data):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.buffer[idx] = data

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
        #### $/sum_k p_k^/alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $/min_k p_k^/alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $/sum_{k=1}^{i} p_k^/alpha  /le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        # Initialize weights
        self.weights = []
        self.indexes = []


        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            self.indexes.append(idx)

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = self.indexes[i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            self.weights.append(weight / max_weight)

        return [self.buffer[i] for i in self.indexes]

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size


#adapted from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoise():
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def _get_distance(self, states, psddAgent):
        if psddAgent.mode == 'step':
            steps = np.asarray(states, dtype = object)
            inputs_states = tc.tensor(np.stack(steps[:, 0]), device=psddAgent.device, dtype = tc.float)
        elif psddAgent.mode == 'episode':
            obs = [ep[0] for ep in states]
            inputs_states = tc.tensor(np.stack(obs), device=psddAgent.device, dtype = tc.float)

        #normalisation
        if psddAgent.state_normalization != "unnormalized":
            inputs_states = psddAgent._normalize_states(inputs_states)
        
        policy_original = psddAgent.actor(inputs_states)
        policy_disturbed = psddAgent.noisy(inputs_states)

        distance = tc.mean((policy_disturbed - policy_original) ** 2 / 2)

        return float(distance)

    def turn_off(self, actor, noisy):
        for key, value in noisy.state_dict().items():
            value.copy_(actor.state_dict()[key].data)

    def _add_noise_and_update_noisy(self, actor, noisy):
        for key, value in noisy.state_dict().items():
            value.copy_(actor.state_dict()[key].data)
            if "fc" in key:
                noise = self.current_stddev * np.random.standard_normal(size=np.shape(value))
                value.copy_(actor.state_dict()[key].data + noise)


    def reset_noise(self, psddAgent, batch_size):
        if psddAgent.per_buffer == True:
            states = psddAgent.buffer.sample(psddAgent.batch_size, psddAgent.per_buffer_beta)
        else:
            states = psddAgent.buffer.sample(psddAgent.batch_size)
        self._add_noise_and_update_noisy(psddAgent.actor, psddAgent.noisy)
        distance = self._get_distance(states, psddAgent)
        self.adapt(distance)
        

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class psddAgent:

    def __init__(self, psdd_file, zones2vars_file, presampledFileNameBase, capacity_limits, state_size = 2*95+1, action_size = 95, seed=0, env_name = "BSS95-760", timesteps = 12, constraints = {}, \
        net_arch = [[32],[32],[[32],[32],[32]]], epsilon = 0.99, epsilon_decay = 0.99, epsilon_min = 0.1, M_start = 10, N = 10, M_decay = 0.01, \
        param_noise_sigma = 0, gamma = 0.99, coeff_entrop = 0.01, direct_training = True, lr_actor = 0.001, lr_critic = 0.01, log_name = "", run = 0, \
        log_dir = "/DATA2/moritz/", nresources = 760, action_normalization = "unnormalized", state_normalization = "unnormalized", buffer_size = 0, \
        per_buffer = False, per_buffer_beta = 1, total_training_episodes = 10_000, per_buffer_alpha = 0.6, \
        mode = "episode", buffer_in_order = False, batch_size = 12, replay_ratio = 0, \
        replay_training = False, layer_normalization = False, target_net_update_every = 0, tau = 0, uniform_action_sampling = 1, \
        pooling = 1, device = 'cuda', no_threads = 4):
        """

        state_normalization:
        options are "unnormalized" (no normalisation), "average" (state is normalised by average and 
        standard deviation seen from start of run until current step), "mixed" (state variables that are representing 
        allocation are divided by the maximal capacities of their respective stations, the variable that acts as a step 
        identifier is divided by 12, as each episode has 12 steps, and the state variables that are representing demand are 
        normalised using the "average" method)

        buffer_size:
        if 0: no buffer. if positive integer: buffer size in episodes or samples (see mode)

        per_buffer:
        whether prioritized experience replay buffer is used or not! if true, then buffer size must be a power of 2

        per_buffer_beta:
        admissible values [0, 1]; is increases linearly from the value given to 1 at the end of the training

        mode:
        "step": elements sampled from the buffer are of the size of (batch_size) steps - cannot be combined with direct 
        training; "episode": elements sampled from the buffer are episodes. 

        buffer_in_order:
        draws the newest episodes from the buffer instead of sampling them randomly;

        replay_ratio:
        nonnegative integer specifying the number of batches drawn from the replay buffer for training per episode

        batch_size:
        number of samples drawn from replay buffer per batch

        layer_normalization:
        turn off/on layer normalisation for both actor and critic

        nresources:
        used for logging purposes only

        target_net_update_every:
        target network gets updated every target_net_update_every episodes; if set to 0, no target network

        tau:
        parameter for polyak averaging; if 0 no polyak averaging

        uniform_action_sampling:
        the algorithm uses presampled lists of uniform environment actions to speed up sampling uniform actions as compared to
        sampling them from a PSDD. uniform action sampling specifies how many list files are prepared and goes through all of 
        them once during training

        pooling:
        number of bikes pooled together as one resource for the psdd
        """
        if pooling > 1:
            NotImplementedError('Pooling might currently not be correctly implemented, check and test the code before removing this assertion')
        tc.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            self.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
        else:
            self.device = tc.device('cpu')
        self.log_dir = log_dir
        psddFileName = psdd_file
        self.zones2vars =  self.readZoneVar(zones2vars_file)
        self.timesteps = timesteps
        self.capacity_limits = capacity_limits
        self.presampledFileNameBase = presampledFileNameBase
        self.nodes, self.num_decom_nodes, self.node_id2pos, self.sample_layer_uniform, self.root_node_id = self.readPSDD(psddFileName)
        self.uniform_action_sampling = uniform_action_sampling
        self.pooling = pooling
        self.actor = actor(self.nodes, self.num_decom_nodes, seed, state_size, net_arch, layer_normalization, self.device, no_threads)
        self.actor.to(self.device)
        self.param_noise_sigma = param_noise_sigma
        self.critic = critic(seed, state_size, action_size, net_arch, layer_normalization, self.device, no_threads)
        self.target_net_update_every = target_net_update_every
        self.tau = tau
        if (self.target_net_update_every > 0) | (self.tau > 0):
            self.target = critic(seed, state_size, action_size, net_arch, layer_normalization, self.device, no_threads)
            self.target.to(self.device)
        self.critic.to(self.device)
        if self.param_noise_sigma > 0:
            assert (self.target_net_update_every > 0) | (self.tau > 0), "No adaptive parameter noise without replay buffer!"
            self.noisy = actor(self.nodes, self.num_decom_nodes, seed, state_size, net_arch, layer_normalization, self.device, no_threads)
            self.adaptive_param_noise = AdaptiveParamNoise(initial_stddev = self.param_noise_sigma, desired_action_stddev = self.param_noise_sigma)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_episode = 0
        self.num_step = 0
        self.total_training_episodes = total_training_episodes
        self.M_start, self.N = M_start, N
        self.M = self.M_start
        self.M_decay = M_decay
        self.gamma = gamma
        self.coeff_entrop = coeff_entrop
        self.direct_training = direct_training
        self.action_normalization = action_normalization
        self.state_normalization = state_normalization
        self.action_size = action_size
        self.demand_size = state_size - action_size -1
        if self.state_normalization == "average":
            self.state_running_stats = RunningStats(state_size)
        elif self.state_normalization == "mixed":
            self.state_running_stats = RunningStats(self.demand_size)
        if self.action_normalization == "average":
            self.action_running_stats = RunningStats(action_size)
        self.per_buffer = per_buffer
        self.beta = per_buffer_beta
        self.beta_start = self.beta
        self.mode = mode
        if buffer_size > 0:
            if per_buffer == True:
                assert math.log(buffer_size, 2) % 1 == 0, "PERBuffer must have a size of 2^i"
                self.buffer = PEReplayBuffer(buffer_size, per_buffer_alpha)
            else:
                self.buffer = ReplayBuffer(buffer_size, buffer_in_order)
            if (self.mode == "step"):
                assert self.direct_training == False, "When mode is set to 'step', no direct training is possible!"
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.replay_training = replay_training
        if uniform_action_sampling > 1:
            self.presampledFileName = self.presampledFileNameBase + str(0) + '.pkl'
        self.UEABuffer = UniformEnvActionsBuffer(self.presampledFileName)
        self.nresources = nresources

        #the first layer of actor network is frozen
        for layer in self.actor.fc_joint:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        for layer in self.actor.norm_joint:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

        self.optimizer_actor = tc.optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()), lr=lr_actor)
        self.optimizer_critic = tc.optim.Adam(self.critic.parameters(), lr=lr_critic)

        #log data
        writerSavePath = self.log_dir + "log/" + env_name + '/' + log_name + '/run_' + str(run)
        while os.path.isdir(writerSavePath):
            writerSavePath += "1"
        os.system('mkdir ' + writerSavePath)
        self.writer = SummaryWriter(logdir = writerSavePath)
        self.successful_proposal_cnt = 0
        self.head_trained_stat = np.zeros(self.num_decom_nodes)

    def _normalize_with_running_stats(self, vector2be_normalised, running_stats):
        """expects a tensor of demand as input"""
        normalised_vector = (vector2be_normalised - tc.tensor(running_stats.avg, device=self.device, dtype=tc.float)) / \
        tc.tensor(running_stats.std, device=self.device, dtype=tc.float)
        return normalised_vector

    def _normalize_actions(self, actions):
        """expects a tensor of states as input"""
        if self.action_normalization == "average":
            actions = self._normalize_with_running_stats(actions, self.action_running_stats)
        elif self.action_normalization == "static":
            actions = actions / tc.tensor(self.capacity_limits, device=self.device, dtype=tc.float)
        else:
            NotImplementedError("available normalization for actions: unnormalized, average, static")
        return actions

    def _normalize_states(self, states):
        """expects a tensor of demand as input"""
        if self.state_normalization == "average":
            states = self._normalize_with_running_stats(states, self.state_running_stats)
        elif self.state_normalization == "mixed":
            #disecting the state into its 3 components and normalising them:
            demand = states[:, :self.demand_size]
            allocation = states[:, self.demand_size:-1]
            episode_number = states[:, -1:]

            #normalising them separately:
            allocation = self._normalize_actions(allocation)
            demand = self._normalize_with_running_stats(demand, self.state_running_stats)
            episode_number = episode_number/tc.tensor(self.timesteps, device=self.device, dtype=tc.float)

            #putting them together again:
            states = tc.cat((demand, allocation, episode_number), 1)
        else:
            raise NotImplementedError("Please choose a valid state normalization option: 'unnormalized', 'average', or \
                'mixed'.")
        return states

    def copyFirstLayer(self):
        for layer in range(len(self.actor.fc_joint)):
            self.actor.state_dict()['fc_joint.' + str(layer) + '.weight'].copy_(self.critic.state_dict()['fc_joint.' + str(layer) + '.weight'].data)
            self.actor.state_dict()['fc_joint.' + str(layer) + '.bias'].copy_(self.critic.state_dict()['fc_joint.' + str(layer) + '.bias'].data)
        for layer in range(len(self.actor.norm_joint)):
            self.actor.state_dict()['norm_joint.' + str(layer) + '.weight'].copy_(self.critic.state_dict()['norm_joint.' + str(layer) + '.weight'].data)
            self.actor.state_dict()['norm_joint.' + str(layer) + '.bias'].copy_(self.critic.state_dict()['norm_joint.' + str(layer) + '.bias'].data)


    def readZoneVar(self, zones2varsFileName):
        zones2varsFile = open(zones2varsFileName, "rb")
        zones2vars=pickle.load(zones2varsFile)
        zones2varsFile.close()

        return zones2vars


    def readPSDD(self, psddFileName):
        # does this work for PSDDs with False nodes? or with True nodes (not really True nodes in PSDD) representing multiple models?
        #read psdd nodes
        cnt = 0
        f = open(psddFileName,'r')
        nodes = {}
        node_id2pos = {}
        sizes = []
        uniform_probs = []
        for line in f:
            if line.startswith('c'): continue
            elif line.startswith('psdd'): continue
            elif line.startswith('F'): continue# no FALSE node
            elif line.startswith('T'): continue
            elif line.startswith('L'):
                node_id, vtree_id, lit = [ int(x) for x in line[2:].split() ]
                nodes[node_id] = ['L', vtree_id, lit]
            elif line.startswith('D'):
                line = line[2:].split()
                node_id,vtree_id,size = [ int(x) for x in line[:3] ]
                if size != 1: 
                    node_id2pos[node_id] = cnt
                    cnt += 1
                    sizes.append(size)
                    thetas = []
                nodes[node_id] = ['D', vtree_id, size, []]
                line_iter = iter(line[3:])
                for i in range(size):
                    p = int(next(line_iter))
                    s = int(next(line_iter))
                    theta = np.exp(float(next(line_iter)))
                    if size != 1:
                        thetas.append(theta)
                    nodes[node_id][3].append([p,s, theta])
                if size != 1:
                    uniform_probs.append(np.asarray(thetas)/sum(np.asarray(thetas)))

        uniform_probs = tc.tensor(np.asarray(uniform_probs), device = self.device)
        sample_layer_uniform = categorical.Categorical(uniform_probs)
               
        f.close()
        return nodes, cnt, node_id2pos, sample_layer_uniform, node_id

    def reset(self):
        self.obs_list = []
        self.action_list = []
        self.action_star_list = []
        self.psdd_action_star_list = []
        self.argmax_actionHeads_list = []
        self.reward_list = []
        self.done_list = []
        self.copyFirstLayer()

    def storeSample(self, obs, env_action, argmax_env_action, argmax_psdd_action, argmax_actionHeads, r, done):
        self.num_step += 1
        self.obs_list.append(obs)
        self.action_list.append(env_action)
        self.action_star_list.append(argmax_env_action)
        self.psdd_action_star_list.append(argmax_psdd_action)
        self.argmax_actionHeads_list.append(argmax_actionHeads)
        self.reward_list.append(r)
        if done:
            self.done_list.append(0)
        else:
            self.done_list.append(1)

        # updating stats for normalising
        if self.state_normalization == "mixed":
            self.state_running_stats.update(obs[:self.demand_size])
        elif self.state_normalization == "average":
            self.state_running_stats.update(obs)
        if self.action_normalization == "average":
            self.action_running_stats.update(env_action)

        # storing sample in buffer:
        if (self.buffer_size > 0) & (done):
            if self.mode == "episode":
                self.buffer.add([self.obs_list, self.action_list, self.action_star_list, self.psdd_action_star_list, \
                    self.argmax_actionHeads_list, self.reward_list, self.done_list])
            elif self.mode == "step": #NOT USING THE LAST STEP HERE -> MAKE CIRCULAR?
                for i in range(self.timesteps - 1):
                    self.buffer.add([self.obs_list[i], self.action_list[i], self.obs_list[i + 1], \
                        self.action_star_list[i + 1], self.reward_list[i], self.psdd_action_star_list[i], \
                        self.argmax_actionHeads_list[i], self.done_list[i]])
                self.buffer.add([self.obs_list[-1], self.action_list[-1], self.obs_list[-1], \
                    self.action_star_list[-1], self.reward_list[-1], self.psdd_action_star_list[-1], \
                    self.argmax_actionHeads_list[-1], self.done_list[-1]])
            else:
                raise ValueError("mode has to be set to 'episode' or 'step'")

    def _episode_based_training(self, data, target_net):

        obs = np.concatenate(np.asarray([ep[0] for ep in data]))
        act = np.concatenate(np.asarray([ep[1] for ep in data]))
        act_star = np.concatenate(np.asarray([ep[2] for ep in data]))
        psdd_act_star = np.concatenate(np.asarray([ep[3] for ep in data]))
        argmax_actionHeads = [ep[4] for ep in data]
        rewards = np.concatenate(np.asarray([ep[5] for ep in data]))
        dones = np.concatenate(np.asarray([ep[6] for ep in data]))
        obs = np.vstack([obs, obs[0]])
        act_star = np.vstack([act_star, act_star[0]])

        if (self.per_buffer) & (self.direct_training == False):
            new_priorities = []
            loss_theta_list = []
            
        #q-values prediction
        inputs_states = tc.tensor(obs[:-1], device=self.device, dtype = tc.float)
        inputs_actions = tc.tensor(act, device=self.device, dtype = tc.float)
     
        #normalisation
        if self.state_normalization != "unnormalized":
            inputs_states = self._normalize_states(inputs_states)
        if self.action_normalization != "unnormalized":
            inputs_actions = self._normalize_actions(inputs_actions)
        
        q_values = self.critic(inputs_states, inputs_actions)

        #q-values target
        inputs_next_states = tc.tensor(obs[1:], device=self.device, dtype = tc.float)
        inputs_argmax_actions = tc.tensor(act_star[1:], device=self.device, dtype = tc.float)

        #normalisation
        if self.state_normalization != "unnormalized":
            inputs_next_states = self._normalize_states(inputs_next_states)
        if self.action_normalization != "unnormalized":
            inputs_argmax_actions = self._normalize_actions(inputs_argmax_actions)
        q_values_next = target_net(inputs_next_states, inputs_argmax_actions).detach()

        rewards = tc.unsqueeze(tc.tensor(rewards, device=self.device, dtype = tc.float), 1)
        dones = tc.unsqueeze(tc.tensor(dones, device=self.device, dtype = tc.float), 1)
        target = rewards + self.gamma * dones * q_values_next

        # log examples with high td error
        td_errors = tc.abs(q_values - target)
        max_td_error = tc.max(td_errors, 0)
        demand_max_td_error = tc.sum(inputs_states[max_td_error.indices][:,:self.demand_size])

        # and with random td error as comparison
        random_index = random.randrange(len(td_errors))
        random_td_error = td_errors[random_index]
        demand_random_td_error = tc.sum(inputs_states[random_index][:self.demand_size])
        self.log(self.num_episode, {'tdErrorStats/max_td_error/value':max_td_error.values, 'tdErrorStats/max_td_error/overall_demand': demand_max_td_error, \
            'tdErrorStats/random_td_error/value':random_td_error, 'tdErrorStats/random_td_error/overall_demand':demand_random_td_error})

        # critic loss
        if (self.per_buffer == False) | (self.direct_training == True):
            loss_theta = tc.mean(tc.square(q_values - target))
        else:
            new_priorities = tc.mean((tc.abs(q_values - target) + 1e-6).reshape((-1, 48)), -1)
            loss_theta = tc.mean(tc.square(q_values - target).reshape((-1, 48)), -1)
            self.buffer.update_priorities(self.buffer.indexes, new_priorities.detach().cpu().numpy())
            weights = tc.tensor(np.asarray(self.buffer.weights), device=self.device, dtype = tc.float)
            loss_theta = tc.mean(weights * loss_theta)        

        # update network params
        self.optimizer_critic.zero_grad()
        loss_theta.backward()
        self.optimizer_critic.step()

        #training proposal network

        # 1-hot encoding psdd action
        psdd_action_star = tc.transpose(tc.tensor(psdd_act_star, device=self.device, \
            dtype = tc.int64), 0, 1)
        psdd_action_star_onehot = nn.functional.one_hot(psdd_action_star, num_classes = 2)

        # masking off network heads of unused decision nodes + recording number of used ones
        no_training_steps = self.timesteps * self.batch_size
        mask = np.zeros((self.num_decom_nodes, no_training_steps))
        step = 0
        no_actionHeads = np.zeros((no_training_steps, 1))
        for epi in argmax_actionHeads:
            for el in epi:
                mask[el, [step for i in range(len(el))]] = 1
                no_actionHeads[step, 0] = len(el)
                step += 1
        mask = tc.tensor(mask, device=self.device, dtype= tc.float)
        no_actionHeads = tc.tensor(no_actionHeads, device=self.device, dtype= tc.float)

        # getting prediction by proposal network
        inputs_states = tc.tensor(np.asarray(obs[:-1]), device=self.device, dtype = tc.float)

        #normalisation
        if self.state_normalization != "unnormalized":
            inputs_states = self._normalize_states(inputs_states)

        output_probs = self.actor(inputs_states)
        prob_norm = tc.mean(tc.sqrt((output_probs - 0.5) ** 2))
        output_probs = tc.transpose(output_probs, 0, 1)

        # proposal loss
        loss_entropy = tc.sum(tc.sum(tc.sum(tc.reshape(mask, (-1, no_training_steps, 1)) * output_probs * \
            tc.log(1e-6+output_probs) / no_actionHeads, -2), -1), -1) # maybe remove masking?
        loss_max_action = tc.sum(tc.sum(tc.sum(tc.reshape(mask, (-1, no_training_steps, 1)) * tc.log(1e-6+output_probs) * \
            psdd_action_star_onehot, -2), -1), -1)
        loss_mu = -loss_max_action + self.coeff_entrop * loss_entropy

        self.optimizer_actor.zero_grad() #set_to_none=True saved around 3min in a 10-80 experiment
        loss_mu.backward()
        self.optimizer_actor.step()

        return loss_theta, loss_mu, loss_max_action, loss_entropy, prob_norm

    
    def _step_based_training(self, steps, target_net):
        t_start = time()

        # critic training
        #getting data
        steps = np.asarray(steps, dtype = object)
        inputs_states = tc.tensor(np.stack(steps[:, 0]), device=self.device, dtype = tc.float)
        inputs_actions = tc.tensor(np.stack(steps[:, 1]), device=self.device, dtype = tc.float)

        #normalisation
        if self.state_normalization != "unnormalized":
            inputs_states = self._normalize_states(inputs_states)
        if self.action_normalization != "unnormalized":
            inputs_actions = self._normalize_actions(inputs_actions)

        q_values = self.critic(inputs_states, inputs_actions)

        inputs_next_states = tc.tensor(np.stack(steps[:, 2]), device=self.device, dtype = tc.float)
        inputs_next_actions = tc.tensor(np.stack(steps[:, 3]), device=self.device, dtype = tc.float)
        
        #normalisation
        if self.state_normalization != "unnormalized":
            inputs_next_states = self._normalize_states(inputs_next_states)
        if self.action_normalization != "unnormalized":
            inputs_next_actions = self._normalize_actions(inputs_next_actions)

        q_values_next = target_net(inputs_next_states, inputs_next_actions).detach()

        rewards = tc.tensor(np.stack(steps[:, 4]), device=self.device, dtype = tc.float)
        dones = tc.tensor(np.stack(steps[:, 7]), device=self.device, dtype = tc.float)
        target = tc.unsqueeze(rewards, 1) + self.gamma * tc.unsqueeze(dones, 1) * q_values_next
        
        # log examples with high td error
        if np.random.rand() <= 1/self.timesteps:
            td_errors = tc.abs(q_values - target)
            max_td_error = tc.max(td_errors, 0)
            demand_max_td_error = tc.sum(inputs_states[max_td_error.indices][:,:self.demand_size])

            # and with random td error as comparison
            random_index = random.randrange(self.batch_size)
            random_td_error = td_errors[random_index]
            demand_random_td_error = tc.sum(inputs_states[random_index][:self.demand_size])
            self.log(self.num_step, {'tdErrorStats/max_td_error/value':max_td_error.values, 'tdErrorStats/max_td_error/overall_demand': demand_max_td_error, \
                'tdErrorStats/random_td_error/value':random_td_error, 'tdErrorStats/random_td_error/overall_demand':demand_random_td_error})
        
        # critic loss
        if self.per_buffer == False:
            loss_theta = tc.mean(tc.square(q_values - target))
        else:
            new_priorities = tc.abs(q_values - target) + 1e-6
            self.buffer.update_priorities(self.buffer.indexes, new_priorities.detach().cpu().numpy())
            weights = tc.tensor(np.asarray(self.buffer.weights), device=self.device, dtype = tc.float)
            loss_theta = tc.mean(weights * tc.square(q_values - target))
        
        self.optimizer_critic.zero_grad()
        loss_theta.backward()
        self.optimizer_critic.step()

        t_proposal = time()

        # proposal training

        psdd_action_star = tc.transpose(tc.tensor(np.stack(steps[:, 5]), device=self.device, dtype = tc.int64), 0, 1)
        psdd_action_star_onehot = nn.functional.one_hot(psdd_action_star, num_classes = 2)

        t_mask = time()
        mask = np.zeros((self.num_decom_nodes, self.batch_size))
        step = 0
        no_actionHeads = np.zeros((self.batch_size, 1))
        for el in steps[:, 6]:
            mask[el, [step for i in range(len(el))]] = 1
            no_actionHeads[step, 0] = len(el)
            step += 1
        mask = tc.tensor(mask, device=self.device, dtype= tc.float)
        no_actionHeads = tc.tensor(no_actionHeads, device=self.device, dtype= tc.float)
    
        t_props = time()
        output_probs = self.actor(inputs_states)
        prob_norm = tc.sum((output_probs - 0.5) ** 2)
        output_probs = tc.transpose(output_probs, 0, 1) # OBDD VERSION 4/4

        t_loss = time()
        # proposal loss:
        loss_max_action = tc.sum(tc.sum(tc.sum(tc.reshape(mask, (-1, self.batch_size, 1)) * tc.log(1e-6+output_probs) * \
            psdd_action_star_onehot, -2), -1), -1)
        if self.coeff_entrop > 0:
            loss_entropy = tc.sum(tc.sum(tc.sum(tc.reshape(mask, (-1, self.batch_size, 1)) * output_probs * \
                tc.log(1e-6+output_probs) / no_actionHeads, -2), -1), -1)
        else:
            loss_entropy = 0
        loss_mu = -loss_max_action + self.coeff_entrop * loss_entropy

        t_backprop = time()
        self.optimizer_actor.zero_grad() #set_to_none=True saves some time
        loss_mu.backward()
        self.optimizer_actor.step()

        #print("total, one_hot, mask, probs, loss, backprop", time() - t_proposal, t_mask - t_proposal, t_props - t_mask, t_loss - t_props, t_backprop - t_loss, time() - t_backprop)

        #print("total, critic, proposal", time() - t_start, t_proposal - t_start, time() - t_proposal)

        return loss_theta, loss_mu, loss_max_action, loss_entropy, prob_norm


    def train(self):
        t_choose_net = time()
        # choose target network:
        if (self.target_net_update_every == 0) & (self.tau == 0): # not using target net
            target_net = self.critic
        elif self.target_net_update_every > 0: # using target net & updating it
            if self.num_episode % self.target_net_update_every == 0:
                if self.tau > 0: # polyak averaging
                    with tc.no_grad():
                        for param, target_param in zip(self.critic.parameters(), self.target.parameters()):
                            target_param.data.mul_(1 - self.tau)
                            tc.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
                else:
                    self.target.load_state_dict(self.critic.state_dict())
            target_net = self.target
        else:
            raise ValueError("target_net_update_every must be >= 0")

        #direct training
        if self.direct_training:
            loss_theta, loss_mu, loss_max_action, loss_entropy, prob_norm = self._episode_based_training([[self.obs_list, \
                self.action_list, self.action_star_list, self.psdd_action_star_list, self.argmax_actionHeads_list, \
                self.reward_list, self.done_list]], target_net)
            self.log(self.num_episode, {'Loss/DirectTraining/mu':loss_mu, 'Loss/DirectTraining/theta': loss_theta, \
                'Loss/DirectTraining/entropy':loss_entropy, 'Loss/DirectTraining/max_action':loss_max_action, \
                'PSDDParams/DirectTraining/norm':prob_norm})
        else:
            loss_mu = 0
            loss_theta = 0
        t_train_replay = time()
        # training from replay:
        if self.replay_training:
            avg_loss_theta = 0
            avg_loss_mu = 0
            avg_loss_max_action = 0
            avg_loss_entropy = 0
            avg_prob_norm = 0
            for i in range(self.replay_ratio):
                if self.mode == "episode":
                    if self.per_buffer == True:
                        eps = self.buffer.sample(self.batch_size, self.beta)
                    else:
                        eps = self.buffer.sample(self.batch_size)
                    loss_theta, loss_mu, loss_max_action, loss_entropy, prob_norm = self._episode_based_training(eps, target_net)
                elif self.mode == "step":
                    if self.per_buffer == True:
                        steps = self.buffer.sample(self.batch_size, self.beta)
                    else:
                        steps = self.buffer.sample(self.batch_size)
                    loss_theta, loss_mu, loss_max_action, loss_entropy, prob_norm = self._step_based_training(steps, target_net)
                avg_loss_theta += loss_theta
                avg_loss_mu += loss_mu
                avg_loss_max_action += loss_max_action
                avg_loss_entropy += loss_entropy
                avg_prob_norm += prob_norm
            t_log = time()
            if self.replay_ratio > 0:
                avg_loss_theta = avg_loss_theta / self.replay_ratio
                avg_loss_mu = avg_loss_mu / self.replay_ratio
                avg_loss_max_action = avg_loss_max_action / self.replay_ratio
                avg_loss_entropy = avg_loss_entropy / self.replay_ratio
                avg_prob_norm = avg_prob_norm / self.replay_ratio
                self.log(self.num_episode, {'Loss/Replay/mu_avg':avg_loss_mu, 'Loss/Replay/theta_avg': avg_loss_theta, \
                    'Loss/Replay/max_action_avg':avg_loss_max_action, 'Loss/Replay/entropy_avg':avg_loss_entropy, \
                    'PSDDParams/Replay/norm_avg':avg_prob_norm})
        #print('total, choose net, train, log', time() - t_choose_net, t_train_replay - t_choose_net, t_log - t_train_replay, time() - t_log)


    def update_and_log_hyperparameters(self):

        # adapt param noise:
        if self.param_noise_sigma > 0:
            self.adaptive_param_noise.reset_noise(self, self.batch_size)

        # logging exploration
        self.log(self.num_episode, {'Exploration/epsilon':self.epsilon, 'Exploration/M':self.M})

        #epsilon decaying
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        #M decaying
        self.M = math.ceil(max(1, self.M_start - self.num_episode * self.M_decay))

        #increasing beta:
        self.beta = self.beta + 1/self.total_training_episodes * (1 - self.beta_start)

        #sampling actions from multiple lists
        if ((self.num_episode) % (math.ceil(self.total_training_episodes/self.uniform_action_sampling)) == 0) & (self.num_episode != 0):
            del self.UEABuffer
            gc.collect()
            if self.uniform_action_sampling > 1:
                no = int(self.presampledFileName[len(self.presampledFileNameBase):-4]) + 1
                self.presampledFileName = self.presampledFileNameBase + str(no) + '.pkl'
            self.UEABuffer = UniformEnvActionsBuffer(self.presampledFileName)
            print('new file: ' + self.presampledFileName)

        self.num_episode += 1


    def getAction(self, obs):
        inputs = tc.unsqueeze(tc.tensor(obs, device=self.device, dtype=tc.float),0)

        # choose actor network:
        if self.param_noise_sigma > 0:
            actor_net = self.noisy
        else:
            actor_net = self.actor

        #normalisation
        if self.state_normalization != "unnormalized":
            inputs = self._normalize_states(inputs)

        outputs = actor_net(inputs)[0]

        proposalEnvActions = []
        proposalPSDDActions = []
        uniformEnvActions = []
        uniformPSDDActions = []

        sample_layer_proposal = categorical.Categorical(outputs)
        sample_shape_N = tc.Size([self.N])
        psdd_actions_proposal = sample_layer_proposal.sample(sample_shape_N)

        #collect neural network head id for all decision nodes used in sampling actions
        proposalHeads = []
        uniformHeads = []

        for i in range(self.N): # can speed up using multiprocessing.Pool.map?
            psdd_action = psdd_actions_proposal[i].tolist()

            #retrieve literals according to psdd
            literals, heads_of_nodes_traversed = self.getLiterals(psdd_action)
            #covert literals to actual action
            env_action = self.litToEnvAction(literals)

            proposalEnvActions.append(env_action)
            proposalPSDDActions.append(psdd_action)
            proposalHeads.append(heads_of_nodes_traversed)

        if self.M > 0:
            indices = self.UEABuffer.get_sample_indices(self.M)
            uniformEnvActions = [self.UEABuffer.buffer[i][1] for i in indices]

        if self.pooling > 1:
            proposalEnvActions = [[zone_occupancy * self.pooling for zone_occupancy in action] for action in proposalEnvActions]

        unionEnvActions = proposalEnvActions + uniformEnvActions

        inputs_states = inputs.repeat(self.N + self.M,1)
        inputs_actions = tc.tensor(unionEnvActions, device=self.device, dtype = tc.float) # what about just taking the psdd actions instead?

        # normalization of actions (states are already normalized)
        if self.action_normalization != "unnormalized":
            inputs_actions = self._normalize_actions(inputs_actions)

        q_values = self.critic(inputs_states, inputs_actions)

        argmax_action_id = np.argmax(q_values.cpu().data.numpy())
        argmax_env_action = unionEnvActions[argmax_action_id]
        if argmax_action_id >= self.N:
            short_argmax_action_id = argmax_action_id - self.N  
            argmax_psdd_action = self.UEABuffer.buffer[indices[short_argmax_action_id]][0]
            argmax_actionHeads = self.UEABuffer.buffer[indices[short_argmax_action_id]][2]

        else:
            argmax_psdd_action = proposalPSDDActions[argmax_action_id]
            argmax_actionHeads = proposalHeads[argmax_action_id]
            self.successful_proposal_cnt += 1     

        # action stats and maxQ for logging

        #how often does each head get trained?
        for head in argmax_actionHeads:
            self.head_trained_stat[head] += 1
        if self.num_step % (self.timesteps - 1) == 0:
            #diff from average action
            argmax_env_action_balance = np.sum((argmax_env_action / self.capacity_limits - self.nresources / \
                sum(self.capacity_limits)) ** 2)
            #statistic for action
            argmax_env_action_scalar_action = sum([i * argmax_env_action[i] for i in range(self.action_size)])/self.action_size
            #diversity of proposal actions:
            proposal_diversity_count = len(set([str(act) for act in proposalEnvActions]))
            avg_zone_std_of_proposed_actions = np.mean(np.std(np.asarray(proposalEnvActions).T, 1))


            #log
            self.log(self.num_step, {"ActionStats/env-actions/argmax/balance":argmax_env_action_balance, \
                "ActionStats/env-actions/argmax/scalar_action":argmax_env_action_scalar_action, \
                "Q-Values/Q*":q_values[argmax_action_id], "ProposedPolicyStats/SuccessfulProposal_Count":self.successful_proposal_cnt, \
                "ProposedPolicyStats/proposal_diversity_count":proposal_diversity_count, \
                "ProposedPolicyStats/avg_zone_std_of_proposed_actions":avg_zone_std_of_proposed_actions, \
                "ActorTraining/min_training_per_head":np.min(self.head_trained_stat), "ActorTraining/avg_training_per_head":np.mean(self.head_trained_stat)})
            self.successful_proposal_cnt = 0

        #epsilon greedy
        if np.random.rand() < self.epsilon:
            selected_env_action = self.UEABuffer.buffer[indices[0]][1]
            selected_psdd_action = self.UEABuffer.buffer[indices[0]][0]
        else:
            selected_env_action = argmax_env_action
            selected_psdd_action = argmax_psdd_action

        return selected_psdd_action, selected_env_action, argmax_env_action, argmax_psdd_action, argmax_actionHeads

    def getLiterals(self, psdd_action):
        #sample literals
        literals = []
        heads_of_nodes_traversed = []
        branching_nodes = [self.root_node_id]
        while len(branching_nodes) != 0:
            node_to_explore = branching_nodes.pop(0)
            if self.nodes[node_to_explore][0] == 'L':
                vtree_id, lit = self.nodes[node_to_explore][1], self.nodes[node_to_explore][2]
                literals.append(lit)
            else: 
            #it is a decomposition node, we need to select which branch to go
                if self.nodes[node_to_explore][2] == 1:
                    branch_id = 0
                else:
                    pos = self.node_id2pos[node_to_explore]
                    branch_id = psdd_action[pos]
                    heads_of_nodes_traversed.append(pos)
                branching_nodes.append(self.nodes[node_to_explore][3][branch_id][0]) #prime
                branching_nodes.append(self.nodes[node_to_explore][3][branch_id][1]) #sub
        return literals, heads_of_nodes_traversed

    def litToEnvAction(self, literals): 
        var_ins = literals
        instantiation_dict = {}
        for i in var_ins:
            if i > 0:
                instantiation_dict[i] = 1
            else:
                instantiation_dict[-i] = 0
        action = []
        for zone in self.zones2vars.keys():
            variables = self.zones2vars[zone]['vars']
            sub_action = 0
            for var in variables:
                if instantiation_dict[var] == 1:
                    sub_action += 1
                else:
                    break
            action.append(sub_action)
        return action


    def save_model(self, path):
        tc.save(self.actor.state_dict(), path + "actor.pt")
        tc.save(self.critic.state_dict(), path + "critic.pt")
        if self.target_net_update_every > 0:
            tc.save(self.target.state_dict(), path + "target.pt")

        running_stats = []
        pklfile = open(path + 'running_stats.pkl', 'ab')
        if self.state_normalization != "unnormalized":
            running_stats.append(self.state_running_stats)
        if self.action_normalization == "average":
            running_stats.append(self.action_running_stats)    
        pickle.dump(running_stats, pklfile)


    def load_model(self, path):
        self.actor.load_state_dict(tc.load(path + "actor.pt"))
        self.critic.load_state_dict(tc.load(path + "critic.pt"))
        if self.target_net_update_every > 0:
            self.target.load_state_dict(tc.load(path + "target.pt"))
        
        running_stats_file = open(path + 'running_stats.pkl', "rb")
        running_stats = pickle.load(running_stats_file)
        running_stats_file.close()
        print(running_stats)
        if self.state_normalization != "unnormalized":
            self.state_running_stats = running_stats[0]
        if self.action_normalization == "average":
            self.action_running_stats = running_stats[-1]


    def log(self, x, to_log):
        """
        to_log: expects dict of variables to be logged and their values
        x: episode or step number
        """
        for var, value in to_log.items():
            self.writer.add_scalar(var, value, x)