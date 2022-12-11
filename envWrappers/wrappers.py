import numpy as np
import gym
import gym_ERSLE
import gym_BSS
from collections import deque

def convert_to_constraints_dict(nbases, nresources, min_constraints, max_constraints):
    constraints = {
        "name": "root_node",
        "equals": nresources,
        "max": nresources,
        "min": nresources,
        "children": []
    }
    for i in range(nbases):
        child = {
            "name": "zone{0}".format(i),
            "zone_id": i,
            "equals": None,
            "min": min_constraints[i],
            "max": max_constraints[i],
            "children": []
        }
        constraints["children"].append(child)
    return constraints

def count_leaf_nodes_in_constraints(constraints):
    if 'children' not in constraints or len(constraints['children']) == 0:
        return 1
    else:
        count = 0
        for child_constraints in constraints['children']:
            count += count_leaf_nodes_in_constraints(child_constraints)
        return count

class MMDPObsStackWrapper(gym.Wrapper):

    def __init__(self, env, k):
        #logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.k = k
        self.last_k_demands = deque([], maxlen=self.k)
        low = list(self.env.observation_space.low)
        low = low[0:self.metadata['nzones']] * self.k + low[self.metadata['nzones']:]
        high = list(self.env.observation_space.high)
        high = high[0:self.metadata['nzones']] * self.k + high[self.metadata['nzones']:]
        self.observation_space = gym.spaces.Box(
            low=np.array(low), high=np.array(high), dtype=self.env.observation_space.dtype)

    def _observation(self):
        assert len(self.last_k_demands) == self.k
        obs = np.concatenate((np.concatenate(
            self.last_k_demands, axis=0), self.obs[self.metadata['nzones']:]), axis=0)
        return obs

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.last_k_demands.append(self.obs[0:self.metadata['nzones']])
        return self._observation()

    def step(self, action):
        self.obs, r, d, info = self.env.step(action)
        self.last_k_demands.append(self.obs[0:self.metadata['nzones']])
        return self._observation(), r, d, info

class ERStoMMDPWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        #logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.metadata['nzones'] = self.metadata['nbases']
        self.metadata['nresources'] = self.metadata['nambs']
        if 'constraints' not in self.metadata or self.metadata['constraints'] is None:
            self.metadata['constraints'] = convert_to_constraints_dict(
                self.metadata['nzones'], self.metadata['nresources'], env.action_space.low, env.action_space.high)
        assert count_leaf_nodes_in_constraints(
            self.metadata['constraints']) == self.metadata['nzones'], "num of leaf nodes in constraints tree should be same as number of zones"

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class BSStoMMDPWrapper(gym.Wrapper):
    def __init__(self, env):
        #logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.metadata['nzones'] = self.metadata['nzones']
        self.metadata['nresources'] = self.metadata['nbikes']
        if 'constraints' not in self.metadata or self.metadata['constraints'] is None:
            self.metadata['constraints'] = convert_to_constraints_dict(
                self.metadata['nzones'], self.metadata['nresources'], env.action_space.low, env.action_space.high)
        assert count_leaf_nodes_in_constraints(
            self.metadata['constraints']) == self.metadata['nzones'], "num of leaf nodes in constraints tree should be same as number of zones"

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class InfeasibleActionDetectionWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return self.env.reset() 

    def _evaluate_constraint(self, node, action):
        if 'children' not in node.keys():
            sub_action = action[node['zone_id']]
        elif len(node['children']) == 0:
            sub_action = action[node['zone_id']]
        else:
            sub_action = 0
            for child in node['children']:
                sub_action += self._evaluate_constraint(child, action)
        if node['min'] is not None:
            assert sub_action >= node['min'], print('min violated: ' + node['name'] + '; ' + str(action) + '; ' + str(node['min']))
        if node['max'] is not None:
            assert sub_action <= node['max'], print('max violated: ' + node['name'] + '; ' + str(action) + '; ' + str(node['max']))
        if node['equals'] is not None:
            assert sub_action == node['equals'], print('equals: ' + node['name'] + '; ' + str(action) + '; ' + str(node['equals']))
        return sub_action

    def step(self, action):
        """assumes all action constraints are part of the constraints in the metadata"""
        root_node = self.metadata['constraints']
        if root_node is not None:
            self._evaluate_constraint(root_node, action)
        return self.env.step(action)