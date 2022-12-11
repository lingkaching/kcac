import os
import numpy as np
import torch as tc


def getLiterals(psdd_action, nodes, root_node_id, node_id2pos):
    #sample literals
    literals = []
    heads_of_nodes_traversed = []
    branching_nodes = [root_node_id]
    while len(branching_nodes) != 0:
        node_to_explore = branching_nodes.pop(0)
        if nodes[node_to_explore][0] == 'L':
            vtree_id, lit = nodes[node_to_explore][1], nodes[node_to_explore][2]
            literals.append(lit)
        else: 
        #it is a decomposition node, we need to select which branch to go
            if nodes[node_to_explore][2] == 1:
                branch_id = 0
            else:
                pos = node_id2pos[node_to_explore]
                branch_id = psdd_action[pos]
                heads_of_nodes_traversed.append(pos)
            branching_nodes.append(nodes[node_to_explore][3][branch_id][0]) #prime
            branching_nodes.append(nodes[node_to_explore][3][branch_id][1]) #sub
    return literals, heads_of_nodes_traversed

def litToEnvAction(literals, zones2vars): 
    var_ins = literals
    instantiation_dict = {}
    for i in var_ins:
        if i > 0:
            instantiation_dict[i] = 1
        else:
            instantiation_dict[-i] = 0
    action = []
    for zone in zones2vars.keys():
        variables = zones2vars[zone]['vars']
        sub_action = 0
        for var in variables:
            if instantiation_dict[var] == 1:
                sub_action += 1
            else:
                break
        action.append(sub_action)
    return action

def readPSDD(psddFileName, device):
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

    uniform_probs = tc.tensor(np.asarray(uniform_probs), device = device)
    sample_layer_uniform = categorical.Categorical(uniform_probs)
           
    f.close()
    return nodes, cnt, node_id2pos, sample_layer_uniform, node_id


def readZoneVar(zones2varsFileName):
    zones2varsFile = open(zones2varsFileName, "rb")
    zones2vars=pickle.load(zones2varsFile)
    zones2varsFile.close()

    return zones2vars

def presample_uniform_actions(name, log_dir, uniform_action_sampling, presampledFileNameBase, no_presampled_actions_per_file, device, pooling, zones2varsFileName, psddFileName):
    uniform_actions_dir = log_dir + 'psddAgentFiles/uniformActions/'
    if not os.path.isdir(uniform_actions_dir):
        os.mkdir(uniform_actions_dir)
    uniform_actions_dir = log_dir + 'psddAgentFiles/uniformActions/' + name + '/'
    if not os.path.isdir(uniform_actions_dir):
        os.mkdir(uniform_actions_dir)
    zones2vars = readZoneVar(zones2varsFileName)
    nodes, num_decom_nodes, node_id2pos, sample_layer_uniform, root_node_id = readPSDD(psddFileName, device)
    print(uniform_action_sampling)
    for i in range(uniform_action_sampling):
        presampledFileName = presampledFileNameBase + str(i) + ".pkl"
        if not os.path.isfile(presampledFileName):
            pklfile = open(presampledFileName, 'ab')
            samples = []
            for j in range(no_presampled_actions_per_file):
                sample_shape = tc.Size([1])
                psdd_actions_uniform = sample_layer_uniform.sample(sample_shape)
                psdd_action = psdd_actions_uniform[0].tolist()
                
                #sample literals according to psdd
                literals, heads_of_nodes_traversed = getLiterals(psdd_action, nodes, root_node_id, node_id2pos)
                #covert literals to actual action
                env_action = litToEnvAction(literals, zones2vars)
                #pooling
                if pooling > 1:
                    env_action = [int(el * 2) for el in env_action]

                samples.append([psdd_action, env_action, heads_of_nodes_traversed])
            pickle.dump(samples, pklfile)
            pklfile.close()
        print(i + 1, 'of', uniform_action_sampling)

presample_uniform_actions('testname', './', 10, 'testuniformactionsampling', 10_000, tc.device("cuda" if tc.cuda.is_available() else "cpu"), 1, 'test-zones2vars.pkl', 'testpsdd.psdd')