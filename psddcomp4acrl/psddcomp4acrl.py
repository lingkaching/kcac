import numpy as np
import pandas as pd
import csv
import os
import shutil
from pysdd.sdd import Vtree, SddManager
import pickle
import itertools
import subprocess


class SDD:

	def __init__(self, sddfile, vtreefile):
		self.sddfile = sddfile
		self.vtreefile = vtreefile


	@classmethod
	def _build_sdd_filename(cls, n, k, constraint_type, path):
		sddfile = path + str(n) + '-' + str(k) + '-' + constraint_type + '.sdd'
		return sddfile


	@classmethod
	def _check_and_correct_constraint(cls, n, k, constraint_type):
		if (constraint_type == 'geq') & (k == 0):
			raise ValueError('k=0 not permissible for "geq" constraints; trivial case')
		if (constraint_type == 'leq') & (n == k):
			raise ValueError('k=n not permissible for "leq" constraints; trivial case')

		#re-routing corner cases:
		if (constraint_type == 'eq') & (k == 0):
			constraint_type = 'leq'
		if (constraint_type == 'eq') & (k == n):
			constraint_type = 'geq'
		return constraint_type


	@classmethod
	def _calculate_sdd_size(cls, n, k, constraint_type):
		no_literal_and_sink_nodes = 2 * n + 1
		if constraint_type == 'geq':
			no_decision_nodes = k * (n - k + 1) - 1
			if (k == n) | (k == 1):
				no_literal_and_sink_nodes -= 1
		elif constraint_type == 'leq':
			no_decision_nodes = (k + 1) * (n - k) - 1
			if (k == n - 1) | (k == 0):
				no_literal_and_sink_nodes -= 1
		elif constraint_type == 'eq':
			no_decision_nodes = (k + 1) * (n - k + 1) - 3
		else:
			raise ValueError('Use onstraint type "eq", "geq" or "leq"')
		size = no_literal_and_sink_nodes + no_decision_nodes
		return size


	@classmethod
	def _create_literals(cls, n, constraint_type):
		literals = ''
		if constraint_type == 'geq':
			for i in range(n):
				literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
				if i != n - 1:
					literals = literals + 'L ' + str(i + n) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
		elif constraint_type == 'leq':
			for i in range(n):
				literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
				if i != n - 1:
					literals = literals + 'L ' + str(i + n) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
		elif constraint_type == 'eq':
			for i in range(n):
				literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
				literals = literals + 'L ' + str(i + n) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
		else:
			raise ValueError('Use onstraint type "eq", "geq" or "leq"')
		return literals


	@classmethod
	def _create_sinks(cls, n, k, constraint_type, sdd_size_str):
		if constraint_type in ['leq', 'geq']:
			if n > 1:
				if ((constraint_type == 'leq') & (k == 0)) | \
				((constraint_type == 'geq') & (k == n)):
					sinks = 'F ' + str(2 * n - 1) + '\n'
				elif ((constraint_type == 'leq') & (k == n - 1)) | \
				((constraint_type == 'geq') & (k == 1)):
					sinks = 'T ' + str(2 * n - 1) + '\n'
				else:
					sinks = 'T ' + str(2 * n - 1) + '\nF ' + str(2 * n) + '\n'
			else:
				sinks = 'no_sinks'
				sdd_size_str = 'sdd 1\n'
		elif constraint_type == 'eq':
			if n > 2:
				sinks = 'F ' + str(2 * n) + '\n'
			else: # case where n = 2 and k = 1
				sinks = 'no_sinks'
				sdd_size_str = 'sdd 5\n'
		else:
			raise ValueError('Use onstraint type "eq", "geq" or "leq"')
		return sinks, sdd_size_str


	@classmethod
	def _np2pd(cls, decision_nodes):
		if decision_nodes.flags['F_CONTIGUOUS']:
			decision_nodes = np.ascontiguousarray(decision_nodes)
		dtype = decision_nodes.dtype
		dtype = [('ID', dtype), ('vtree', dtype), ('no_kids', dtype), ('tprime', dtype), ('tsub', dtype), ('fprime', dtype), ('fsub', dtype), ('level', dtype)]
		decision_nodes.dtype = dtype
		decision_nodes[::-1].sort(0, order = 'level')
		decision_nodes = decision_nodes[['ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']]
		decision_nodes = pd.DataFrame(decision_nodes.flatten())
		decision_nodes['prefix'] = 'D'
		return decision_nodes


	@classmethod
	def _create_decision_nodes(cls, n, k, constraint_type):
		if constraint_type == 'geq':
			# create decision node array and row+column grid:
			decision_nodes = np.zeros([8, k, n - k + 1], dtype = int)
			grid = np.mgrid[0:k,0:n - k + 1]
			#set level
			decision_nodes[7] = grid[0] + grid[1] + 1
			#set IDs
			decision_nodes[0] = (2 * n + 1) + (n - k + 1) * grid[0] + grid[1]
			if (k == n) | (k == 1):
				decision_nodes[0] -= 1
			#set vtree-nodes:
			decision_nodes[1] = decision_nodes[7] * 2 - 1
			#set number of elements:
			decision_nodes[2] = 2
			#set true child, prime:
			decision_nodes[3] = decision_nodes[7] - 1
			#set false child, prime:
			decision_nodes[5] = decision_nodes[7] - 1 + n
			#set true child, sub:
			decision_nodes[4] = decision_nodes[0] + n - k + 1
			decision_nodes[4][k - 1] = 2 * n - 1
			decision_nodes[4][k - 2][n - k] = n - 1
			#set false child, sub:
			decision_nodes[6] = decision_nodes[0] + 1
			decision_nodes[6][:, n - k] = 2 * n
			if k == n:
				decision_nodes[6][:, n - k] -= 1
			decision_nodes[6][k - 1][n - k - 1] = n - 1
			# reshaping the array so that  info for each decision node is represented by 1 row
			decision_nodes = np.reshape(decision_nodes.T, (k * (n - k + 1), 8), order = 'C')
			# configuring the numpy array into a sorted pandas dataframe:
			decision_nodes = cls._np2pd(decision_nodes)
			#removing the bottom OBDD decision node that is subsumed into a literal in SDD
			decision_nodes.drop(decision_nodes.head(1).index,inplace=True)
			
		elif constraint_type == 'leq':
			# create decision node array and row+column grid:
			decision_nodes = np.zeros([8, k + 1, n - k], dtype = int)
			grid = np.mgrid[0:k + 1, 0:n - k]
			#set level
			decision_nodes[7] = grid[0] + grid[1] + 1
			#set IDs
			decision_nodes[0] = (2 * n + 1) + (n - k) * grid[0] + grid[1]
			if (k == 0) | (k == n - 1):
				decision_nodes[0] -= 1
			#set vtree-nodes:
			decision_nodes[1] = decision_nodes[7] * 2 - 1
			#set number of elements:
			decision_nodes[2] = 2
			#set true child, prime:
			decision_nodes[3] = decision_nodes[7] - 1 + n
			#set false child, prime:
			decision_nodes[5] = decision_nodes[7] - 1
			#set true child, sub:
			decision_nodes[4] = decision_nodes[0] + n - k
			decision_nodes[4][k] = 2 * n
			if k == 0:
				decision_nodes[4][k] -= 1
			decision_nodes[4][k - 1][n - k - 1] = n - 1
			#set false child, sub:
			decision_nodes[6] = decision_nodes[0] + 1
			decision_nodes[6][:, n - k - 1] = 2 * n - 1
			decision_nodes[6][k][n - k - 2] = n - 1
			# reshaping the array so that  info for each decision node is represented by 1 row
			decision_nodes = np.reshape(decision_nodes.T, ((k + 1) * (n - k), 8), order = 'C')
			# configuring the numpy array into a pandas dataframe:
			decision_nodes = cls._np2pd(decision_nodes)
			#removing the bottom OBDD decision node that is subsumed into a literal in SDD
			decision_nodes.drop(decision_nodes.head(1).index,inplace=True)

		elif constraint_type == 'eq':
			# create decision node array and row+column grid:
			decision_nodes = np.zeros([8, k + 1, n - k + 1], dtype = int)
			grid = np.mgrid[0:k + 1,0:n - k + 1]
			#set level
			decision_nodes[7] = grid[0] + grid[1] + 1
			#set IDs
			if (n == 2) & (k == 1):
				decision_nodes[0] = (2 * n) + (n - k + 1) * grid[0] + grid[1]
			else:
				decision_nodes[0] = (2 * n + 1) + (n - k + 1) * grid[0] + grid[1]
			#set vtree-nodes:
			decision_nodes[1] = decision_nodes[7] * 2 - 1
			#set number of elements:
			decision_nodes[2] = 2
			#set true child, prime:
			decision_nodes[3] = decision_nodes[7] - 1
			#set false child, prime:
			decision_nodes[5] = decision_nodes[7] - 1 + n
			#set true child, sub:
			decision_nodes[4] = decision_nodes[0] + n - k + 1
			decision_nodes[4][k] = 2 * n
			decision_nodes[4][k - 1][n - k - 1] = 2 * n - 1
			decision_nodes[4][k - 2][n - k] = n - 1
			#set false child, sub:
			decision_nodes[6] = decision_nodes[0] + 1
			decision_nodes[6][:, n - k] = 2 * n
			decision_nodes[6][k - 1][n - k - 1] = n - 1
			decision_nodes[6][k][n - k - 2] = 2 * n - 1
			# rectify decision node ID shift due to decision node being deleted from second last row in SDD representation
			decision_nodes[0][k] = decision_nodes[0][k] - 1
			decision_nodes[4][k - 1, :n - k - 1] = decision_nodes[4][k - 1, :n - k - 1] - 1
			decision_nodes[6][k, :n - k - 2] = decision_nodes[6][k, :n - k - 2] - 1
			# reshaping the array so that  info for each decision node is represented by 1 row
			decision_nodes = np.reshape(decision_nodes.T, ((k + 1) * (n - k + 1), 8), order = 'C')
			# configuring the numpy array into a pandas dataframe:
			decision_nodes = cls._np2pd(decision_nodes)
			#removing the bottom OBDD decision node that is subsumed into a literal in SDD
			decision_nodes.drop(decision_nodes.head(3).index,inplace=True)

		else:
			raise ValueError('Use onstraint type "eq", "geq" or "leq"')

		return decision_nodes


	@classmethod
	def _unadjusted_cardinality_constraint_to_sdd(cls, n, k, constraint_type, path):
		"""
		parameters:
		n: number of variables, a positive integer; 
		k: bound, positive integer; 
		constraint_type: either "leq" meaning <= or less than equal to, "eq" meaning = or equal to, or "geq" meaning >= or greater than of equal to; 
		path: path were sdd file will be stored

		returns:
		path to sdd file
		"""

		# check whether file exists already
		sddfile = cls._build_sdd_filename(n, k, constraint_type, path)
		files = os.listdir(path)
		if sddfile[len(path):] in files:
			return sddfile

		else:
			#reject trivial cases and change constraints for others:
			constraint_type = cls._check_and_correct_constraint(n, k, constraint_type)

			#create header
			header = 'c ids of sdd nodes start at 0\nc sdd nodes appear bottom-up, children before parents\nc\nc file syntax:\nc sdd count-of-sdd-nodes\nc F id-of-false-sdd-node\nc T id-of-true-sdd-node\nc L id-of-literal-sdd-node id-of-vtree literal\nc D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*\nc\n'

			#create sdd-size row:
			sdd_size = cls._calculate_sdd_size(n, k, constraint_type)
			sdd_size_str = 'sdd ' + str(sdd_size) + '\n'

			#create rows for literals:
			literals = cls._create_literals(n, constraint_type)

			#create rows for sinks:
			sinks, sdd_size_str = cls._create_sinks(n, k, constraint_type, sdd_size_str)

			#create rows for decision nodes:
			decision_nodes = cls._create_decision_nodes(n, k, constraint_type)

			# write header, sdd size and literals to .sdd-file
			file = open(sddfile, 'w')
			if sinks == 'no_sinks':
				file.write(header + sdd_size_str + literals)
			else:
				file.write(header + sdd_size_str + literals + sinks)
			file.close()

			# append decision nodes to .sdd-file
			col_order = ['prefix', 'ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']
			decision_nodes[col_order].to_csv(sddfile, mode = 'a', sep = ' ', header = False, index = False, quoting=csv.QUOTE_NONE)

			return sddfile


	@classmethod
	def _adjust_sdd(cls, tot_vars, sdd_vars, sdd_file, path, name):
		"""this function requires the SDD to follow a right-linear vtree and the \
		variable ordering of the vtree to be ascending, e.g. 1, 3, 5 but not 3, 2, 7"""

		# make new filename
		adjusted_sdd_file = os.path.join(path, name + '.sdd')



		# if the sdd_vars are at the beginning of the tot_vars write the old file to new location with new name
		if tot_vars[:sdd_vars[-1]] == sdd_vars:
			shutil.copyfile(sdd_file, adjusted_sdd_file)
			return adjusted_sdd_file

		else:
			# read in sdd
			file = open(sdd_file, 'r')
			lines = file.readlines()
			file.close()

			# decompose sdd
			header = ''
			size = 0
			literals = []
			sinks = []
			decisions = []
			for line in lines:
				if line[0] == 'c':
					header = header + line
				elif line[0] == 's':
					size = int(line[4:])
				elif line[0] == 'L':
					literals.append(line.split())
				elif line[0] == 'D':
					decisions.append(line.split())
				else:
					sinks.append(line.split())

			# discard unused vars
			last_var = sdd_vars[-1]
			last_var_index = tot_vars.index(last_var)
			tot_vars = tot_vars[:last_var_index + 1]

			# map old vtree nodes to new literals for old sdd
			vtree_node = 0
			vtree_node2var = {}
			for var in sdd_vars:
				vtree_node2var[vtree_node] = var
				vtree_node += 2
			#replace the old literals with new ones
			for literal in literals:
				literal[3] = vtree_node2var[int(literal[2])] * \
				int(abs(int(literal[3])) / int(literal[3]))

			# make dictionary mapping variables to their literal descriptions list
			literal_nodes = {}
			for literal_node in literals:
				var = abs(literal_node[3])
				if var not in literal_nodes.keys():
					literal_nodes[var] = [literal_node]
				else:
					literal_nodes[var].append(literal_node)

			# assign vtree nodes to literals:
			id_new_node = size
			vtree_node = 0
			new_literals = {}
			for var in tot_vars:
				# reassign for existing ones
				if var in literal_nodes.keys():
					for literal in literal_nodes[var]:
						literal[2] = vtree_node
				# create entire literal description for new ones
				else:
					literal_nodes[var] = [['L', id_new_node, vtree_node, var], \
					['L', id_new_node + 1, vtree_node, -var]]
					new_literals[var] = id_new_node
					new_literals[-var] = id_new_node + 1
					id_new_node += 2
				vtree_node += 2
			vtree_node -= 2

			# handle special case of no decision nodes:
			if decisions == []:
				tot_vars = tot_vars[:-1]
				new_decision_nodes = []
				new_layer = []
				decision_node_with_literal = ['D', str(id_new_node), str(vtree_node - 1), \
				str(2), str(new_literals[tot_vars[-1]]), str(0), \
				str(new_literals[-tot_vars[-1]]), str(0)]
				id_new_node += 1
				vtree_node -= 2
				new_layer.append(decision_node_with_literal)
				new_decision_nodes.append(new_layer)
				tot_vars = tot_vars[:-1]

				if len(tot_vars) > 0:
					for i in range(tot_vars[-1]):
						new_layer = []
						new_decision_node = ['D', str(id_new_node), str(vtree_node -1), str(2), \
						str(new_literals[tot_vars[-1 - i]]), str(id_new_node - 1), \
						str(new_literals[-tot_vars[-1 - i]]), str(id_new_node - 1)]
						id_new_node += 1
						vtree_node -= 2
						new_layer.append(new_decision_node)
						new_decision_nodes.append(new_layer)		

			else:
				# normal case:
				# sort decision nodes by vtree node:
				decision_nodes = {}
				for decision_node in decisions:
					if decision_node[2] not in decision_nodes.keys():
						decision_nodes[decision_node[2]] = [decision_node]
					else:
						decision_nodes[decision_node[2]].append(decision_node)

				# build new decision node structure
				tot_vars = tot_vars[:-1]
				start = True
				new_decision_nodes = []
				old_vtree_nodes = [str(el) for el in sorted(\
					[int(el) for el in decision_nodes.keys()])]
				i = -1
				replace_key_with_value = {}
				breaker = 0
				while len(tot_vars) > 0:
					breaker += 1
					if tot_vars[-1] not in sdd_vars:
						if start == True:
							succeeding_node = decision_nodes[old_vtree_nodes[i]][0]
							new_decision_node = ['D', str(id_new_node), str(vtree_node - 1), str(2), \
							str(new_literals[tot_vars[-1]]), str(succeeding_node[5]), \
							str(new_literals[-tot_vars[-1]]), str(succeeding_node[5])]
							new_decision_nodes.append([new_decision_node])
							decision_nodes[old_vtree_nodes[i]][0][5] = str(id_new_node)
							id_new_node += 1
							vtree_node -= 2
							start == False
						else:
							preceeding_node = new_decision_nodes[-1][0]
							new_decision_node = ['D', str(id_new_node), str(vtree_node - 1), str(2), \
							str(new_literals[tot_vars[-1]]), str(preceeding_node[1]), \
							str(new_literals[-tot_vars[-1]]), str(preceeding_node[1])]
							replace_key_with_value[str(preceeding_node[1])] = str(id_new_node)
							new_decision_nodes.append([new_decision_node])
							id_new_node += 1
							vtree_node -= 2
					else:
						start = False
						new_layer = []
						for decision_node in decision_nodes[old_vtree_nodes[i]]:
							new_decision_node = decision_node
							new_decision_node[2] = str(vtree_node - 1)
							loop = True
							while loop:
								loop = False
								for el in [5, 7]:
									if new_decision_node[el] in replace_key_with_value.keys():
										new_decision_node[el] = \
										replace_key_with_value[new_decision_node[el]]
										loop = True
							new_layer.append(new_decision_node)
						new_decision_nodes.append(new_layer)
						i -= 1
						vtree_node -= 2
					tot_vars = tot_vars[:-1]

			# update size
			size = id_new_node

			# create strs:
			size_str = 'sdd ' + str(size) + '\n'
			literals_str = ''
			for var in literal_nodes.keys():
				for lit in literal_nodes[var]:
					literals_str = literals_str + ' '.join([str(el) for el in lit]) + '\n'
			sinks_str = ''
			for sink in sinks:
				sinks_str = sinks_str + ' '.join([str(el) for el in sink]) + '\n'
			decision_nodes_str = ''
			for layer in new_decision_nodes:
				for node in layer:
					decision_nodes_str = decision_nodes_str + \
					' '.join([str(el) for el in node]) + '\n'
			decision_nodes_str = decision_nodes_str[:-1]

			# read out the new sdd
			file = open(adjusted_sdd_file, 'w')
			file.write(header + size_str + literals_str + sinks_str + decision_nodes_str)
			file.close()

			return adjusted_sdd_file


	@classmethod
	def create_sdd_and_vtree_for_cardinality_constraint(cls, constraint_variables, bound, constraint_type, no_vtree_vars, name, path='./'):
		no_vars = len(constraint_variables)

		#create reservoir of preconstructed SDDs if it does not already exist
		reservoir = os.path.join(path, 'reservoir/')
		if not os.path.isdir(reservoir):
			os.mkdir(reservoir)

		#get unadjusted sdd file
		unadjusted_sddfile = cls._unadjusted_cardinality_constraint_to_sdd(no_vars, bound, constraint_type, reservoir)

		#adjust sdd file
		constraint_variables.sort()
		vtree_variables = [i + 1 for i in range(no_vtree_vars)]
		sddfile = cls._adjust_sdd(vtree_variables, constraint_variables, unadjusted_sddfile, path, name)

		#construct right-linear vtree:
		vtree_filename = str(no_vtree_vars) + '.vtree'
		if vtree_filename not in os.listdir(reservoir):
			vtree = Vtree(var_count = no_vtree_vars, var_order = vtree_variables, vtree_type = 'right')
			vtree.save(bytes(reservoir + vtree_filename, 'utf-8'))
		vtreefile = reservoir + vtree_filename

		return(cls(sddfile, vtreefile))


	@classmethod
	def apply(cls, sdds, apply_type, name, path = './'):

		#check if all vtrees are the same:
		vtreefile = sdds[0].vtreefile
		for sdd in sdds:
			if sdd.vtreefile != vtreefile:
				NotImplementedError("Cannot conjoin SDDs with differing vtrees!")

		#prepare manager
		vtree = Vtree.from_file(bytes(vtreefile, 'utf-8'))
		manager = SddManager(vtree = vtree)
		node = manager.read_sdd_file(bytes(sdds[0].sddfile, 'utf-8'))

		#conjoin/disjoin sdds
		for sdd in sdds[1:]:
			if apply_type == 'conjoin':
				node = node.conjoin(manager.read_sdd_file(bytes(sdd.sddfile, "utf-8")))
			elif apply_type == 'disjoin':
				node = node.disjoin(manager.read_sdd_file(bytes(sdd.sddfile, "utf-8")))
			#garbage collect unused nodes
			node.ref()
			manager.garbage_collect()
			node.deref()

		#write sdd to file and instantiate and return the new SDD
		sddfile = os.path.join(path, name + '.sdd')
		node.save(bytes(sddfile, 'utf-8'))
		return(cls(sddfile, vtreefile))


	def minimize(self, name, path = './'):
		#prepare manager
		vtree = Vtree.from_file(bytes(self.vtreefile, 'utf-8'))
		manager = SddManager(vtree = vtree)
		node = manager.read_sdd_file(bytes(self.sddfile))

		#minimize
		node.ref()
		node.minimize()
		node.deref()

		#write sdd & vtree to file and change filenames
		self.sddfile = os.path.join(path, name + '.sdd')
		node.save(bytes(self.sddfile, 'utf-8'))
		self.vtreefile = os.path.join(path, name + '.vtree')
		vtree = node.vtree()
		vtree.save(bytes(self.vtreefile, 'utf-8'))


	@classmethod
	def from_files(cls, sddfile, vtreefile):
		return(cls(sddfile, vtreefile))



class ResourceConstraintSDD(SDD):

	def __init__(self, sddfile, vtreefile, ents2vars):
		self.ents2vars = ents2vars
		super(ResourceConstraintSDD, self).__init__(sddfile, vtreefile)


	@classmethod
	def _extract_sdd_variables(cls, nentities, nresources, constraints):
		
		# default number of vars: number of resources
		ents2vars = {}
		for i in range(nentities):
			ents2vars[i] = {'no':nresources}

		# reduce number of vars using '<=' and '=' constraints:
		for constraint in constraints:
			if (constraint[1] == 'leq') | (constraint[1] == 'eq'):
				for var in constraint[0]:
					ents2vars[var]['no'] = min(ents2vars[var]['no'], constraint[2])

		# add var numbers:
		counter = 1
		for i in range(nentities):
			ents2vars[i]['vars'] = range(counter, ents2vars[i]['no'] + counter)
			counter += ents2vars[i]['no']

		return ents2vars, counter - 1


	@classmethod
	def _translate_and_check_constraint(cls, constraint, ents2vars):
		variables = list(itertools.chain(*[ents2vars[i]['vars'] for i in constraint[0]]))
		constraint[0] = variables
		if (constraint[1] == 'leq') & (len(constraint[0]) <= constraint[2]):
			constraint = 'trivial case'
		return constraint


	@classmethod
	def calculate_assymsdd_size(cls, no_vars):
		if no_vars == 2:
			size = 5
		else:
			literals = no_vars * 2 - 1
			sinks = 2
			decision_nodes = 2 + 2 * (no_vars - 3) + 1
			size = literals + sinks + decision_nodes
		return size


	@classmethod
	def create_literals_assymsdd(cls, no_vars):
		literals = ''
		for i in range(no_vars):
			if i != no_vars - 1:
				literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
			literals = literals + 'L ' + str(i + no_vars - 1) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
		return literals


	@classmethod
	def create_sinks_assymsdd(cls, no_vars):
		sinks = 'T ' + str(no_vars * 2 - 1) + '\n'
		if no_vars > 2:
			sinks = sinks + 'F ' + str(no_vars * 2) + '\n'
		return sinks


	@classmethod
	def np2pd4assymsdd(cls, decision_nodes):
		if decision_nodes.flags['F_CONTIGUOUS']:
			decision_nodes = np.ascontiguousarray(decision_nodes)
		dtype = decision_nodes.dtype
		dtype = [('ID', dtype), ('vtree', dtype), ('no_kids', dtype), ('tprime', dtype), ('tsub', dtype), ('fprime', dtype), ('fsub', dtype)]
		decision_nodes.dtype = dtype
		decision_nodes[::-1].sort(0, order = 'vtree')
		decision_nodes = decision_nodes[['ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']]
		decision_nodes = pd.DataFrame(decision_nodes.flatten())
		decision_nodes['prefix'] = 'D'
		return decision_nodes


	@classmethod
	def construct_intermediate_decision_nodes(cls, no_vars, count, vtree_node):
		decision_nodes = np.zeros([7, no_vars - 3, 2], dtype = int)
		grid = np.mgrid[0:no_vars - 3,0:2]

		#set id:
		decision_nodes[0] = count + grid[0] * 2 + grid[1]
		#set vtree:
		decision_nodes[1] = vtree_node - grid[0] * 2
		#set number of elements:
		decision_nodes[2] = 2
		#set true prime:
		decision_nodes[3] = no_vars - 3 - grid[0]
		#set true sub:
		decision_nodes[4] = 2 * no_vars + 2 * (grid[0] + 1) - 1
		decision_nodes[4][:, 1] = no_vars * 2
		#set false prime:
		decision_nodes[5] = 2 * no_vars - 4 - grid[0]
		#set false sub:
		decision_nodes[6] = 2 * no_vars + 2 * (grid[0] + 1)

		# reshaping the array so that  info for each decision node is represented by 1 row
		decision_nodes = np.reshape(decision_nodes.T, (2 * (no_vars - 3), 7), order = 'C')
		# configuring the numpy array into a sorted pandas dataframe:
		decision_nodes = cls.np2pd4assymsdd(decision_nodes)

		return decision_nodes


	@classmethod
	def create_decision_nodes_assymsdd(cls, no_vars):
		if no_vars == 2:
			leaf_layer = 'D 4 1 2 0 3 1 2\n'
			intermediate_layers = 'no_vars is 2'
			root_layer = 'no_vars is 2'


		else:
			# add final decision nodes:
			count = no_vars * 2 + 1
			vtree_node = no_vars * 2 - 3
			leaf_layer = 'D ' + str(count) + ' ' + str(vtree_node) + ' ' + str(2) + \
			' ' + str(no_vars - 2) + ' ' + str(no_vars * 2 - 1) + ' ' + \
			str(2 * no_vars - 3) + ' ' + str(2 * no_vars - 2) + '\n' + \
			'D ' + str(count + 1) + ' ' + str(vtree_node) + ' ' + str(2) + ' ' + \
			str(no_vars - 2) + ' ' + str(no_vars * 2) + ' ' + str(2 * no_vars - 3) + \
			' ' + str(2 * no_vars - 2) + '\n'
			count += 2
			vtree_node -= 2

			# add intermediate layers:
			if no_vars > 3:
				intermediate_layers = cls.construct_intermediate_decision_nodes(no_vars, count, vtree_node)
				count = count + 2 * (no_vars - 3)
			else:
				intermediate_layers = 'no_vars is 3'

			# add root layer:
			root_layer = 'D ' + str(count) + ' 1 2 0 ' + \
			str(count - 2) + ' ' + str(no_vars - 1) + ' ' + str(count - 1)

		return leaf_layer, intermediate_layers, root_layer


	@classmethod
	def symmetry_breaking_sdd(cls, no_vars, path):

		# check if exists
		sddfile = path + 'symmetry_breaker' + '-' + str(no_vars) + '.sdd'
		files = os.listdir(path)

		if sddfile[len(path):] in files:
			return sddfile

		else:
			#create header
			header = 'c ids of sdd nodes start at 0\nc sdd nodes appear bottom-up, children before parents\nc\nc file syntax:\nc sdd count-of-sdd-nodes\nc F id-of-false-sdd-node\nc T id-of-true-sdd-node\nc L id-of-literal-sdd-node id-of-vtree literal\nc D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*\nc\n'

			#create sdd-size row:
			sdd_size = cls.calculate_assymsdd_size(no_vars)
			sdd_size_str = 'sdd ' + str(sdd_size) + '\n'

			#create rows for literals:
			literals = cls.create_literals_assymsdd(no_vars)

			#create rows for sinks:
			sinks = cls.create_sinks_assymsdd(no_vars)

			#create rows for decision nodes:
			leaf_layer, intermediate_layers, root_layer = cls.create_decision_nodes_assymsdd(no_vars)

			#write everyting to file and returning file name:
			file = open(sddfile, 'w')
			file.write(header + sdd_size_str + literals + sinks + leaf_layer)
			file.close()
			if isinstance(intermediate_layers, str) == False:
				col_order = ['prefix', 'ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']
				intermediate_layers[col_order].to_csv(sddfile, mode = 'a', sep = ' ', header = False, index = False, quoting=csv.QUOTE_NONE)
			if root_layer != 'no_vars is 2':
				file = open(sddfile, 'a')
				file.write(root_layer)
				file.close()

			return sddfile


	@classmethod
	def _construct_symmetry_breaking_sdd(cls, entity, ents2vars, tot_vars, name, path):
		#construct symmetry breaker
		no_vars = ents2vars[entity]['no']
		if no_vars == 1:
			return "trivial case"
		else:
			reservoir = path + 'reservoir/'
			filename = cls.symmetry_breaking_sdd(no_vars, reservoir)
		
			#adjust the sdd
			adjusted_sdd_file = SDD._adjust_sdd(range(1, tot_vars + 1), ents2vars[entity]['vars'], filename, path, name)

			return adjusted_sdd_file


	@classmethod
	def create_sdd_and_vtree_for_resource_constraints(cls, nentities, nresources, resource_constraints, name, path = './'):
		if type(resource_constraints) == dict:
			_, resource_constraints = cls.translate_constraint_dicts_into_constraint_lists(resource_constraints)
		
		#obtain entitity to variable mapping and total number of variables, save the former:
		ents2vars, tot_vars = cls._extract_sdd_variables(nentities, nresources, resource_constraints)
		pklfile = open(os.path.join(path, 'ents2vars-' + name + '.pkl'), 'ab')
		pickle.dump(ents2vars, pklfile)
		pklfile.close()
		
		#construct right-linear vtree:
		variables = list(range(1, tot_vars + 1, 1))
		vtree = Vtree(var_count = tot_vars, var_order = variables, vtree_type = 'right')
		vtree_filename = os.path.join(path, name + '.vtree')
		vtree.save(bytes(vtree_filename, 'utf-8'))

		#construct global sum constraint sdd:
		global_sum_constraint = SDD.create_sdd_and_vtree_for_cardinality_constraint(variables, nresources, 'eq', tot_vars, name, path)

		#construct constraint sdds:
		sdds = [global_sum_constraint]
		i = 0
		for constraint in resource_constraints:
			constraint = cls._translate_and_check_constraint(constraint, ents2vars)
			if isinstance(constraint, str) == False:
				sdd = SDD.create_sdd_and_vtree_for_cardinality_constraint(constraint[0], constraint[2], constraint[1], tot_vars, name + "-subconstraint" + str(i) , path)
				sdds.append(sdd)
				i += 1
		
		#construct symmetry breaking constraints
		name_original = name
		for entity in range(nentities):
			name = name_original + '-symmetry_breaker' + str(entity)
			assymsdd_filename = cls._construct_symmetry_breaking_sdd(entity, ents2vars, tot_vars, name, path = './')
			if assymsdd_filename != 'trivial case':
				sdd = SDD.from_files(assymsdd_filename, vtree_filename)
				sdds.append(sdd)

		resource_constraint_sdd = SDD.apply(sdds, 'conjoin', name_original, path)

		return(cls(resource_constraint_sdd.sddfile, resource_constraint_sdd.vtreefile, pklfile))


	@staticmethod
	def translate_constraint_dicts_into_constraint_lists(node):
		constraints = []
		if 'children' not in node.keys():
		    zone_ids = set([node['zone_id']])
		elif len(node['children']) == 0:
		    zone_ids = set([node['zone_id']])
		else:
			zone_ids = set([])
			for child in node['children']:
				new_zone_ids, new_constraints = translate_constraints(child)
				zone_ids = zone_ids.union(new_zone_ids)
				constraints += new_constraints
		if node['min'] is not None:
			if node['min'] > 0:
				constraints.append([list(zone_ids), 'geq', int(node['min'])])
		if node['max'] is not None:
		    constraints.append([list(zone_ids), 'leq', int(node['max'])])
		if node['equals'] is not None:
		    constraints.append([list(zone_ids), 'eq', int(node['equals'])])
		return zone_ids, constraints



class PSDD:

	def __init__(self, psddfile, vtreefile):
		self.psddfile = psddfile
		self.vtreefile = vtreefile


	@classmethod
	def from_files(cls, psddfile, vtreefile):
		return(cls(psddfile, vtreefile))


	@classmethod
	def conjoin_sdds_to_psdd(cls, vtreefile, sdds, name, path = './'):
		cur_dir = subprocess.check_output(['pwd']).decode("utf-8").strip()
		print(cur_dir)
		# loc_conjoin_sdds2psdd = cur_dir + '/psdd/mult_sdd2psdd'
		loc_conjoin_sdds2psdd =  '../psdd/mult_sdd2psdd'
		psdd_dir = path
		psddfile = psdd_dir + name + '.psdd'
		command = [loc_conjoin_sdds2psdd, vtreefile]
		for sdd in sdds:
			command.append(sdd.sddfile)
		command.append(psddfile)
		print(command)
		subprocess.run(command)
		return(cls(psddfile, vtreefile))



class ResourceConstraintPSDD(PSDD, ResourceConstraintSDD):

	def __init__(self, psddfile, vtreefile, ents2vars):
		self.ents2vars = ents2vars
		self.psddfile = psddfile
		self.vtreefile = vtreefile


	@classmethod
	def create_psdd_and_vtree_for_resource_constraints(cls, nentities, nresources, resource_constraints, name, path = './'):
		if type(resource_constraints) == dict:
			_, resource_constraints = cls.translate_constraint_dicts_into_constraint_lists(resource_constraints)
		
		#obtain entitity to variable mapping and total number of variables, save the former:
		ents2vars, tot_vars = cls._extract_sdd_variables(nentities, nresources, resource_constraints)
		pklfile = open(os.path.join(path, 'ents2vars-' + name + '.pkl'), 'ab')
		pickle.dump(ents2vars, pklfile)
		pklfile.close()
		
		#construct right-linear vtree:
		variables = list(range(1, tot_vars + 1, 1))
		vtree = Vtree(var_count = tot_vars, var_order = variables, vtree_type = 'right')
		vtree_filename = os.path.join(path, name + '.vtree')
		vtree.save(bytes(vtree_filename, 'utf-8'))

		#construct global sum constraint sdd:
		global_sum_constraint = SDD.create_sdd_and_vtree_for_cardinality_constraint(variables, nresources, 'eq', tot_vars, name, path)

		#construct constraint sdds:
		sdds = [global_sum_constraint]
		i = 0
		for constraint in resource_constraints:
			constraint = cls._translate_and_check_constraint(constraint, ents2vars)
			if isinstance(constraint, str) == False:
				sdd = SDD.create_sdd_and_vtree_for_cardinality_constraint(constraint[0], constraint[2], constraint[1], tot_vars, name + "-subconstraint" + str(i) , path)
				sdds.append(sdd)
				i += 1
		
		#construct symmetry breaking constraints
		name_original = name
		for entity in range(nentities):
			name = name_original + '-symmetry_breaker' + str(entity)
			assymsdd_filename = cls._construct_symmetry_breaking_sdd(entity, ents2vars, tot_vars, name, path = './')
			if assymsdd_filename != 'trivial case':
				sdd = SDD.from_files(assymsdd_filename, vtree_filename)
				sdds.append(sdd)


		resource_constraint_psdd = PSDD.conjoin_sdds_to_psdd(vtree_filename, sdds, name_original, path)

		return(cls(resource_constraint_psdd.psddfile, resource_constraint_psdd.vtreefile, pklfile))

