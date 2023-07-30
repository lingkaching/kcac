from psddcomp4acrl import *
from pysdd.sdd import Vtree, SddManager
from scipy.special import comb

## TEST 1:
# create SDD and vtree for cardinality constraint creates sth succesfully?
sdd = SDD.create_sdd_and_vtree_for_cardinality_constraint([1,3,2,6,4], 2, 'leq', 8, "test1", path='./')
# vtreefile can be read in by PySDD package?
vtree = Vtree.from_file(bytes(sdd.vtreefile, 'utf-8'))
# sdd can be read in?
manager = SddManager(vtree = vtree,  auto_gc_and_minimize = False)
node = manager.read_sdd_file(bytes(sdd.sddfile, 'utf-8'))
# model count before symmetry breaking is correct?
model_count_constructed_sdd = node.global_model_count()
correct_model_count = (2 ** 3) * sum(comb(5, i) for i in range(2 + 1))
print(model_count_constructed_sdd)
print(correct_model_count)
assert(model_count_constructed_sdd == correct_model_count)
# check all models whether they satisfy the constraint
models = node.models()
for model in models:
	summing_up = 0
	for i in [1,3,2,6,4]:
		summing_up += model[i]
	assert summing_up <= 2


## TEST 2 and TEST 3
# create resource constraint SDD and vtree succesfully?
resource_constraint_sdd =  ResourceConstraintSDD.create_sdd_and_vtree_for_resource_constraints(10, 15, [[[1,2], 'leq', 2], [[3,4,5], 'leq', 2], [[6,7,8,9], 'leq', 3], [[6,7,8,9], 'geq', 1]], 'test2', path = './')

# small toy example:
resource_constraint_sdd =  ResourceConstraintSDD.create_sdd_and_vtree_for_resource_constraints(3, 4, [[[0], 'leq', 2], [[1], 'leq', 2], [[2], 'leq', 3], [[0,1], 'geq', 3]], 'test3a', path = './')
# vtreefile can be read in by PySDD package?
vtree = Vtree.from_file(bytes(resource_constraint_sdd.vtreefile, 'utf-8'))
# sdd can be read in?
manager = SddManager(vtree = vtree,  auto_gc_and_minimize = False)
node = manager.read_sdd_file(bytes(resource_constraint_sdd.sddfile, 'utf-8'))
# model count is correct?
model_count_constructed_sdd = node.global_model_count()
assert(model_count_constructed_sdd == 3)
# check models
models = node.models()
for model in models:
	print(model)
# correct!


# small toy example 2:
resource_constraint_sdd =  ResourceConstraintSDD.create_sdd_and_vtree_for_resource_constraints(3, 4, [[[0], 'leq', 2], [[1], 'leq', 2], [[2], 'leq', 3]], 'test3b', path = './')
# vtreefile can be read in by PySDD package?
vtree = Vtree.from_file(bytes(resource_constraint_sdd.vtreefile, 'utf-8'))
# sdd can be read in?
manager = SddManager(vtree = vtree,  auto_gc_and_minimize = False)

node = manager.read_sdd_file(bytes(resource_constraint_sdd.sddfile, 'utf-8'))
# model count is correct?
model_count_constructed_sdd = node.global_model_count()
assert(model_count_constructed_sdd == 8)
# check models
models = node.models()
for model in models:
	print(model)
#correct!

#TEST 4 (PSDD)
groups = [[0, 1, 5, 6, 7], [2, 3, 4, 8, 9], [10, 15, 20], [11, 12, 16, 17], [13, 14, 18, 19, 24], [21, 22, 23]]
upper_bound = [20, 20, 12, 16, 20, 12]
lower_bound = [6, 6, 4, 5, 6, 4] 
local_upper_bound = 4
local_lower_bound = 0

constraints = []
cnt = 0
for group in groups:
	constraints.append([group, 'leq', upper_bound[cnt]])
	constraints.append([group, 'geq', lower_bound[cnt]])
	cnt += 1
for i in range(25):
	constraints.append([[i], 'leq', local_upper_bound])
	# add the below constraint if local_lower_bound is greater than zero
	# constraints.append([[i], 'geq', local_lower_bound])
resource_constraint_sdd =  ResourceConstraintPSDD.create_sdd_and_vtree_for_resource_constraints(25, 32, constraints, 'test4', path = './')
