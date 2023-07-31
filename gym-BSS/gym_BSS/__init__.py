from gym.envs.registration import register
from math import floor

def build_constraints_2_level_simple(nzones, zone_ids_per_group, group_mins, group_maxs, local_min, local_max, nresources=1):
    constraints = {
        "name": "root",
        "equals": nresources,
        "min": nresources,
        "max": nresources,
        "children": []
    }
    zones_consumed_set = set()
    for zone_ids, group_min, group_max, group_index in zip(zone_ids_per_group, group_mins, group_maxs, range(len(zone_ids_per_group))):
        group = {
            "name": "group{0}".format(group_index),
            "min": group_min,
            "max": group_max,
            "equals": None,
            "children": []
        }
        for zone_id, group_zone_index in zip(zone_ids, range(len(zone_ids))):
            assert zone_id not in zones_consumed_set, "Something wrong. zone_id {0} cannot be part of more than one group".format(
                zone_id)
            zone = {
                "name": "G{0}Z{1}".format(group_index, group_zone_index),
                "min": local_min,
                "max": local_max,
                "equals": None,
                "zone_id": zone_id
            }
            zones_consumed_set.add(zone_id)
            group['children'].append(zone)
        constraints['children'].append(group)
    assert len(
        zones_consumed_set) == nzones, "All zone_ids need to be specified as part of some group"
    return constraints

# unconstrained bss envs

demands = ['actual-data-art', 'actual-poisson-art', 'actual-poisson-OD-art']
scens_train = [list(range(1, 21)), list(range(0, 30)), list(range(0, 30))]
scens_test = [list(range(21, 61)), list(range(30, 100)), list(range(30, 100))]

zones_5 = [[0,1,2,3,4]]
lower_5 = [0]
upper_5_100 = [100]
upper_5_150 = [150]
upper_5_200 = [200]

for j in range(len(demands)):
    constraints = build_constraints_2_level_simple(5, zones_5, lower_5, upper_5_100, 0, 23, 100)
    register(
        id='BSSEnv-5zones-100bikes-' + demands[j] + '-v0',
        entry_point='gym_BSS.envs:BSSEnv',
        kwargs={
            'nzones': 5,
            'data_set_name': demands[j],
            'scenarios': scens_train[j],
            'data_dir': 'default_data-5zones-100bikes-actual-data-art',
            'constraints': constraints
        }
    )

    register(
        id='BSSEnvTest-5zones-100bikes-' + demands[j] + '-v0',
        entry_point='gym_BSS.envs:BSSEnv',
        kwargs={
            'nzones': 5,
            'data_set_name': demands[j],
            'scenarios': scens_test[j],
            'data_dir': 'default_data-5zones-100bikes-actual-data-art',
            'constraints': constraints
        }
    )
    constraints = build_constraints_2_level_simple(5, zones_5, lower_5, upper_5_200, 0, 47, 200)
    register(
        id='BSSEnv-5zones-200bikes-' + demands[j] + '-v0',
        entry_point='gym_BSS.envs:BSSEnv',
        kwargs={
            'nzones': 5,
            'data_set_name': demands[j],
            'scenarios': scens_train[j],
            'data_dir': 'default_data-5zones-200bikes-actual-data-art',
            'constraints': constraints
        }
    )

    register(
        id='BSSEnvTest-5zones-200bikes-' + demands[j] + '-v0',
        entry_point='gym_BSS.envs:BSSEnv',
        kwargs={
            'nzones': 5,
            'data_set_name': demands[j],
            'scenarios': scens_test[j],
            'data_dir': 'default_data-5zones-200bikes-actual-data-art',
            'constraints': constraints
        }
    )
    constraints = build_constraints_2_level_simple(5, zones_5, lower_5, upper_5_150, 0, 35, 150)
    register(
        id='BSSEnv-10zones-150bikes-' + demands[j] + '-v0',
        entry_point='gym_BSS.envs:BSSEnv',
        kwargs={
            'nzones': 10,
            'data_set_name': demands[j],
            'scenarios': scens_train[j],
            'data_dir': 'default_data-10zones-150bikes-actual-data-art'
            'constraints': constraints
        }
    )

    register(
        id='BSSEnvTest-10zones-150bikes-' + demands[j] + '-v0',
        entry_point='gym_BSS.envs:BSSEnv',
        kwargs={
            'nzones': 10,
            'data_set_name': demands[j],
            'scenarios': scens_test[j],
            'data_dir': 'default_data-10zones-150bikes-actual-data-art'
            'constraints': constraints
        }
    )



# env tight
zones_15 = [[4, 10, 12], [3, 1, 14, 7], [0, 5, 9, 11], [2, 6, 8, 13]]
bikes = [20]
lowers_15 = [[floor(3*i/15), floor(4*i/15), floor(4*i/15), floor(4*i/15)] for i in bikes]
local_maxs_15 = [floor(i*2/15) for i in bikes]
uppers_15 = [[max(i), max(i), max(i), max(i)] for i in lowers_15]

for i in range(len(bikes)):
    constraints = build_constraints_2_level_simple(15, zones_15, lowers_15[i], uppers_15[i], 0, local_maxs_15[i], bikes[i])

    demands = ['actual-data-art', 'actual-poisson-art', 'actual-poisson-OD-art']
    scens_train = [list(range(1, 21)), list(range(0, 30)), list(range(0, 30))]
    scens_test = [list(range(21, 61)), list(range(30, 100)), list(range(30, 100))]
    for j in range(len(demands)):
        register(
            id='BSSEnv-15zones-' + str(bikes[i]) + 'bikes-constraints100_' + str(uppers_15[i][0]) + '-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 15,
                'data_set_name': demands[j],
                'scenarios': scens_train[j],
                'data_dir': 'default_data-15zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

        register(
            id='BSSEnvTest-15zones-' + str(bikes[i]) + 'bikes-constraints100_' + str(uppers_15[i][0]) + '-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 15,
                'data_set_name': demands[j],
                'scenarios': scens_test[j],
                'data_dir': 'default_data-15zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

zones_20 = [[5, 16, 2, 3, 14], [7, 1, 19, 13], [17, 8, 9, 4], [10, 15, 12], [11, 18, 0, 6]]
no_zones = 20
bikes = [27]
lowers_20 = [[floor(5*i/no_zones), floor(4*i/no_zones), floor(4*i/no_zones), floor(3*i/no_zones), floor(4*i/no_zones)] for i in bikes]
local_maxs_20 = [floor(i*2/no_zones) for i in bikes]
uppers_20 = [[max(i), max(i), max(i), max(i), max(i)] for i in lowers_20]

for i in range(len(bikes)):
    constraints = build_constraints_2_level_simple(20, zones_20, lowers_20[i], uppers_20[i], 0, local_maxs_20[i], bikes[i])

    demands = ['actual-data-art', 'actual-poisson-art', 'actual-poisson-OD-art']
    scens_train = [list(range(1, 21)), list(range(0, 30)), list(range(0, 30))]
    scens_test = [list(range(21, 61)), list(range(30, 100)), list(range(30, 100))]
    for j in range(len(demands)):
        register(
            id='BSSEnv-20zones-' + str(bikes[i]) + 'bikes-constraints100_' + str(uppers_20[i][0]) + '-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 20,
                'data_set_name': demands[j],
                'scenarios': scens_train[j],
                'data_dir': 'default_data-20zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

        register(
            id='BSSEnvTest-20zones-' + str(bikes[i]) + 'bikes-constraints100_' + str(uppers_20[i][0]) + '-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 20,
                'data_set_name': demands[j],
                'scenarios': scens_test[j],
                'data_dir': 'default_data-20zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

zones_30 = [[6, 18, 15], [20, 1, 19, 3], [12, 24, 28, 21], [29, 10, 23, 9, 13], [14, 8, 25, 27], [4, 26, 0, 16], [7, 17, 5], [2, 11, 22]]
no_zones = 30
bikes = [40]
lowers_30 = [[floor(3*i/no_zones), floor(4*i/no_zones), floor(4*i/no_zones), floor(5*i/no_zones), floor(4*i/no_zones), floor(4*i/no_zones), floor(3*i/no_zones), floor(3*i/no_zones)] for i in bikes]
local_maxs_30 = [floor(i*2/no_zones) for i in bikes]
uppers_30 = [[max(i), max(i), max(i), max(i), max(i), max(i), max(i), max(i)] for i in lowers_30]

for i in range(len(bikes)):
    constraints = build_constraints_2_level_simple(30, zones_30, lowers_30[i], uppers_30[i], 0, local_maxs_30[i], bikes[i])

    demands = ['actual-data-art', 'actual-poisson-art', 'actual-poisson-OD-art']
    scens_train = [list(range(1, 21)), list(range(0, 30)), list(range(0, 30))]
    scens_test = [list(range(21, 61)), list(range(30, 100)), list(range(30, 100))]
    for j in range(len(demands)):
        register(
            id='BSSEnv-30zones-' + str(bikes[i]) + 'bikes-constraints100_' + str(uppers_30[i][0]) + '-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 30,
                'data_set_name': demands[j],
                'scenarios': scens_train[j],
                'data_dir': 'default_data-30zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

        register(
            id='BSSEnvTest-30zones-' + str(bikes[i]) + 'bikes-constraints100_' + str(uppers_30[i][0]) + '-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 30,
                'data_set_name': demands[j],
                'scenarios': scens_test[j],
                'data_dir': 'default_data-30zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

# constrained envs 15 zones
zones_15 = [[4, 10, 12], [3, 1, 14, 7], [0, 5, 9, 11], [2, 6, 8, 13]]
bikes = [16, 18, 20, 22, 24, 26]
lowers_15 = [[floor(3*i/15), floor(4*i/15), floor(4*i/15), floor(4*i/15)] for i in bikes]
local_maxs_15 = [floor(i*2/15) for i in bikes]
uppers_15 = [[3*i, 4*i, 4*i, 4*i] for i in local_maxs_15]
for i in range(len(bikes)):
    constraints = build_constraints_2_level_simple(15, zones_15, lowers_15[i], uppers_15[i], 0, local_maxs_15[i], bikes[i])

    demands = ['actual-data-art', 'actual-poisson-art', 'actual-poisson-OD-art']
    scens_train = [list(range(1, 21)), list(range(0, 30)), list(range(0, 30))]
    scens_test = [list(range(21, 61)), list(range(30, 100)), list(range(30, 100))]
    for j in range(len(demands)):
        register(
            id='BSSEnv-15zones-' + str(bikes[i]) + 'bikes-constraints100-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 15,
                'data_set_name': demands[j],
                'scenarios': scens_train[j],
                'data_dir': 'default_data-15zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )

        register(
            id='BSSEnvTest-15zones-' + str(bikes[i]) + 'bikes-constraints100-' + demands[j] + '-v0',
            entry_point='gym_BSS.envs:BSSEnv',
            kwargs={
                'nzones': 15,
                'data_set_name': demands[j],
                'scenarios': scens_test[j],
                'data_dir': 'default_data-15zones-' + str(bikes[i]) + 'bikes-actual-data-art',
                'constraints': constraints
            }
        )





# very constrained envs
zones_10 = [[5, 3, 7, 1], [0, 8, 4], [9, 2, 6]]
zones_15 = [[4, 10, 12], [3, 1, 14, 7], [0, 5, 9, 11], [2, 6, 8, 13]]
zones_15_7g = [[4, 10],[12, 7],[3, 14],[1, 8],[6, 2],[13, 5], [0, 9, 11]]

# zones_20 = [[], [], [], [], []]

# lower_10_50 = []
lower_10_75 = [3, 2, 2]
lower_10_100 = [4, 3, 3]
# lower_15_50 = []
# lower_15_75 = []
lower_15_100 = [3, 4, 4, 4]
lower_15_7g_100 = [2, 2, 2, 2, 2, 2, 3]
# lower_20_50 = []
# lower_20_75 = []
# lower_20_100 = []

upper_10 = [2*4, 2*3, 2*3]
upper_15 = [2*3, 2*4, 2*4, 2*4]
upper_15_7g_3 = [3, 3, 3, 3, 3, 3, 3]
#upper_20 = [2*, 2*, 2*, 2*, 2*]

upper_10_5 = [5, 5, 5]
upper_10_4 = [4, 4, 4]

# constraints_10_50 = build_constraints_2_level_simple(10, zones_10, lower_10_50, upper_10, 0, 2, 10)
# constraints_10_75 = build_constraints_2_level_simple(10, zones_10, lower_10_75, upper_10, 0, 2, 10)
constraints_10_75_4 = build_constraints_2_level_simple(10, zones_10, lower_10_75, upper_10_4, 0, 2, 10)
constraints_10_100 = build_constraints_2_level_simple(10, zones_10, lower_10_100, upper_10, 0, 2, 10)
constraints_10_100_4 = build_constraints_2_level_simple(10, zones_10, lower_10_100, upper_10_4, 0, 2, 10)
# constraints_15_50 = build_constraints_2_level_simple(15, zones_15, lower_15_50, upper_15, 0, 2, 15)
# constraints_15_75 = build_constraints_2_level_simple(15, zones_15, lower_15_75, upper_15, 0, 2, 15)
constraints_15_100 = build_constraints_2_level_simple(15, zones_15, lower_15_100, upper_15, 0, 2, 15)
constraints_15_7g_100_3_l3 = build_constraints_2_level_simple(15, zones_15_7g, lower_15_7g_100, upper_15_7g_3, 0, 3, 15)
# constraints_20_50 = build_constraints_2_level_simple(20, zones_20, lower_20_50, upper_20, 0, 2, 20)
# constraints_20_75 = build_constraints_2_level_simple(20, zones_20, lower_20_75, upper_20, 0, 2, 20)
# constraints_20_100 = build_constraints_2_level_simple(20, zones_20, lower_20_100, upper_20, 0, 2, 20)

# register(
#     id='BSSEnv-10zones-10bikes-constraints50-actual-data-art-v0',
#     entry_point='gym_BSS.envs:BSSEnv',
#     kwargs={
#         'nzones': 10,
#         'data_set_name': 'actual-data-art',
#         'scenarios': list(range(1, 21)),
#         'data_dir': 'default_data-10zones-10bikes-actual-data-art',
#         'constraints': constraints_10_50
#     }
# )

# register(
#     id='BSSEnvTest-10zones-10bikes-constraints50-actual-data-art-v0',
#     entry_point='gym_BSS.envs:BSSEnv',
#     kwargs={
#         'nzones': 10,
#         'data_set_name': 'actual-data-art',
#         'scenarios': list(range(21, 61)),
#         'data_dir': 'default_data-10zones-10bikes-actual-data-art',
#         'constraints': constraints_10_50
#     }
# )

register(
    id='BSSEnv-10zones-10bikes-constraints75_4-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-10bikes-actual-data-art',
        'constraints': constraints_10_75_4
    }
)

register(
    id='BSSEnvTest-10zones-10bikes-constraints75_4-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-10bikes-actual-data-art',
        'constraints': constraints_10_75_4
    }
)

register(
    id='BSSEnv-10zones-10bikes-constraints100_4-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-10bikes-actual-data-art',
        'constraints': constraints_10_100_4
    }
)

register(
    id='BSSEnvTest-10zones-10bikes-constraints100_4-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-10bikes-actual-data-art',
        'constraints': constraints_10_100_4
    }
)

register(
    id='BSSEnv-10zones-10bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-10bikes-actual-data-art',
        'constraints': constraints_10_100
    }
)

register(
    id='BSSEnvTest-10zones-10bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-10bikes-actual-data-art',
        'constraints': constraints_10_100
    }
)


register(
    id='BSSEnv-15zones-15bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 15,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-15zones-15bikes-actual-data-art',
        'constraints': constraints_15_100
    }
)

register(
    id='BSSEnvTest-15zones-15bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 15,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-15zones-15bikes-actual-data-art',
        'constraints': constraints_15_100
    }
)

register(
    id='BSSEnv-15zones-15bikes-constraints7g_100_3_l3-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 15,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-15zones-15bikes-actual-data-art',
        'constraints': constraints_15_7g_100_3_l3
    }
)

register(
    id='BSSEnvTest-15zones-15bikes-constraints7g_100_3_l3-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 15,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-15zones-15bikes-actual-data-art',
        'constraints': constraints_15_7g_100_3_l3
    }
)
## more envs 

# zonings based on closeness of station centres
zones_10 = [[6, 2, 9], [0, 8, 4], [1, 5, 7, 3]] #based on 10-38
zones_14 = [[1, 8, 5, 12, 3, 11], [6, 7, 0, 9], [13, 2, 4, 10]] #based on 14-57
zones_5 = [[0,1,2,3,4]]

lower_10_50 = [6, 6, 8]
lower_14_50 = [10, 8, 8]
lower_10_75 = [9, 9, 11]
lower_14_75 = [15, 12, 12]
lower_10_100 = [11, 11, 15]
lower_14_100 = [20, 16, 16]
lower_5 = [0]

upper_5_19 = [9*5]
upper_5_150 = [35*5]
upper_10 = [8*3, 8*3, 8*4]
upper_14 = [11*5, 11*4, 11*4]

constraints_5_19 = build_constraints_2_level_simple(5, zones_5, lower_5, upper_5_19, 0, 9, 19)
constraints_5_150 = build_constraints_2_level_simple(5, zones_5, lower_5, upper_5_150, 0, 35, 150)
constraints_10_50 = build_constraints_2_level_simple(10, zones_10, lower_10_50, upper_10, 0, 8, 38)
constraints_14_50 = build_constraints_2_level_simple(14, zones_14, lower_14_50, upper_14, 0, 11, 57)
constraints_10_75 = build_constraints_2_level_simple(10, zones_10, lower_10_75, upper_10, 0, 8, 38)
constraints_14_75 = build_constraints_2_level_simple(14, zones_14, lower_14_75, upper_14, 0, 11, 57)
constraints_10_100 = build_constraints_2_level_simple(10, zones_10, lower_10_100, upper_10, 0, 8, 38)
constraints_14_100 = build_constraints_2_level_simple(14, zones_14, lower_14_100, upper_14, 0, 11, 57)


# 5%
#default_data-5zones-19bikes-actual-data-art
register(
    id='BSSEnv-5zones-19bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 5,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-5zones-19bikes-actual-data-art',
        'constraints': constraints_5_19
    }
)

register(
    id='BSSEnvTest-5zones-19bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 5,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-5zones-19bikes-actual-data-art',
        'constraints': constraints_5_19
    }
)

# 5%
#default_data-5zones-150bikes-actual-data-art
register(
    id='BSSEnv-5zones-150bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 5,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-5zones-150bikes-actual-data-art',
        'constraints': constraints_5_150
    }
)

register(
    id='BSSEnvTest-5zones-150bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 5,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-5zones-150bikes-actual-data-art',
        'constraints': constraints_5_150
    }
)

# 10%
#default_data-10zones-76bikes-actual-data-art
register(
    id='BSSEnv-10zones-38bikes-constraints50-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-38bikes-actual-data-art',
        'constraints': constraints_10_50
    }
)

register(
    id='BSSEnvTest-10zones-38bikes-constraints50-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-38bikes-actual-data-art',
        'constraints': constraints_10_50
    }
)

register(
    id='BSSEnv-10zones-38bikes-constraints75-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-38bikes-actual-data-art',
        'constraints': constraints_10_75
    }
)

register(
    id='BSSEnvTest-10zones-38bikes-constraints75-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-38bikes-actual-data-art',
        'constraints': constraints_10_75
    }
)

register(
    id='BSSEnv-10zones-38bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-38bikes-actual-data-art',
        'constraints': constraints_10_100
    }
)

register(
    id='BSSEnvTest-10zones-38bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-38bikes-actual-data-art',
        'constraints': constraints_10_100
    }
)
# 15%
#default_data-14zones-114bikes-actual-data-art
register(
    id='BSSEnv-14zones-57bikes-constraints50-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-14zones-57bikes-actual-data-art',
        'constraints': constraints_14_50
    }
)

register(
    id='BSSEnvTest-14zones-57bikes-constraints50-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-14zones-57bikes-actual-data-art',
        'constraints': constraints_14_50
    }
)

register(
    id='BSSEnv-14zones-57bikes-constraints75-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-14zones-57bikes-actual-data-art',
        'constraints': constraints_14_75
    }
)

register(
    id='BSSEnvTest-14zones-57bikes-constraints75-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-14zones-57bikes-actual-data-art',
        'constraints': constraints_14_75
    }
)

register(
    id='BSSEnv-14zones-57bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-14zones-57bikes-actual-data-art',
        'constraints': constraints_14_100
    }
)

register(
    id='BSSEnvTest-14zones-57bikes-constraints100-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-14zones-57bikes-actual-data-art',
        'constraints': constraints_14_100
    }
)


## fractional envs (fractions of full env)

# 5%
#default_data-5zones-38bikes-actual-data-art
register(
    id='BSSEnv-5zones-38bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 5,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-5zones-38bikes-actual-data-art',
    }
)

# 10%
#default_data-10zones-76bikes-actual-data-art
register(
    id='BSSEnv-10zones-76bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-10zones-76bikes-actual-data-art',
    }
)

register(
    id='BSSEnvTest-10zones-76bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 10,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-10zones-76bikes-actual-data-art',
    }
)

# 15%
#default_data-14zones-114bikes-actual-data-art
register(
    id='BSSEnv-14zones-114bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-14zones-114bikes-actual-data-art',
    }
)

register(
    id='BSSEnvTest-14zones-114bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 14,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-14zones-114bikes-actual-data-art',
    }
)


# 20%
#default_data-19zones-152bikes-actual-data-art
register(
    id='BSSEnv-19zones-152bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 19,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-19zones-152bikes-actual-data-art',
    }
)

# 25%
#default_data-24zones-190bikes-actual-data-art
register(
    id='BSSEnv-24zones-190bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 24,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-24zones-190bikes-actual-data-art',
    }
)

# 50%
#default_data-48zones-380bikes-actual-data-art
register(
    id='BSSEnv-48zones-380bikes-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 48,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-48zones-380bikes-actual-data-art',
    }
)

#3zones-0.05average-actual-data-art
constraints3_10_v0 = build_constraints_2_level_simple(3, [[1, 2], [0]], [4, 2], [9, 7], 0, 8, 10)

register(
    id='BSSEnv-3zones-0.05average-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 3,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-3zones-0.05average-actual-data-art'
    }
)

register(
    id='BSSEnvTest-3zones-0.05average-actual-data-art-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 3,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-3zones-0.05average-actual-data-art'
    }
)

register(
    id='BSSEnv-3zones-0.05average-actual-data-art-constraints0-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 3,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21)),
        'data_dir': 'default_data-3zones-0.05average-actual-data-art',
        'constraints': constraints3_10_v0
    }
)

register(
    id='BSSEnvTest-3zones-0.05average-actual-data-art-constraints0-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'nzones': 3,
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61)),
        'data_dir': 'default_data-3zones-0.05average-actual-data-art',
        'constraints': constraints3_10_v0
    }
)

# original env:

register(
    id='BSSEnv-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21))
    }
)

register(
    id='BSSEnvTest-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61))
    }
)

register(
    id='BSSEnv-v1',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-art',
        'scenarios': list(range(0, 30))
    }
)

register(
    id='BSSEnvTest-v1',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-art',
        'scenarios': list(range(30, 100))
    }
)

register(
    id='BSSEnv-v2',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-OD-art',
        'scenarios': list(range(0, 30))
    }
)

register(
    id='BSSEnvTest-v2',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-OD-art',
        'scenarios': list(range(30, 100))
    }
)
