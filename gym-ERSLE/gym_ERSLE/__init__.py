from gym.envs.registration import register

from gym_ERSLE.pyERSEnv import Scene4  # noqa F401
from gym_ERSLE.pyERSEnv import Scene5  # noqa F401
from gym_ERSLE.pyERSEnv import SgScene  # noqa F401
from gym_ERSLE.pyERSEnv import ToyScene  # noqa F401
from gym_ERSLE.pyERSEnv.scenes.constraints import (build_constraints_2_level_simple,
                                                   build_constraints_min_max_every)

version_to_scene_map = {
    'v3': 'gym_ERSLE:ToyScene',
    'v4': 'gym_ERSLE:Scene4',
    'v5': 'gym_ERSLE:Scene5',
    'v6': 'gym_ERSLE:Scene5',
    'v7': 'gym_ERSLE:Scene5',
    'v8': 'gym_ERSLE:Scene5'
}

version_to_ambs_map = {'v4': 24, 'v5': 40, 'v6': 32, 'v7': 24, 'v8': 16}
version_to_bases_map = {'v4': 12, 'v5': 25, 'v6': 25, 'v7': 25, 'v8': 25}

constraints1 = build_constraints_min_max_every(
    25, 5, 2, 10, 0, 8, nresources=32)

constraints2 = build_constraints_2_level_simple(25,
                                                [
                                                    [0, 1, 5, 6, 7],
                                                    [2, 3, 4, 8, 9],
                                                    [10, 15, 20],
                                                    [11, 12, 16, 17],
                                                    [13, 14, 18, 19, 24],
                                                    [21, 22, 23]
                                                ],
                                                [2, 2, 1, 2, 2, 1],
                                                [16, 16, 14, 16, 16, 14],
                                                0,
                                                6, nresources=32)


for version in ['v3', 'v4']:
    for decision_interval in [1, 10, 15, 20, 30, 60, 120, 240, 360, 720, 1440]:
        for ca in [False, True]:
            for im in [False, True]:
                for dynamic in [False, True]:
                    for blips in [False, True]:
                        register(
                            id='pyERSEnv{0}{1}{2}{3}{4}-{5}'.format(
                                '-im' if im else '',
                                '-ca' if ca else '',
                                '-dynamic' if dynamic else '',
                                '-blips' if blips else '',
                                '-{0}'.format(decision_interval) if decision_interval > 1 else '',
                                version
                            ),
                            entry_point=version_to_scene_map[version],
                            kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                    'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips},
                            max_episode_steps=10000000,
                            nondeterministic=False
                        )


for version in ['v5', 'v6', 'v7', 'v8']:
    for decision_interval in [1, 10, 15, 20, 30, 60, 120, 240, 360, 720, 1440]:
        for ca in [False, True]:
            for im in [False, True]:
                for dynamic in [False, True]:
                    for blips in [False, True]:
                        for cap in [None, 2, 4, 6, 8, 10]:
                            for nmin in [0, 1, 2]:
                                for constraints in [None, constraints2]:
                                    register(
                                        id='pyERSEnv{0}{1}{2}{3}{4}{5}{6}{7}-{8}'.format(
                                            '-im' if im else '',
                                            '-ca' if ca else '',
                                            '-dynamic' if dynamic else '',
                                            '-blips' if blips else '',
                                            '-min{0}'.format(nmin) if nmin > 0 else '',
                                            '-cap{0}'.format(cap) if cap else '',
                                            '-constraints' if constraints else '',
                                            '-{0}'.format(
                                                decision_interval) if decision_interval > 1 else '',
                                            version
                                        ),
                                        entry_point=version_to_scene_map[version],
                                        kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                                'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips,
                                                'nbases': version_to_bases_map[version], 'nambs': version_to_ambs_map[version],
                                                'nhospitals': 36, 'nmin': nmin, 'ncap': cap, 'constraints': constraints},
                                        max_episode_steps=10000000,
                                        nondeterministic=False
                                    )
                                    register(
                                        id='SgERSEnv{0}{1}{2}{3}{4}{5}{6}{7}-{8}'.format(
                                            '-im' if im else '',
                                            '-ca' if ca else '',
                                            '-dynamic' if dynamic else '',
                                            '-blips' if blips else '',
                                            '-min{0}'.format(nmin) if nmin > 0 else '',
                                            '-cap{0}'.format(cap) if cap else '',
                                            '-constraints' if constraints else '',
                                            '-{0}'.format(
                                                decision_interval) if decision_interval > 1 else '',
                                            version
                                        ),
                                        entry_point='gym_ERSLE:SgScene',
                                        kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                                'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips,
                                                'nbases': version_to_bases_map[version], 'nambs': version_to_ambs_map[version],
                                                'nhospitals': 36, 'nmin': nmin, 'ncap': cap, 'constraints': constraints},
                                        max_episode_steps=10000000,
                                        nondeterministic=False
                                    )
# additional envs 2022
constraints3 = build_constraints_2_level_simple(25,
                                                [
                                                    [0, 1, 5, 6, 7],
                                                    [2, 3, 4, 8, 9],
                                                    [10, 15, 20],
                                                    [11, 12, 16, 17],
                                                    [13, 14, 18, 19, 24],
                                                    [21, 22, 23]
                                                ],
                                                [4, 4, 2, 3, 4, 2],
                                                [8, 8, 6, 7, 8, 6],
                                                0,
                                                4, nresources=32)

for version in ['v5', 'v6', 'v7', 'v8']:
    for decision_interval in [1, 10, 15, 20, 30, 60, 120, 240, 360, 720, 1440]:
        for ca in [False, True]:
            for im in [False, True]:
                for dynamic in [False, True]:
                    for blips in [False, True]:
                        for cap in [None, 2, 4, 6, 8, 10]:
                            for nmin in [0, 1, 2]:
                                for constraints in [constraints3]:
                                    register(
                                        id='pyERSEnv{0}{1}{2}{3}{4}{5}{6}{7}-{8}'.format(
                                            '-im' if im else '',
                                            '-ca' if ca else '',
                                            '-dynamic' if dynamic else '',
                                            '-blips' if blips else '',
                                            '-min{0}'.format(nmin) if nmin > 0 else '',
                                            '-cap{0}'.format(cap) if cap else '',
                                            '-constraints3' if constraints else '',
                                            '-{0}'.format(
                                                decision_interval) if decision_interval > 1 else '',
                                            version
                                        ),
                                        entry_point=version_to_scene_map[version],
                                        kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                                'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips,
                                                'nbases': version_to_bases_map[version], 'nambs': version_to_ambs_map[version],
                                                'nhospitals': 36, 'nmin': nmin, 'ncap': cap, 'constraints': constraints},
                                        max_episode_steps=10000000,
                                        nondeterministic=False
                                    )
                                    register(
                                        id='SgERSEnv{0}{1}{2}{3}{4}{5}{6}{7}-{8}'.format(
                                            '-im' if im else '',
                                            '-ca' if ca else '',
                                            '-dynamic' if dynamic else '',
                                            '-blips' if blips else '',
                                            '-min{0}'.format(nmin) if nmin > 0 else '',
                                            '-cap{0}'.format(cap) if cap else '',
                                            '-constraints3' if constraints else '',
                                            '-{0}'.format(
                                                decision_interval) if decision_interval > 1 else '',
                                            version
                                        ),
                                        entry_point='gym_ERSLE:SgScene',
                                        kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                                'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips,
                                                'nbases': version_to_bases_map[version], 'nambs': version_to_ambs_map[version],
                                                'nhospitals': 36, 'nmin': nmin, 'ncap': cap, 'constraints': constraints},
                                        max_episode_steps=10000000,
                                        nondeterministic=False
                                    )

# thighter constraints exp 1
groups = [[0, 1, 5, 6, 7], [2, 3, 4, 8, 9], [10, 15, 20], [11, 12, 16, 17], [13, 14, 18, 19, 24], [21, 22, 23]]
upper_2 = [10, 10, 6, 8, 10, 6]
upper_4 = [20, 20, 12, 16, 20, 12]
upper_6 = [30, 30, 18, 24, 30, 18]
lower_25 = [2, 2, 1, 1, 2, 1]
lower_50 = [3, 3, 2, 3, 3, 2]
lower_75 = [5, 5, 3, 4, 5, 3] # also 80 and 85 --> sum is 25: 78.125%
lower_80 = [5, 5, 3, 4, 5, 3]
lower_875 = [6, 6, 3, 4, 6, 3] # sum is 28
lower_90 = [6, 6, 3, 5, 6, 3] # sum is 29
lower_100 = [6, 6, 4, 5, 6, 4] # sum is 31
upper_group_6 = [6] * 6
upper_group_8 = [8] * 8
upper_group_10 = [10] * 10


constraints6_80 = build_constraints_2_level_simple(25, groups, lower_80, upper_6, 0, 6, nresources=32)
constraints4_80 = build_constraints_2_level_simple(25, groups, lower_80, upper_4, 0, 4, nresources=32)
constraints2_80 = build_constraints_2_level_simple(25, groups, lower_80, upper_2, 0, 2, nresources=32)
constraints6_90 = build_constraints_2_level_simple(25, groups, lower_90, upper_6, 0, 6, nresources=32)
constraints4_90 = build_constraints_2_level_simple(25, groups, lower_90, upper_4, 0, 4, nresources=32)
constraints2_90 = build_constraints_2_level_simple(25, groups, lower_90, upper_2, 0, 2, nresources=32)
constraints6_100 = build_constraints_2_level_simple(25, groups, lower_100, upper_6, 0, 6, nresources=32)
constraints4_100 = build_constraints_2_level_simple(25, groups, lower_100, upper_4, 0, 4, nresources=32)
constraints2_100 = build_constraints_2_level_simple(25, groups, lower_100, upper_2, 0, 2, nresources=32)

constraints6_75 = build_constraints_2_level_simple(25, groups, lower_75, upper_6, 0, 6, nresources=32)
constraints4_75 = build_constraints_2_level_simple(25, groups, lower_75, upper_4, 0, 4, nresources=32)
constraints2_75 = build_constraints_2_level_simple(25, groups, lower_75, upper_2, 0, 2, nresources=32)
constraints6_50 = build_constraints_2_level_simple(25, groups, lower_50, upper_6, 0, 6, nresources=32)
constraints4_50 = build_constraints_2_level_simple(25, groups, lower_50, upper_4, 0, 4, nresources=32)
constraints2_50 = build_constraints_2_level_simple(25, groups, lower_50, upper_2, 0, 2, nresources=32)
constraints6_25 = build_constraints_2_level_simple(25, groups, lower_25, upper_6, 0, 6, nresources=32)
constraints4_25 = build_constraints_2_level_simple(25, groups, lower_25, upper_4, 0, 4, nresources=32)
constraints2_25 = build_constraints_2_level_simple(25, groups, lower_25, upper_2, 0, 2, nresources=32)

constraints2_80_6 = build_constraints_2_level_simple(25, groups, lower_80, upper_group_6, 0, 2, nresources=32)
constraints2_80_8 = build_constraints_2_level_simple(25, groups, lower_80, upper_group_8, 0, 2, nresources=32)
constraints2_80_10 = build_constraints_2_level_simple(25, groups, lower_80, upper_group_10, 0, 2, nresources=32)
constraints4_80_6 = build_constraints_2_level_simple(25, groups, lower_80, upper_group_6, 0, 4, nresources=32)
constraints4_80_8 = build_constraints_2_level_simple(25, groups, lower_80, upper_group_8, 0, 4, nresources=32)
constraints4_80_10 = build_constraints_2_level_simple(25, groups, lower_80, upper_group_10, 0, 4, nresources=32)

constraints2_90_6 = build_constraints_2_level_simple(25, groups, lower_90, upper_group_6, 0, 2, nresources=32)
constraints2_90_8 = build_constraints_2_level_simple(25, groups, lower_90, upper_group_8, 0, 2, nresources=32)
constraints2_90_10 = build_constraints_2_level_simple(25, groups, lower_90, upper_group_10, 0, 2, nresources=32)
constraints4_90_6 = build_constraints_2_level_simple(25, groups, lower_90, upper_group_6, 0, 4, nresources=32)
constraints4_90_8 = build_constraints_2_level_simple(25, groups, lower_90, upper_group_8, 0, 4, nresources=32)
constraints4_90_10 = build_constraints_2_level_simple(25, groups, lower_90, upper_group_10, 0, 4, nresources=32)

constraints2_100_6 = build_constraints_2_level_simple(25, groups, lower_100, upper_group_6, 0, 2, nresources=32)
constraints2_100_8 = build_constraints_2_level_simple(25, groups, lower_100, upper_group_8, 0, 2, nresources=32)
constraints2_100_10 = build_constraints_2_level_simple(25, groups, lower_100, upper_group_10, 0, 2, nresources=32)
constraints4_100_6 = build_constraints_2_level_simple(25, groups, lower_100, upper_group_6, 0, 4, nresources=32)
constraints4_100_8 = build_constraints_2_level_simple(25, groups, lower_100, upper_group_8, 0, 4, nresources=32)
constraints4_100_10 = build_constraints_2_level_simple(25, groups, lower_100, upper_group_10, 0, 4, nresources=32)

constraints_ul = [constraints2_80_6, constraints2_80_8, constraints2_80_10, constraints4_80_6, constraints4_80_8, constraints4_80_10, \
constraints2_90_6, constraints2_90_8, constraints2_90_10, constraints4_90_6, constraints4_90_8, constraints4_90_10, \
constraints2_100_6, constraints2_100_8, constraints2_100_10, constraints4_100_6, constraints4_100_8, constraints4_100_10]
constraint_ul_names = ["constraints2_80_6", "constraints2_80_8", "constraints2_80_10", "constraints4_80_6", "constraints4_80_8", "constraints4_80_10", \
"constraints2_90_6", "constraints2_90_8", "constraints2_90_10", "constraints4_90_6", "constraints4_90_8", "constraints4_90_10", \
"constraints2_100_6", "constraints2_100_8", "constraints2_100_10", "constraints4_100_6", "constraints4_100_8", "constraints4_100_10"]
constraints_n = [constraints2_80, constraints2_90, constraints2_100, constraints4_80, constraints4_90, constraints4_100, constraints6_80, constraints6_90, constraints6_100]
constraint_n_names = ["constraints2_80", "constraints2_90", "constraints2_100", "constraints4_80", "constraints4_90", "constraints4_100", "constraints6_80", "constraints6_90", "constraints6_100"]
constraints = [constraints2_25, constraints4_25, constraints6_25, constraints2_50, constraints4_50, constraints6_50, constraints2_75, constraints4_75, constraints6_75]
constraint_names = ["constraints2_25", "constraints4_25", "constraints6_25", "constraints2_50", "constraints4_50", "constraints6_50", "constraints2_75", "constraints4_75", "constraints6_75"]
constraints = constraints + constraints_n + constraints_ul
constraint_names = constraint_names + constraint_n_names + constraint_ul_names


for version in ['v5', 'v6', 'v7', 'v8']:
    for decision_interval in [1, 10, 15, 20, 30, 60, 120, 240, 360, 720, 1440]:
        for ca in [False, True]:
            for im in [False, True]:
                for dynamic in [False, True]:
                    for blips in [False, True]:
                        for cap in [None, 2, 4, 6, 8, 10]:
                            for nmin in [0, 1, 2]:
                                for i in range(len(constraints)):
                                    register(
                                        id='pyERSEnv{0}{1}{2}{3}{4}{5}{6}{7}-{8}'.format(
                                            '-im' if im else '',
                                            '-ca' if ca else '',
                                            '-dynamic' if dynamic else '',
                                            '-blips' if blips else '',
                                            '-min{0}'.format(nmin) if nmin > 0 else '',
                                            '-cap{0}'.format(cap) if cap else '',
                                            '-'+constraint_names[i] if constraints[i] else '',
                                            '-{0}'.format(
                                                decision_interval) if decision_interval > 1 else '',
                                            version
                                        ),
                                        entry_point=version_to_scene_map[version],
                                        kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                                'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips,
                                                'nbases': version_to_bases_map[version], 'nambs': version_to_ambs_map[version],
                                                'nhospitals': 36, 'nmin': nmin, 'ncap': cap, 'constraints': constraints[i]},
                                        max_episode_steps=10000000,
                                        nondeterministic=False
                                    )
                                    register(
                                        id='SgERSEnv{0}{1}{2}{3}{4}{5}{6}{7}-{8}'.format(
                                            '-im' if im else '',
                                            '-ca' if ca else '',
                                            '-dynamic' if dynamic else '',
                                            '-blips' if blips else '',
                                            '-min{0}'.format(nmin) if nmin > 0 else '',
                                            '-cap{0}'.format(cap) if cap else '',
                                            '-'+constraint_names[i] if constraints[i] else '',
                                            '-{0}'.format(
                                                decision_interval) if decision_interval > 1 else '',
                                            version
                                        ),
                                        entry_point='gym_ERSLE:SgScene',
                                        kwargs={'discrete_action': not ca, 'discrete_state': not im,
                                                'decision_interval': decision_interval, 'dynamic': dynamic, 'random_blips': blips,
                                                'nbases': version_to_bases_map[version], 'nambs': version_to_ambs_map[version],
                                                'nhospitals': 36, 'nmin': nmin, 'ncap': cap, 'constraints': constraints[i]},
                                        max_episode_steps=10000000,
                                        nondeterministic=False
                                    )