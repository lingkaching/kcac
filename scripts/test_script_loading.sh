#!/bin/bash
declare -a lrs=(1.0)
declare -a critic_lr_multis=(10)
declare -a envs=('SgERSEnv-ca-dynamic-blips-cap6-constraints2_50-30-v6')
declare -a pids=()
base_lr=0.0001
declare -a NNs=("[[], [128, 96], [[128], [], [96]]]")
declare -a NN_names=("org")
declare -a Ns=(10)
declare -a sigmas=(0)
declare -a taus=(0.001)
declare -a trains=(0)
declare -a SNs=("average")
declare -a train_everys=(48)
declare -a RSFs=(1)

for seed in $(seq 0 0)
do
	for NN in "${!NNs[@]}"
	do
		for critic_lr_multi in "${critic_lr_multis[@]}"
		do
			for env in "${envs[@]}"
			do
				for N in "${Ns[@]}"
				do
					for sigma in "${sigmas[@]}"
					do
						for tau in "${taus[@]}"
						do
							for train in "${trains[@]}"
							do
								for SN in "${SNs[@]}"
								do
									for train_every in "${train_everys[@]}"
									do
										for RSF in "${RSFs[@]}"
										do
											for lr in "${lrs[@]}"
											do
												lr_actor="$(echo "scale=10; $lr*$base_lr" | bc)"
												lr_critic="$(echo "scale=10; $lr_actor*$critic_lr_multi" | bc)"
												python main.py \
													--psdd_file="/DATA/moritz/psddAgentFiles/psdds/BSgERSEnv-ca-dynamic-blips-cap6-constraints2_50-30-v6.psdd" \
													--log_name=test\
													--log_dir="/DATA/moritz/" \
													--load_model_path="/DATA/moritz/log/SgERSEnv-ca-dynamic-blips-cap6-constraints2_50-30-v6/short_path/run_0/9999" \
													--save_model_every=1000 \
													--seed=$seed \
													--no_zones=25 \
													--no_resources=32 \
													--env_name=$env \
													--no_train_episodes=$train \
													--no_test_episodes=10 \
													--no_exploration_episodes=0 \
													--no_stack=3 \
													--NN="${NNs[NN]}" \
													--epsilon=0.9995 \
													--epsilon_decay=0.9995 \
													--epsilon_min=0.05 \
													--param_noise_sigma=$sigma \
													--M_start=1000 \
													--M_decay=0 \
													--N=$N \
													--coeff_entrop=0.0 \
													--lr_actor=$lr_actor \
													--lr_critic=$lr_critic \
													--buffer_size=100000 \
													--mode="step" \
													--replay_training \
													--train_every=$train_every \
													--batch_size=128 \
													--replay_ratio=1 \
													--state_normalization=$SN \
													--action_normalization="average" \
													--layer_normalization \
													--target_net_update_every=1 \
													--tau=$tau \
													--gamma=1.0 \
													--device="cpu" \
													--no_threads=1 \
													--uniform_action_sampling=20 \
													--reward_scaling_factor=$RSF \
													&
												pid=$!
												pids+=($pid)
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
	for process in "${pids[@]}"
	do
		echo $process
		#wait $process
	done
done
# alternatives: 
# env_name: "SgERSEnv-Poisson+Surge"
# buffer_mode: "episode"
# adding PER-Buffer: --per_buffer --per_buffer_alpha 0.123 -per_buffer_beta 0.321
# state_normalization: "mixed", "unnormalized"

