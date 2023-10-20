#!/bin/bash
#SBATCH -J sample_DT_gird_system_notStateSum_restrictThermalOnOff_useHistoryState_len25_probDisconnection
#SBATCH -o res/sample_DT_gird_system_notStateSum_restrictThermalOnOff_useHistoryState_len25_probDisconnection.txt
#SBATCH -c 4
#SBATCH -x dell-gpu-04,dell-gpu-10,dell-gpu-11,dell-gpu-23
source /home/LAB/zhutc/anaconda3/bin/activate gird_system
python train.py --buffer_type 'FIFO' \
--max_episode 200000 --max_timestep 10000 \
--max_buffer_size 50000 --min_state 1 \
--model_update_freq 1024 --model_save_freq 100 \
--init_model 1 --save_model 0 --output_res 1 \
--load_model 0 --load_state_normalization 1 --update_state_normalization 0 \
--use_mini_batch 1 --use_state_norm 1 \
--reflect_actionspace 1 --add_balance_loss 1 --balance_loss_rate 0.01 \
--balance_loss_rate_decay_rate 0.0 --balance_loss_rate_decay_freq 10000 \
--min_balance_loss_rate 0.001 --split_balance_loss 1 \
--danger_region_rate 0.1 --save_region_rate 0.6 \
--save_region_balance_loss_rate 0.0001 --warning_region_balance_loss_rate 0.001 \
--danger_region_balance_loss_rate 0.01 \
--use_history_state 1 --use_history_action 0 \
--history_state_len 25 --gru_num_layers 2 --gru_hidden_size 64 \
--sample_by_train 1 --data_block_size 500000 --min_episode_length 30 \
--sample_data_dir './sample_data/episode_data/' --active_function 'tanh' \
--punish_balance_out_range 0 --punish_balance_out_range_rate 0.0 \
--reward_from_env 1 --reward_for_survive 0.0 \
--lr_actor 1e-5 --lr_critic 1e-3 --lr_decay_step_size 400 --lr_decay_gamma 0.9 \
--batch_size 1024 --mini_batch_size 128 \
--ban_prob_disconnection 0 --ban_check_gen_status 1 \
--ban_thermal_on 0 --ban_thermal_off 0 --restrict_thermal_on_off 1 \
--state_normal_load_path './save_model/state_normal_model/min_state_1/1200_save_model.pth' \
--res_file_dir './res/sample_res/' \
--res_file_name 'sample_DT_gird_system_notStateSum_restrictThermalOnOff_useHistoryState_len25_probDisconnection.txt'