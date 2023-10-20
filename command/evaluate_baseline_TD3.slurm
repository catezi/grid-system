#!/bin/bash
source /root/anaconda3/bin/activate grid_system
python evaluate.py --buffer_type 'FIFO' --agent_type 'TD3' \
--max_episode 200000 --max_timestep 10000 \
--max_buffer_size 50000 --min_state 1 \
--model_update_freq 1024 --model_save_freq 100 \
--init_model 1 --save_model 0 --output_res 0 \
--load_model 1 --load_state_normalization 1 --update_state_normalization 0 \
--use_mini_batch 1 --use_state_norm 1 \
--reflect_actionspace 1 --add_balance_loss 1 --balance_loss_rate 0.01 \
--balance_loss_rate_decay_rate 0.0 --balance_loss_rate_decay_freq 10000 \
--min_balance_loss_rate 0.001 --split_balance_loss 1 \
--danger_region_rate 0.1 --save_region_rate 0.6 \
--save_region_balance_loss_rate 0.0001 --warning_region_balance_loss_rate 0.001 \
--danger_region_balance_loss_rate 0.01 \
--use_history_state 0 --use_history_action 0 --use_topology_info 0 \
--active_function 'tanh' \
--punish_balance_out_range 0 --punish_balance_out_range_rate 0.0 \
--reward_from_env 1 --reward_for_survive 0.0 \
--quick_tqdm 0 --num_process 4 --total_sample_episode 250 \
--lr_actor 1e-5 --lr_critic 1e-3 --lr_decay_step_size 400 --lr_decay_gamma 0.9 \
--batch_size 1024 --mini_batch_size 128 \
--ban_prob_disconnection 0 --ban_check_gen_status 1 \
--ban_thermal_on 1 --ban_thermal_off 1 --restrict_thermal_on_off 0 \
--state_normal_load_path './save_model/state_normal_model/min_state_1/1200_save_model.pth' \
--model_load_path "/root/qiuyue/gird_system/save_model/TD3/17500_save_model.pth"