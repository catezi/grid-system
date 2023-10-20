srun -c 8 -x inspur-gpu-05 --pty bash
srun -c 16 --gres=gpu:1 -p dell --pty bash
srun -c 32 --gres=gpu:1 -p dell -x dell-gpu-25 --pty bash
srun -c 32 --gres=gpu:1 -p sugon --pty bash
srun -c 16 --gres=gpu:1 -p dell --pty bash
srun -c 4 --gres=gpu:1 -p sugon --pty bash
srun -c 4 --gres=gpu:1 -p sugon -x sugon-gpu-5 --pty bash
srun -c 4 --gres=gpu:1 -p dell --pty bash
srun -c 8 --gres=gpu:2 -p sugon --pty bash
srun -c 8 --gres=gpu:2 -p dell -x dell-gpu-25 --pty bash
srun -c 8 -p inspur -x inspur-gpu-10,inspur-gpu-09 --pty bash
srun -c 32 -p inspur -w inspur-gpu-01 --pty bash
srun -c 16 -p inspur -w inspur-gpu-03 --pty bash
srun -c 32 -p inspur -w inspur-gpu-05 --pty bash
srun -c 16 -p inspur -w inspur-gpu-07 --pty bash
srun -c 16 -p inspur -w inspur-gpu-09 --pty bash
srun -c 16 -p inspur -w inspur-gpu-13 --pty bash
srun -c 32 -p inspur -x inspur-gpu-05,inspur-gpu-10 --pty bash
srun -c 32 -p inspur -x inspur-gpu-01,inspur-gpu-02,inspur-gpu-03,inspur-gpu-10 --pty bash
srun -c 32 -p dell --pty bash
srun -c 32 --gres=gpu:1 -x dell-gpu-19 -p dell --pty bash
#SBATCH -x dell-gpu-22,dell-gpu-31,dell-gpu-32



srun -c 32 --gres=gpu:1 -p sugon --pty bash
jupyter notebook

conda activate decision-transformer-atari
source /home/LAB/anaconda3/bin/activate decision-transformer-atari
source /home/LAB/anaconda3/bin/activate decision-transformer-atari
source /home/LAB/anaconda3/bin/activate decision-transformer-gym
source /home/LAB/zhutc/anaconda3/bin/activate decision-transformer-gym
source /home/LAB/zhutc/anaconda3/bin/activate decision-transformer-atari-A100
conda activate decision-transformer-atari-A100
conda activate DDPG_MC_hunter
source /home/LAB/anaconda3/bin/activate gird_system
source /home/LAB/anaconda3/bin/activate DDPG_MC_hunter

source /home/LAB/zhutc/anaconda3/bin/activate gird_system



python train.py --agent_type 'A2C' --buffer_type 'FIFO' \
--max_episode 200000 --max_timestep 20000 \
--max_buffer_size 32 --min_state 1 \
--model_update_freq 32 --model_save_freq 100 \
--init_model 1 --save_model 1 --output_res 1 \
--load_model 0 --load_state_normalization 1 --update_state_normalization 0 \
--use_mini_batch 0 --use_state_norm 1 \
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
--lr_actor 1e-5 --lr_critic 1e-3 --lr_decay_step_size 400 --lr_decay_gamma 0.9 \
--batch_size 32 --mini_batch_size 128 \
--ban_prob_disconnection 0 --ban_check_gen_status 1 \
--ban_thermal_on 0 --ban_thermal_off 0 --restrict_thermal_on_off 1 \
--state_normal_load_path './save_model/state_normal_model/min_state_1/1200_save_model.pth' \
--model_save_dir './save_model/baseline_model/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection/' \
--res_file_dir './res/baseline_res/' \
--res_file_name 'train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection.txt'



python train.py --agent_type 'DDPG' --buffer_type 'FIFO' \
--max_episode 200000 --max_timestep 10000 \
--max_buffer_size 50000 --min_state 1 \
--model_update_freq 64 --model_save_freq 100 \
--init_model 1 --save_model 1 --output_res 1 \
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
--use_topology_info 0 --active_function 'tanh' \
--punish_balance_out_range 0 --punish_balance_out_range_rate 0.0 \
--reward_from_env 1 --reward_for_survive 0.0 \
--lr_actor 1e-5 --lr_critic 1e-3 --lr_decay_step_size 400 --lr_decay_gamma 0.9 \
--batch_size 64 --mini_batch_size 32 \
--ban_prob_disconnection 0 --ban_check_gen_status 1 \
--ban_thermal_on 0 --ban_thermal_off 0 --restrict_thermal_on_off 1 \
--state_normal_load_path './save_model/state_normal_model/min_state_1/1200_save_model.pth' \
--model_save_dir './save_model/hihdm_model/train_gird_system_DDPG_testReflectActionspace_balanceLossDecay_testBalanceLoss_split_notStateSum_restrictThermalOnOff_probDisconnection_useHisState_hisLen25/' \
--res_file_dir './res/hihdm_res/' \
--res_file_name 'train_gird_system_DDPG_testReflectActionspace_balanceLossDecay_testBalanceLoss_split_notStateSum_restrictThermalOnOff_probDisconnection_useHisState_hisLen25.txt'



python train.py --agent_type 'PPO' --buffer_type 'FIFO' \
--max_episode 200000 --max_timestep 10000 \
--max_buffer_size 50000 --min_state 1 \
--model_update_freq 64 --model_save_freq 100 \
--init_model 1 --save_model 1 --output_res 1 \
--load_model 0 --load_state_normalization 1 --update_state_normalization 0 \
--use_mini_batch 1 --use_state_norm 1 \
--reflect_actionspace 1 --add_balance_loss 1 --balance_loss_rate 0.01 \
--balance_loss_rate_decay_rate 0.0 --balance_loss_rate_decay_freq 10000 \
--min_balance_loss_rate 0.001 --split_balance_loss 1 \
--danger_region_rate 0.1 --save_region_rate 0.6 \
--save_region_balance_loss_rate 0.0001 --warning_region_balance_loss_rate 0.001 \
--danger_region_balance_loss_rate 0.01 \
--use_history_state 0 --use_history_action 0 \
--use_topology_info 0 --active_function 'tanh' \
--punish_balance_out_range 0 --punish_balance_out_range_rate 0.0 \
--reward_from_env 1 --reward_for_survive 0.0 \
--lr_actor 1e-5 --lr_critic 1e-3 --lr_decay_step_size 400 --lr_decay_gamma 0.9 \
--batch_size 64 --mini_batch_size 32 \
--ban_prob_disconnection 0 --ban_check_gen_status 1 \
--ban_thermal_on 0 --ban_thermal_off 0 --restrict_thermal_on_off 0 \
--state_normal_load_path './save_model/state_normal_model/min_state_1/1200_save_model.pth' \
--model_save_dir './save_model/baseline_model/train_grid_system_PPO_testBalanceLoss_split_notStateSum_probDisconnection/' \
--res_file_dir './res/baseline_res/' \
--res_file_name 'train_grid_system_PPO_testBalanceLoss_split_notStateSum_probDisconnection.txt'






python plot_toplogy.py


sbatch ./command/evaluate_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_useHisState_disconnect_len25.slurm
sbatch ./command/evaluate_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_useHisState_disconnect_len25_1.slurm
sbatch ./command/evaluate_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_useHisState_disconnect_len25_2.slurm
sbatch ./command/evaluate_grid_system_DDPG_testBalance,Loss_split_notStateSum_restrictThermalOnOff_useHisState_disconnect_len25_3.slurm

sbatch ./command/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_probDisconnection.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_probDisconnection_ban.slurm
sbatch ./command/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection_ban.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_probDisconnection.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_probDisconnection_ban.slurm

sbatch ./command/train_grid_system_PPO_testBalanceLoss_split_notStateSum_probDisconnection.slurm
sbatch ./command/train_grid_system_PPO_testBalanceLoss_split_notStateSum_probDisconnection_ban.slurm



sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_probDisconnection_useHisState_hisLen25_agent2.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_probDisconnection_useHisState_hisLen25_agent1.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_probDisconnection_useHisState_hisLen25_agent3.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_probDisconnection_useHisState_hisLen25_agent4.slurm




sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent1.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent2.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent3.slurm

sbatch ./command/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent1.slurm
sbatch ./command/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent2.slurm
sbatch ./command/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent3.slurm

sbatch ./command/train_grid_system_A2C_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent4.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_probDisconnection_ban_agent4.slurm

sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_probDisconnection_agent1.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_probDisconnection_agent2.slurm
sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_probDisconnection_agent3.slurm

sbatch ./command/train_grid_system_DDPG_testBalanceLoss_split_notStateSum_probDisconnection_agent4.slurm


sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_banProbDisConnect_banSingle_agent1.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_banProbDisConnect_banSingle_agent2.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_banProbDisConnect_banSingle_agent3.slurm
sbatch ./command/train_grid_system_TD3_testBalanceLoss_split_notStateSum_banProbDisConnect_banSingle_agent4.slurm


conda create -n grid2op_control_env --clone decision-transformer-gym



conda create -n language_condition_MAT python=3.8



