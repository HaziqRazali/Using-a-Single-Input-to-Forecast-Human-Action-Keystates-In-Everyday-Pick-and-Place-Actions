#!/bin/bash

# read arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2- -d=)   
    
    case "$KEY" in
            arch)       				arch=${VALUE} ;;
            root)       				root=${VALUE} ;;
            name)       				name=${VALUE} ;;
			
			# # # # # # # # # # # #
			# dataset parameters  #
            # # # # # # # # # # # #
			
			data_loader)        		data_loader=${VALUE} ;;
			batch_size)					batch_size=${VALUE} ;;
			inp_length)					inp_length=${VALUE} ;;
			out_length)					out_length=${VALUE} ;;
			time_step_size)				time_step_size=${VALUE} ;;
			padded_length)				padded_length=${VALUE} ;;
			
			# # # # # # # # # # # #
			# learning parameters #
            # # # # # # # # # # # #
			
			lr)							lr=${VALUE} ;;
			tr_step)					tr_step=${VALUE} ;;
			va_step)					va_step=${VALUE} ;;
			loss_names)					loss_names=${VALUE} ;;
			loss_functions)				loss_functions=${VALUE} ;;
			loss_weights)				loss_weights=${VALUE} ;;
			task_names)					task_names=${VALUE} ;;
			task_component1)			task_component1=${VALUE} ;;
			task_component2)			task_component2=${VALUE} ;;
			
			# # # # # # # # #
			# pose network  #
            # # # # # # # # #
			
			pose_encoder_units)			pose_encoder_units=${VALUE} ;;
			pose_encoder_activations)	pose_encoder_activations=${VALUE} ;;
			pose_mu_var_units)			pose_mu_var_units=${VALUE} ;;
			pose_mu_var_activations)	pose_mu_var_activations=${VALUE} ;;
			pose_decoder_units)			pose_decoder_units=${VALUE} ;;
			pose_decoder_activations)	pose_decoder_activations=${VALUE} ;;
					
			# # # # # # # # #
			# time network  #
            # # # # # # # # #
			
			delta_pose_encoder_units)		delta_pose_encoder_units=${VALUE} ;;
			delta_pose_encoder_activations)	delta_pose_encoder_activations=${VALUE} ;;
			time_encoder_units)				time_encoder_units=${VALUE} ;;
			time_encoder_activations)		time_encoder_activations=${VALUE} ;;
			time_mu_var_units)				time_mu_var_units=${VALUE} ;;
			time_mu_var_activations)		time_mu_var_activations=${VALUE} ;;
			time_decoder_units)				time_decoder_units=${VALUE} ;;
			time_decoder_activations)		time_decoder_activations=${VALUE} ;;
			
            *)   
    esac  
done

# args
args="args_mogaze"

# optimization
gpu_num=${!#}
lr=${lr:-1e-3}
tr_step=${tr_step:-500}
va_step=${va_step:-1000}
loss_names=${loss_names:-}
loss_functions=${loss_functions:-}
loss_weights=${loss_weights:-}

# checkpoint task names
task_names=${task_names:-}
task_component1=${task_component1:-}
task_component2=${task_component2:-}

dataset_root="./data"
data_loader=${data_loader:-}
batch_size=${batch_size:-32}
inp_length=${inp_length:-5}
out_length=${out_length:-5}
time_step_size=${time_step_size:-12}
padded_length=${padded_length:-0}

# # # # #
# model #
# # # # #

architecture=${arch:-vanilla}
pose_encoder_units=${pose_encoder_units:-}
pose_encoder_activations=${pose_encoder_activations:-}
pose_mu_var_units=${pose_mu_var_units:-}
pose_mu_var_activations=${pose_mu_var_activations:-}
pose_decoder_units=${pose_decoder_units:-}
pose_decoder_activations=${pose_decoder_activations:-}

delta_pose_encoder_units=${delta_pose_encoder_units:-}
delta_pose_encoder_activations=${delta_pose_encoder_activations:-}
time_encoder_units=${time_encoder_units:-}
time_encoder_activations=${time_encoder_activations:-}
time_mu_var_units=${time_mu_var_units:-}
time_mu_var_activations=${time_mu_var_activations:-}
time_decoder_units=${time_decoder_units:-}
time_decoder_activations=${time_decoder_activations:-}

# # # # # # # # #
# checkpointing #
# # # # # # # # #

restore_from_checkpoint=${restore_from_checkpoint:-0}
log_root=$root
weight_root=$root
model_name=$name

cd ../..
CUDA_VISIBLE_DEVICES=${gpu_num} python train.py --args $args \
`# dataset` \
--dataset_root $dataset_root --data_loader $data_loader --batch_size $batch_size \
--inp_length $inp_length --out_length $out_length --time_step_size $time_step_size --padded_length $padded_length \
`# optimization` \
--lr $lr --tr_step $tr_step --va_step $va_step --loss_names $loss_names --loss_functions $loss_functions --loss_weights $loss_weights \
--task_names $task_names --task_components $task_component1 --task_components $task_component2 \
`# model` \
--architecture $architecture \
`# pose network` \
--pose_encoder_units $pose_encoder_units --pose_encoder_activations $pose_encoder_activations \
--pose_mu_var_units $pose_mu_var_units --pose_mu_var_activations $pose_mu_var_activations \
--pose_decoder_units $pose_decoder_units --pose_decoder_activations $pose_decoder_activations \
`# time network` \
--delta_pose_encoder_units $delta_pose_encoder_units --delta_pose_encoder_activations $delta_pose_encoder_activations \
--time_encoder_units $time_encoder_units --time_encoder_activations $time_encoder_activations \
--time_mu_var_units $time_mu_var_units --time_mu_var_activations $time_mu_var_activations \
--time_decoder_units $time_decoder_units --time_decoder_activations $time_decoder_activations \
`# checkpointing` \
--log_root "$log_root" --weight_root "$weight_root" --model_name "$model_name" \
--restore_from_checkpoint $restore_from_checkpoint
cd shell_scripts/ICASSP2022