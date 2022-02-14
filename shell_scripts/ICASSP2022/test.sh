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
			
			# # # # # # # # #
			# checkpointing #
            # # # # # # # # #					
			
			epoch_names)				epoch_names=${VALUE} ;;
			result_root)				result_root=${VALUE} ;;
			result_name)				result_name=${VALUE} ;;
			strict)						strict=${VALUE} ;;
			
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

# dataset
dataset_root="./data"
data_loader=${data_loader:-}
batch_size=${batch_size:-32}
inp_length=${inp_length:-5}
out_length=${out_length:-5}
time_step_size=${time_step_size:-12}
padded_length=${padded_length:-0}
test_set="test"

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

num_samples=256

# # # # # # # # #
# checkpointing #
# # # # # # # # #

weight_root=$root
model_name=$name
epoch_names=${epoch_names:-}
layer_names1="inp_pose_encoder key_pose_encoder pose_mu pose_log_var key_pose_decoder"
layer_names2="delta_pose_encoder time_encoder time_mu time_log_var time_decoder"
result_root=${result_root:-}
result_name=${result_name:-}
strict=${strict:-0}

cd ../..
CUDA_VISIBLE_DEVICES=${gpu_num} python test.py --args $args \
`# dataset` \
--dataset_root $dataset_root --data_loader $data_loader --batch_size $batch_size --test_set $test_set \
--inp_length $inp_length --out_length $out_length --time_step_size $time_step_size --padded_length $padded_length \
`# model` \
--architecture $architecture \
--num_samples $num_samples \
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
--weight_root "$weight_root" --model_name "$model_name" \
--result_root "$result_root" --result_name "$result_name" \
--epoch_names $epoch_names \
--layer_names $layer_names1 --layer_names $layer_names2 \
--strict $strict
cd shell_scripts/ICASSP2022