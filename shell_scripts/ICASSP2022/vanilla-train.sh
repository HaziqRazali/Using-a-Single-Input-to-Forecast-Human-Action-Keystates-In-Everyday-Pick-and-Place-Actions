./train.sh arch="vanilla" root="ICASSP2022-vanilla" name="200k" \
`# dataset parameters` \
data_loader="mogaze" \
inp_length=5 out_length=5 step_size=12 padded_length=0 \
`# learning parameters` \
lr=1e-3 tr_step=500 va_step=1000 \
loss_names="key_pose pose_posterior time time_posterior" \
loss_functions="mse kl_loss mse kl_loss" \
loss_weights="[1.0]*1000 [1.0]*1000 [1.0]*1000 [1.0]*1000" \
task_names="key_pose time" \
task_component1="key_pose pose_posterior" task_component2="time time_posterior" \
`# pose network` \
pose_encoder_units="63 256 128" pose_encoder_activations="relu relu" \
pose_mu_var_units="256 64 8" pose_mu_var_activations="relu none" \
pose_decoder_units="136 256 63" pose_decoder_activations="relu none" \
`# time network` \
delta_pose_encoder_units="63 256 128" delta_pose_encoder_activations="relu relu" \
time_encoder_units="1 256 16" time_encoder_activations="relu relu" \
time_mu_var_units="144 64 8" time_mu_var_activations="relu none" \
time_decoder_units="136 256 1" time_decoder_activations="relu relu" \
0