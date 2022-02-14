./test.sh arch="vanilla" root="ICASSP2022-vanilla" name="200k" \
`# dataset parameters` \
data_loader="mogaze" \
inp_length=5 out_length=5 time_step_size=12 padded_length=0 \
`# pose network` \
pose_encoder_units="63 256 128" pose_encoder_activations="relu relu" \
pose_mu_var_units="256 64 8" pose_mu_var_activations="relu none" \
pose_decoder_units="136 256 63" pose_decoder_activations="relu none" \
`# time network` \
delta_pose_encoder_units="63 256 128" delta_pose_encoder_activations="relu relu" \
time_encoder_units="1 256 16" time_encoder_activations="relu relu" \
time_mu_var_units="144 64 8" time_mu_var_activations="relu none" \
time_decoder_units="136 256 1" time_decoder_activations="relu relu" \
`# checkpointing` \
epoch_names="key_pose_epoch_0016_best_0016.pt time_epoch_0035_best_0035.pt" \
strict=0 \
result_root="ICASSP2022-vanilla" result_name="200k" \
0