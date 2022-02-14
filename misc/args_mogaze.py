import os
import argparse

def argparser():

    # Argument Parser
    ##################################################### 
    parser = argparse.ArgumentParser()
    
    # # # # # #
    # dataset #
    # # # # # #
    
    parser.add_argument('--dataset_root', required=True, type=str)
    parser.add_argument('--data_loader',  required=True, type=str)
    parser.add_argument('--batch_size',   default=8, type=int)
    parser.add_argument('--test_set', type=str)
    
    # actions
    parser.add_argument('--actions', nargs="*", default=["pick","place"], type=str)
    parser.add_argument('--use_instructions', default=0, type=int)    
    
    # objects
    parser.add_argument('--furniture_names', nargs="*", type=str)
    parser.add_argument('--grid_sizes', nargs="*", action="append", type=int)
    parser.add_argument('--object_padded_length', default=0, type=int)
    
    # pose
    parser.add_argument('--inp_length', default=0, type=int)
    parser.add_argument('--out_length', default=0, type=int)    
    parser.add_argument('--time_step_size', default=0, type=int)
    parser.add_argument('--load_mid_pose', default=0, type=int)
    parser.add_argument('--preload_mid_pose', default=1, type=int)
    parser.add_argument('--pose_padded_length', default=0, type=int)
    parser.add_argument('--pose_type', choices=["xyz","exp"], type=str)
    
    # variables to unpad at test time
    parser.add_argument('--unpad', nargs="*", type=str)
    
    # # # # # # # # # # 
    # model settings  # 
    # # # # # # # # # #
    
    parser.add_argument('--architecture', required=True, type=str)
            
    # # # # # # # # #
    # Pose Network  #
    # # # # # # # # #
    
    parser.add_argument('--key_object', type=str)
    parser.add_argument('--pose_encoder_units', nargs="*", type=int)
    parser.add_argument('--pose_encoder_activations', nargs="*", type=str)
    parser.add_argument('--pose_mu_var_units', nargs="*", type=int)
    parser.add_argument('--pose_mu_var_activations', nargs="*", type=str)
    parser.add_argument('--pose_decoder_units', nargs="*", type=int)
    parser.add_argument('--pose_decoder_activations', nargs="*", type=str)

    # # # # # # # # #    
    # Time Network  #
    # # # # # # # # #
    
    parser.add_argument('--key_pose', type=str)
    parser.add_argument('--delta_pose_encoder_units', nargs="*", type=int)
    parser.add_argument('--delta_pose_encoder_activations', nargs="*", type=str)
    parser.add_argument('--time_encoder_units', nargs="*", type=int)
    parser.add_argument('--time_encoder_activations', nargs="*", type=str)
    parser.add_argument('--time_mu_var_units', nargs="*", type=int)
    parser.add_argument('--time_mu_var_activations', nargs="*", type=str)
    parser.add_argument('--time_decoder_units', nargs="*", type=int)
    parser.add_argument('--time_decoder_activations', nargs="*", type=str)
        
    # obstacle
    parser.add_argument('--obstacle_pad_size', nargs="*", default=[0.1,0.1], type=float)
                
    # # # # # # # # # # # # # # # # # # # # # # # # 
    # checkpointing                               #
    # # # # # # # # # # # # # # # # # # # # # # # # 
    
    parser.add_argument('--log_root', default="", type=str)
    parser.add_argument('--weight_root', default="", type=str)
    
    parser.add_argument('--model_name', default="", type=str)
    parser.add_argument('--epoch_names', nargs="*", type=str)                   # task specific epoch names
    parser.add_argument('--layer_names', nargs="*", action="append", type=str)  # task specific layer names
    
    parser.add_argument('--restore_from_checkpoint', default=0, type=int)
    parser.add_argument('--strict', default=0, type=int)
    
    parser.add_argument('--task_names', nargs="*", type=str)                       # task names
    parser.add_argument('--task_components', nargs="*", action="append", type=str) # task components e.g. key_pose = key_pose + pose_posterior
    parser.add_argument('--result_root', default="", type=str)
    parser.add_argument('--result_name', default="", type=str)

    # general optimization
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--tr_step', type=int)
    parser.add_argument('--va_step', type=int)
    parser.add_argument('--loss_names', nargs="*", type=str)
    parser.add_argument('--loss_functions', nargs="*", type=str)
    parser.add_argument('--loss_weights', nargs="*", type=str)
    parser.add_argument('--freeze', default="None", type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--reset_loss', nargs="*", type=int)
    
    # additional experiments
    parser.add_argument('--diversity_test', default="none", type=str)
    parser.add_argument('--generalizability_test', default=0, type=int)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--eval_type', default="mean", type=str)
    parser.add_argument('--save_result', default=0, type=int)
    parser.add_argument('--use_key_object', default=1, type=int)
    
    # for RNN key pose forecasting
    parser.add_argument('--rnn_encoder_units', nargs="*", type=int)
    parser.add_argument('--rnn_decoder_units', nargs="*", type=int)
    parser.add_argument('--prior_units', nargs="*", type=int)
    parser.add_argument('--prior_activations', nargs="*", type=str)
    parser.add_argument('--posterior_units', nargs="*", type=int)
    parser.add_argument('--posterior_activations', nargs="*", type=str)
    parser.add_argument('--eval_length', default=0, type=int)
    
    # for CNN
    parser.add_argument('--l', nargs="*", action="append", type=int)
    parser.add_argument('--s', nargs="*", action="append", type=int)
    parser.add_argument('--d', nargs="*", action="append", type=int)
    
    # parse
    args, unknown = parser.parse_known_args()
    
    args.log_root = os.path.join("./logs/",args.log_root)
    args.weight_root = os.path.join("./weights/",args.weight_root)
    args.result_root = os.path.join("./results/",args.result_root)
        
    # the lists containing the loss schedules must have the same length
    if args.loss_weights is not None:
        args.loss_weights = [eval(x) for x in args.loss_weights]
        assert len(set(map(len,args.loss_weights))) == 1
    
    # the loss names, functions and weights must have the same length    
    if args.loss_names is not None or args.loss_functions is not None or args.loss_weights is not None:
        assert len(args.loss_names) == len(args.loss_functions)
        assert len(args.loss_functions) == len(args.loss_weights)
        
    return args
