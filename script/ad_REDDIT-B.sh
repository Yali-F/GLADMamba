python main.py -exp_type ad -DS REDDIT-BINARY -rw_dim 8 -dg_dim 8 -hidden_dim 8 -batch_size 16 -batch_size_test 16 -num_epoch 800 -alpha 0.2 -num_layer 6 -GNN_Encoder GCN -graph_level_pool global_mean_pool -eval_freq 5 -d_state 3 -l 6          
