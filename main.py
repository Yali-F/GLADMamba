import os   
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from model import GLADMamba
from data_loader import *
import argparse
import numpy as np
import torch
import random
import sklearn.metrics as skm
import torch_geometric


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='ad')
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-GNN_Encoder', type=str, default='GCN')                            
    parser.add_argument('-graph_level_pool', type=str, default='global_mean_pool')

    parser.add_argument('-d_model', type=int, default=64, help='Model dimension')                        
    parser.add_argument('-dt_rank', type=int, default=4, help='')
    parser.add_argument('-d_state', type=int, default=4, help='SSM state expansion factor')
    parser.add_argument('-d_conv', type=int, default=4, help='Local convolution width')
    parser.add_argument('-conv_bias', type=bool, default=True, help='')      
    parser.add_argument('-bias', type=bool, default=True, help='')
    parser.add_argument('-l', type=int, default=5, help='')

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)


if __name__ == '__main__':
    setup_seed(0)
    args = arg_parse()

    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    tot_auc_list = [[] for _ in range(args.num_epoch // args.eval_freq)]      

    aucs = []
    for trial in range(args.num_trial):
        setup_seed(trial + 1)
        
        if args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial])

        dataset_num_features = meta['num_feat']
        n_train = meta['num_train']

        args.feat_dim = meta['num_feat']

        if trial == 0:
            print('================')
            print('Exp_type: {}'.format(args.exp_type))
            print('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            print('num_features: {}'.format(dataset_num_features))
            print('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            print('hidden_dim: {}'.format(args.hidden_dim))
            print('num_gc_layers: {}'.format(args.num_layer))
            print('GNN_Encoder: {}'.format(args.GNN_Encoder))
            print('feature_dim: {}'.format(args.feat_dim))
            print('================')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GLADMamba(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.num_epoch + 1):
            if args.is_adaptive:
                if epoch == 1:
                    weight_g, weight_n = 1, 1
                else:
                    weight_g, weight_n = std_g ** args.alpha, std_n ** args.alpha
                    weight_sum = (weight_g  + weight_n) / 2
                    weight_g, weight_n = weight_g/weight_sum, weight_n/weight_sum

            model.train()
            loss_all = 0
            if args.is_adaptive:
                loss_g_all, loss_n_all = [], []

            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                g_f, g_s, n_f, n_s = model(data, data.x, data.x_s, data.edge_index, data.batch, data.num_graphs, args)
                loss_g = model.calc_loss_g(g_f, g_s)
                loss_n = model.calc_loss_n(n_f, n_s, data.batch)
                if args.is_adaptive:
                    loss = weight_g * loss_g.mean() + weight_n * loss_n.mean()
                    loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                    loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                else:
                    loss = loss_g.mean() + loss_n.mean()
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

            if args.is_adaptive:
                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)

            if epoch % args.eval_freq == 0:
                model.eval()

                y_score_all = []
                y_true_all = []
                
                for data in dataloader_test:
                    data = data.to(device)
                    g_f, g_s, n_f, n_s = model(data, data.x, data.x_s, data.edge_index, data.batch, data.num_graphs, args)
                    y_score_g = model.calc_loss_g(g_f, g_s)
                    y_score_n = model.calc_loss_n(n_f, n_s, data.batch)
                    if args.is_adaptive:
                        y_score = (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
                    else:
                        y_score = y_score_g + y_score_n

                    y_true = data.y
                    y_score_all = y_score_all + y_score.detach().cpu().tolist()
                    y_true_all = y_true_all + y_true.detach().cpu().tolist()           
                auc = skm.roc_auc_score(y_true_all, y_score_all)

                print('[EVAL] Epoch: {:03d} | AUC:{:.4f}'.format(epoch, auc))

                tot_auc_list[epoch // args.eval_freq - 1].append(auc)                     
        print('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(trial, auc))
        aucs.append(auc)

    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    auc_list = [(np.mean(auc), np.std(auc), (idx + 1) * args.eval_freq) for idx, auc in enumerate(tot_auc_list)]
    # for row in auc_list:
    #     print(row)
    auc_list.sort(key = lambda x: (-x[0], x[1], x[2]))

    print('[The Final result is] Avg_Auc:{:.4f} +- {:.4f}, achieved in {} epoch'.format(auc_list[0][0], auc_list[0][1], auc_list[0][2]))
    print(args.exp_type, args.DS)