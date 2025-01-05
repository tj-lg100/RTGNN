import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RTGNN_pro')
    parser.add_argument('--win_size', type=int, default=30, help='window size')
    parser.add_argument('--dataset_name', type=str, default='hs300', help='dataset name')
    parser.add_argument('--horizon', type=int, default=1, help='prediction horizon')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--heads', type=int, default=4, help='number of heads')
    parser.add_argument('--alpha', type=float, default=1, help='alpha parameter')
    parser.add_argument('--beta', type=float, default=2e-5, help='beta parameter')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--t_att_heads', type=int, default=6, help='number of temporal attention heads, need to be able to divide `win_size`')
    parser.add_argument('--gru_layers', type=int, default=1, help='number of GRU layers')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--rank_margin', type=float, default=0.1, help='rank margin')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    return parser.parse_args()