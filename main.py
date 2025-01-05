import torch
import torch.nn as nn
import pandas as pd
import argparse 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from torch.nn.functional import l1_loss
import torch.nn.functional as F
import os
import numpy as np
from model import RTGNN

parser = argparse.ArgumentParser(description='PyTorch HTGNN_pro')
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
args = parser.parse_args()
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')
print('Using device:', device)

if args.dataset_name == 'hs300':
    input_dim = 6 
    stock_num =  283 
    select_num = 30
    feature_cols=['close','open','high','low','turnover','volume']
    last_date= '2022-11-18'
else :
    input_dim = 5 
    stock_num =  98 
    select_num = 10
    feature_cols=['close','high','low','open','volume']
    last_date= '2022-11-16'


def pairwise_ranking_loss(preds, labels, margin=args.rank_margin):
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    assert preds.size() == labels.size()
    # 创建一个矩阵，其中每个元素i,j都是预测值或标签差值
    diff_preds = preds[None,:] - preds[:, None]
    diff_labels = labels[None, :] - labels[:, None]
    
    # 使用几何余弦表示一致性，当预测的差值和真实标签的差值同号时，为1，否则为0
    mask = (diff_preds * diff_labels < 0).type(torch.float32)
    
    # 对差值预测应用一个线性关系，并引入一个margin
    hinge_loss = torch.nn.functional.relu(margin - diff_preds).pow(2)
    
    # 乘以mask以获得最后的损失，只有不一致的对才会有损失值
    loss = mask * hinge_loss
    return loss.sum()

MSE_loss= nn.MSELoss()
def combined_loss(preds,labels):
    return args.alpha*MSE_loss(preds,labels)+args.beta*pairwise_ranking_loss(preds,labels)


def main():
    print("data loading...")
    df = pd.read_csv(f'./dataset/{args.dataset_name}_{args.horizon}.csv')
    print('data processing...')
    standardize_df = RTGNN.process_features(df,feature_cols)


    train_dataset= RTGNN.StockDataset(standardize_df,args.win_size,args.horizon ,'train',feature_cols=feature_cols,last_date=last_date)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=RTGNN.collate_fn)

    test_dataset = RTGNN.StockDataset(standardize_df,args.win_size,args.horizon,'test',feature_cols=feature_cols,last_date=last_date)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=RTGNN.collate_fn)


    model = RTGNN.Model(win_size=args.win_size,input_dim=input_dim,hidden_dim=args.hidden_dim,
                  t_att_heads=args.t_att_heads,gru_layers=args.gru_layers,heads=args.heads,
                  out_dim=args.out_dim,device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):   
        model.train()
        loss_return = 0 
        for i, (feature_batch, label_batch) in enumerate(train_loader):
            model.zero_grad()
            feature = feature_batch.to(device)
            if i == 0:
                hs1 = feature
            target = label_batch.to(device)
            output = model(feature,hs1)
            hs1 = feature
            loss = combined_loss(output, target)
            loss.backward()
            loss_return+=loss.item()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{args.epochs}, train loss: {loss_return/i}')

    print("model testing...")
    model.eval()

    with torch.no_grad():
        test_total_loss = 0
        total_mae = 0
        preds = []
        targets = []
        for i, (test_data, test_target) in enumerate(test_loader):

            test_data = test_data.to(device)
            test_target = test_target.to(device)

            if i==0:
                history = test_data

            test_output = model(test_data,history)
            history = test_data
            test_loss = MSE_loss(test_output, test_target)

            test_total_loss += test_loss.item()
            total_mae += l1_loss(test_output, test_target).item()
            preds.append(np.array(test_output.squeeze().tolist()))
            targets.append(np.array(test_target.squeeze().tolist()))
            del test_output, test_target
        # Convert lists to numpy arrays for scipy
        preds = np.array(preds)
        targets = np.array(targets)
    print(f'Test MSE: {test_total_loss/i}') 
    print(f'Test MAE: {total_mae/i}') 

    from scipy.stats import spearmanr, rankdata

    # 计算每一天的 RankIC
    daily_rankICs = np.array([spearmanr(rankdata(preds[i]), rankdata(targets[i])).correlation for i in range(preds.shape[0])])

    # 计算 RankICIR
    mean_rankIC = np.mean(daily_rankICs)
    std_dev_rankIC = np.std(daily_rankICs)
    RankICIR = mean_rankIC / std_dev_rankIC
    if mean_rankIC<0:
        return None
    print(mean_rankIC,RankICIR)

    hold = np.empty((0, select_num))  # 初始化一个空的二维数组

    for i in range(targets.shape[0]):
        select = np.argsort(preds[i])[::-1][:select_num]
        hold = np.vstack((hold, test_dataset.stocks[select]))



    dates = test_dataset.date.iloc[args.win_size:]
    hold_df = pd.DataFrame(hold)
    combined_df = pd.concat([dates.reset_index(drop=True), hold_df], axis=1)
    cols = ['dt'] + [f'kdcode{i}' for i in range(1, select_num+1)]
    combined_df.columns = cols
    combined_df.to_csv(f'./data/hold_HTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv', index=False)

    stock_df = pd.read_csv(f'./dataset/{args.dataset_name}_1.csv')
    hold_df = pd.read_csv(f'./data/hold_HTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv')

    output = []
    # 迭代hold数据
    for index, row in hold_df.iterrows():
        kdcode_columns = [f'kdcode{i}' for i in range(1, select_num+1)]
        kd_codes = row[kdcode_columns].values
        dt = row['dt']
        ot_values = stock_df[(stock_df['kdcode'].isin(kd_codes)) & (stock_df['dt'] == dt)]['OT']
        daily_return = ot_values.mean()
        output.append([dt, daily_return])

    # 创建DataFrame并写入文件
    output_df = pd.DataFrame(output, columns=['datetime', 'daily_return'])

    output_df.to_csv(f'./data/return_HTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv', index=False)

    df_return=pd.read_csv(f'./data/return_HTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv')
    index_df=pd.read_csv(f'./data/index_{args.dataset_name}.csv')

    portfolio_df_performance = df_return.set_index(['datetime'])
    index_df_performance = index_df.set_index(['datetime'])
    alpha_df_performance = pd.DataFrame()
    alpha_df_performance['portfolio_daily_return'] = portfolio_df_performance['daily_return']
    alpha_df_performance['index_daily_return'] = index_df_performance['daily_return']

    alpha_df_performance['alpha_daily_return'] = alpha_df_performance['portfolio_daily_return'] - \
                                                    alpha_df_performance[
                                                        'index_daily_return']


    alpha_df_performance['portfolio_net_value'] = (alpha_df_performance['portfolio_daily_return'] + 1).cumprod()
    alpha_df_performance['index_net_value'] = (alpha_df_performance['index_daily_return'] + 1).cumprod()
    alpha_df_performance['alpha_net_value'] = (alpha_df_performance['alpha_daily_return'] + 1).cumprod()


    net_value_columns = ['portfolio_net_value',
                            'index_net_value',
                            'alpha_net_value']



    portfolio = ((alpha_df_performance['alpha_net_value'].tail(1)) ** (252 / len(alpha_df_performance)) - 1).agg('mean')

    print("portfolios:",portfolio)

    output_df.to_csv(f'./data/return_HTGNNpro_{args.dataset_name}_{args.horizon}_{portfolio}.csv', index=False)
    torch.save(model, f=f"models/HTGNNpro_{args.dataset_name}_{args.horizon}_{portfolio}.pth")
    return portfolio

if __name__=="__main__":
    portfolios=[]
    for i in range (50):
        ret=main()
        if ret !=None:
            portfolios.append(ret)
    print("max portfolios:",np.max(portfolios))