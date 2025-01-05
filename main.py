import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import torch
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader

from model import RTGNN
from utils.config import parse_args
from utils.predict import evaluate_and_predict



def main():
    args=parse_args()
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
            loss = RTGNN.combined_loss(output, target,args.alpha,args.beta,args.rank_margin)
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
            test_loss = RTGNN.MSE_loss(test_output, test_target)

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

    # calculate RankIC
    daily_rankICs = np.array([spearmanr(rankdata(preds[i]), rankdata(targets[i])).correlation for i in range(preds.shape[0])])

    # calculate RankICIR
    mean_rankIC = np.mean(daily_rankICs)
    std_dev_rankIC = np.std(daily_rankICs)
    RankICIR = mean_rankIC / std_dev_rankIC
    if mean_rankIC<0:
        return None
    print(mean_rankIC,RankICIR)

    # evaluate 
    portfolio = evaluate_and_predict(model, test_dataset, preds, targets, args, mean_rankIC, select_num)
    torch.save(model, f=f"models/RTGNN_{args.dataset_name}_{args.horizon}_{portfolio}.pth")

    return portfolio




if __name__=="__main__":
    portfolios=[]
    for i in range (50):
        ret=main()
        if ret !=None:
            portfolios.append(ret)
    print("max portfolios:",np.max(portfolios))