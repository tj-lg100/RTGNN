import pandas as pd
import numpy as np

def evaluate_and_predict(model, test_dataset, preds, targets, args, mean_rankIC, select_num):
    hold = np.empty((0, select_num))  

    for i in range(targets.shape[0]):
        select = np.argsort(preds[i])[::-1][:select_num]
        hold = np.vstack((hold, test_dataset.stocks[select]))

    dates = test_dataset.date.iloc[args.win_size:]
    hold_df = pd.DataFrame(hold)
    combined_df = pd.concat([dates.reset_index(drop=True), hold_df], axis=1)
    cols = ['dt'] + [f'kdcode{i}' for i in range(1, select_num+1)]
    combined_df.columns = cols
    combined_df.to_csv(f'./data/hold_RTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv', index=False)

    stock_df = pd.read_csv(f'./dataset/{args.dataset_name}_1.csv')
    hold_df = pd.read_csv(f'./data/hold_RTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv')

    output = []
   
    for index, row in hold_df.iterrows():
        kdcode_columns = [f'kdcode{i}' for i in range(1, select_num+1)]
        kd_codes = row[kdcode_columns].values
        dt = row['dt']
        ot_values = stock_df[(stock_df['kdcode'].isin(kd_codes)) & (stock_df['dt'] == dt)]['OT']
        daily_return = ot_values.mean()
        output.append([dt, daily_return])

    
    output_df = pd.DataFrame(output, columns=['datetime', 'daily_return'])

    output_df.to_csv(f'./data/return_RTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv', index=False)

    df_return=pd.read_csv(f'./data/return_RTGNNpro_{args.dataset_name}_{args.horizon}_{mean_rankIC}.csv')
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
    output_df.to_csv(f'./data/return_RTGNNpro_{args.dataset_name}_{args.horizon}_{portfolio}.csv', index=False)
    return portfolio