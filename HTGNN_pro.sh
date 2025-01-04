python HTGNN_wo_encoder.py --win_size 30 \
                    --dataset_name hs300 \
                    --horizon 1 \
                    --hidden_dim 128 \
                    --out_dim 1 \
                    --heads 4 \
                    --alpha 1 \
                    --beta 2e-5 \
                    --epochs 60 \
                    --t_att_heads 6 \
                    --gru_layers 1 \
                    --lr 2e-4 \
                    --rank_margin 0.1 \
                    --gpu 2 \
                    >> HTGNN_hs300_wo_encoder.log

python HTGNN_wo_graph.py --win_size 30 \
                    --dataset_name hs300 \
                    --horizon 1 \
                    --hidden_dim 128 \
                    --out_dim 1 \
                    --heads 4 \
                    --alpha 1 \
                    --beta 2e-5 \
                    --epochs 60 \
                    --t_att_heads 6 \
                    --gru_layers 1 \
                    --lr 2e-4 \
                    --rank_margin 0.1 \
                    --gpu 2 \
                    >> HTGNN_hs300_wo_graph.log

python HTGNN_wo_rank.py --win_size 30 \
                    --dataset_name hs300 \
                    --horizon 1 \
                    --hidden_dim 128 \
                    --out_dim 1 \
                    --heads 4 \
                    --alpha 1 \
                    --beta 0 \
                    --epochs 60 \
                    --t_att_heads 6 \
                    --gru_layers 1 \
                    --lr 2e-4 \
                    --rank_margin 0.1 \
                    --gpu 2 \
                    >> HTGNN_hs300_wo_rank.log