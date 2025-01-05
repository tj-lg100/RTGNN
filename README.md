# Relational Temporal Graph Neural Networks (RTGNN)

## 1. Prepare you training data

## 2. Train you model
* Before training, make sure to change the parameters in `main.sh` .

 ``` config
win_size=30          # The window size for slicing the time series data
dataset_name=hs300   # The name of the dataset being used
horizon=1            # The prediction horizon; specifies how many future steps the model will predict.
hidden_dim=128       # Dimensionality of the hidden layers in the model
out_dim=1            # Dimensionality of the model's output
heads=4              # Number of attention heads in the graph attention network
alpha=1              # Hyperparameter balancing different loss terms
beta=2e-5            # Regularization weight to prevent overfitting
epochs=60            # Number of epochs for training
t_att_heads=6        # Number of attention heads in the temporal attention mechanism
gru_layers=1         # Number of GRU layers
lr=2e-4              # Learning rate for the optimizer
rank_margin=0.1      # Margin in ranking loss for pairwise tasks
gpu=1                # Specifies which GPU to use for training
 ```

* Install required packages

  ``` shell
  pip install -r requirements.txt
  ```

* Training 

  ``` shell
  bash main.sh
  ```

  ## 3. Citing

* If you find **RTGNN** is useful for your research, please consider citing the following papers:

  ``` latex
  
  ```