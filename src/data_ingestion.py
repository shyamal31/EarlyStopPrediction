import pandas as pd
import numpy as np

class DataIngestion:

    def __init__(self, train_loss_path=None, eval_loss_path=None, eval_acc_path = None, hp_space_path= None):
        self.train_loss_path = train_loss_path
        self.eval_loss_path = eval_loss_path
        self.eval_acc_path = eval_acc_path
        self.hp_space_path = hp_space_path
    
    def create_dataframes(self, path = None):
        if path:
            df = pd.read_csv(path, index_col = 0)
            return df
        train_loss_df = pd.read_csv(self.train_loss_path)
        eval_loss_df = pd.read_csv(self.eval_loss_path)
        eval_acc_df = pd.read_csv(self.eval_acc_path)
        hp_space_df = pd.read_csv(self.hp_space_path)

        return train_loss_df, eval_loss_df, eval_acc_df, hp_space_df

