import pandas as pd
import numpy as np
class DataCleaning:

    def __init__(self, train_loss_df, eval_loss_df, eval_acc_df, hp_space_df):
        self.train_loss_df = train_loss_df
        self.eval_loss_df = eval_loss_df
        self.eval_acc_df = eval_acc_df
        self.hp_space_df =hp_space_df
    
    def compute_fixed_window_means(self,array, window_size=50):
        result = self.hp_space_df.to_dict(orient = 'list')
        for row in array:
            ep_start = 1
            row_means = []
            for i in range(0, len(row), window_size):
                window = row[i:i + window_size]
                window_mean = window.mean()
                row_means.append(window_mean)
                if f'epoch_{ep_start}' not in result:
                    result[f'epoch_{ep_start}'] = [window_mean]
                else:
                    result[f'epoch_{ep_start}'].append(window_mean)
                ep_start+=1
        return pd.DataFrame(result)

    def data_clean(self):
        self.eval_loss_df.drop(index =[93, 94], inplace = True)
        self.eval_acc_df.drop(index =[93, 94], inplace = True)
        self.eval_loss_df.reset_index(inplace = True)
        self.eval_loss_df.drop(columns = ['index'], inplace = True)
        self.eval_acc_df.reset_index(inplace = True)
        self.eval_acc_df.drop(columns = ['index'], inplace = True)

        self.preprocessed_train_loss_df = self.compute_fixed_window_means(self.train_loss_df.iloc[1:,4:].values.astype('float64'))

        remove_indices = self.eval_acc_df[self.eval_acc_df.epoch_150.isna()].index.values
        self.preprocessed_train_loss_df.drop(index = remove_indices, inplace = True)
        self.eval_loss_df.drop(index = remove_indices, inplace = True)
        self.eval_acc_df.drop(index = remove_indices, inplace = True)

        return self.preprocessed_train_loss_df, self.eval_loss_df, self.eval_acc_df
    
