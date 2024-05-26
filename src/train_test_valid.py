import pandas as pd


class TrainTestValidSplit:

    def __init__(self,train_indices, valid_indices, test_indices):
        self.train = train_indices
        self.valid = valid_indices
        self.test = test_indices

    def early_stop_dataframe(self, early_stop_epoch, preprocessed_train_loss_df,eval_loss_df,eval_acc_df,hp_space_df):
        t_loss_df = preprocessed_train_loss_df[f'epoch_{early_stop_epoch}']
        e_loss_df = eval_loss_df[f'epoch_{early_stop_epoch}']
        e_acc_df = eval_acc_df[[f'epoch_{early_stop_epoch}', 'epoch_150']]
        df = pd.concat([hp_space_df,t_loss_df, e_loss_df,e_acc_df], axis = 1)
        df.rename(columns = {'epoch_150':'output_acc_150'}, inplace = True)

        return df
    
    def train_test_valid_split(self, early_stop_epoch, preprocessed_train_loss_df,eval_loss_df,eval_acc_df,hp_space_df):
        df = self.early_stop_dataframe(early_stop_epoch, preprocessed_train_loss_df,eval_loss_df,eval_acc_df,hp_space_df)

        train_df = df[df.index.isin(self.train)]
        validation_df = df[df.index.isin(self.valid)]
        test_df = df[df.index.isin(self.test)]

        x_train = train_df.drop(columns = 'output_acc_150')
        x_val = validation_df.drop(columns = 'output_acc_150')
        x_test = test_df.drop(columns = 'output_acc_150')

        y_train = train_df[['output_acc_150']]
        y_val = validation_df[['output_acc_150']]
        y_test = test_df[['output_acc_150']]

        x_test.dropna(inplace = True)
        y_test.dropna(inplace = True)

        return x_train,y_train, x_val, y_val, x_test, y_test