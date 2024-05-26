from sklearn.metrics import r2_score
import pandas as pd

class ModelEvaluation:

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X)
        y_pred = [x[0] for x in y_pred]
        result = r2_score(self.y, y_pred)
        return y_pred, result

    def save_evaluation(self,test_predictions:dict):
        test_pred_df = pd.DataFrame(test_predictions)
        final_predictions_df = pd.concat([test_pred_df, self.y.reset_index()], axis = 1)
        final_predictions_df.drop(columns = ['index'], inplace = True)
        final_predictions_df.rename(columns = {'5 epochs' : 'pred_acc_5', '10 epochs':'pred_acc_10', '20 epochs':'pred_acc_20', '30 epochs':'pred_acc_30', '60 epochs':'pred_acc_60'}, inplace = True)
        
        final_predictions_df.to_csv("output/predictions.csv")


        




        
