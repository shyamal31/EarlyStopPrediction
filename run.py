from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaning
from src.train_test_valid import TrainTestValidSplit
from src.model_selection_training import accModel
from src.model_evaluation import ModelEvaluation
import pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd

def fullTraining(save_models = False):
    root_path = "Data" 
    train_loss_path = root_path +"/train_loss.csv"
    eval_loss_path = root_path +"/eval_loss.csv"
    eval_acc_path = root_path +"/eval_acc.csv"
    hp_space_path = root_path +"/HP_space.csv"
    di_obj = DataIngestion(train_loss_path, eval_loss_path, eval_acc_path, hp_space_path)
    train_loss_df, eval_loss_df, eval_acc_df, hp_space_df = di_obj.create_dataframes()
    dc = DataCleaning(train_loss_df, eval_loss_df, eval_acc_df, hp_space_df)
    preprocessed_train_loss_df, eval_loss_df, eval_acc_df = dc.data_clean()

    #train-test-valid split
    with open('Data/dataPartition.txt', 'r') as file:
        data = file.read()

    # Parse the lists
    train_data = data.split('train: ')[1].split('validation: ')[0].strip().strip('[]').split(', ')
    validation_data = data.split('validation: ')[1].split('test: ')[0].strip().strip('[]').split(', ')
    test_data = data.split('test: ')[1].strip().strip('[]').split(', ')

    # Convert string data to integers
    train = [int(i) for i in train_data]
    validation = [int(i) for i in validation_data]
    test = [int(i) for i in test_data]

    ttv = TrainTestValidSplit(train, validation, test)

    #create train test valid split according to early stop epoch
    x_train_5, y_train, x_val_5, y_val, x_test_5, y_test = ttv.train_test_valid_split(5,preprocessed_train_loss_df, eval_loss_df, eval_acc_df, hp_space_df)
    x_train_10, _, x_val_10, _, x_test_10, _ = ttv.train_test_valid_split(5,preprocessed_train_loss_df, eval_loss_df, eval_acc_df, hp_space_df)
    x_train_20, _, x_val_20, _, x_test_20, _ = ttv.train_test_valid_split(5,preprocessed_train_loss_df, eval_loss_df, eval_acc_df, hp_space_df)
    x_train_30, _, x_val_30, _, x_test_30, _ = ttv.train_test_valid_split(5,preprocessed_train_loss_df, eval_loss_df, eval_acc_df, hp_space_df)
    x_train_60, _, x_val_60, _, x_test_60, _ = ttv.train_test_valid_split(5,preprocessed_train_loss_df, eval_loss_df, eval_acc_df, hp_space_df)

    train_test_data = {
    '5 epochs': (x_train_5,x_test_5),
    '10 epochs': (x_train_10,x_test_10),
    '20 epochs': (x_train_20,  x_test_20),
    '30 epochs': (x_train_30, x_test_30),
    '60 epochs': (x_train_60, x_test_60)
}
    test_predictions = {}
    test_results = {}
    eval = None
    for test_data_name, (x_train,x_test) in train_test_data.items():
        accM_obj= accModel()
        acc_model = accM_obj.train(np.array(x_train),np.array(y_train))
        if save_models:
            accM_obj.save_model(acc_model, f'model_{test_data_name}')
        
        eval = ModelEvaluation(acc_model, x_test, y_test)
        y_pred, score = eval.evaluate_model()
        test_predictions[test_data_name] = y_pred
        test_results[test_data_name]= score
    
    eval.save_evaluation(test_predictions)

def usePreTrainedModel(early_stop_epoch):
    test_predictions = {}
    di = DataIngestion()
    test_file_path = f"Data/Test/Epoch {early_stop_epoch}/XTest{early_stop_epoch}.csv"
    x_test = di.create_dataframes(test_file_path)
    y_test = di.create_dataframes("Data/Test/yTest.csv")
    model_file_path = f"PreTrainedModels/model_{early_stop_epoch} epochs"
    model = pickle.load(open(model_file_path, 'rb'))

    eval = ModelEvaluation(model, x_test, y_test)
    y_pred, score = eval.evaluate_model()
    test_predictions['original_acc_150'] = list(y_test['output_acc_150'])
    test_predictions['pred_acc_{early_stop_epoch}'] = y_pred
    pd.DataFrame(test_predictions).to_csv(f'output/epoch_{early_stop_epoch}_predictions.csv')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-s','--save', help= "Saves the trained models", type = bool,choices=[True, False])
    parser.add_argument('-p', '--pretrained', help = "Uses Already Trained Models to make predictions", type =int, choices = [5,10,20,30,60])

    args = parser.parse_args()
    if args.pretrained is not None:
        usePreTrainedModel(args.pretrained)
    else:
        if args.save:
            fullTraining(True)
        else:
            fullTraining()




    





