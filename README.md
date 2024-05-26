# Early Stop via Prediction for Hyperparameter Tuning
## Introduction

DNN model training takes time. In hyperparameter tuning (i.e., finding the best values for some hyperparameters for a DNN model), we would need to train the DNN model repeatedly on each of the hyperparameter values and run the test. It would take even more time.

The objective of this project is to see whether we can predict the final accuracy of a DNN model for given hyperparameters values, without training the DNN model completely. In another word, can we tell how good the hyperparamter values are by just observing the results in the first small number of training eporches of the DNN model? If we can do that, it can save a lot of time in hyperparameter tuning. 
 
The method we want to try in this project is to build a Machine Learning model (called accModel) to make the prediction. For a given DNN X and a set of k hyperparameters H=<h1, h2, ..., hk> to tune, the input to the accModel includes (i) a sample hyperparameter vector Hi=<h1i, h2i, ..., hki>, where hki is the sample value of the kth hyperparameter; (ii) the observed training and validation loss values and accuracy values in the first E eporchs; (iii) a positive integer M (M>E). The output of the accModel is the predicted validation accuracy of DNN X after M eporchs of training with Hi as the hyperparameter vector. 

## Dataset

The provided dataset were obtained by training **<u>CifarNet</u>** on the **Cifar-10** dataset. *CifarNet* contains two convolutional layers. There are two hyperparameters, ***L*** and ***H***, at each of the two convolutional layers. So there are four hyperparameters in the hyperparameter vector of *CifarNet*: <L1, H1, L2, H2>. 

* HP_space.csv: the sampled hyperparameter vector values. 

* train_loss.csv: the **training loss values**. The file is organized as follows:

  - Each row gives the observations on *CifarNet* on one sampled hyperparameter vector.  
  - The first 4 columns indicate the hyperparameter vector value (it's the same as in `HP_space.csv`). 
  - The following columns show the training losses. 150 epochs are recorded, and each epoch contains 50 mini-batches.

* eval_loss.csv: the **validation loss values** of *CifarNet*. It is in a similar format as `train_loss.csv`, but reports only one loss in each epoch.

* eval_acc.csv: the **validation accuracies** of *CifarNet*. It is in a similar format as `train_loss.csv`, but reports only one accuracy in each epoch.

* dataPartition.txt: the samples (i.e., line numbers in the above data files) to be used as training dataset, validation dataset, and test dataset for the development of accModel.

## Approach

To tackle this problem, we created separate machine learning models for each early stop setting. For each model, we transformed the dataset appropriately using the provided data files. These models were then tested on different test datasets.

Given that this is a regression task (predicting a continuous value), we tried various regression models. Among all the models tested, linear regression provided the best results. Although the Random Forest regressor also showed promising results, it required more computational resources for training. Therefore, we selected linear regression as our accModel due to its efficiency and comparable performance.

### Steps in Detail

1. Data Preprocessing: 
    - Read the hyperparameter values and performance metrics from the provided CSV files.
    - Organize the data to match the input format required by the `accModel`.

2. Model Training: 
    - For each early stop setting `(E=5, 10, 20, 30, 60)`, we trained a separate linear regression model.
    - The input features included the hyperparameter values and performance metrics from the first `E` epochs.

3. Model Evaluation: 
    - Evaluate the `accModel` on the test dataset.
    - Report the predicted validation accuracy after 150 epochs.

By implementing this approach, we aim to significantly reduce the time required for hyperparameter tuning by accurately predicting the final validation accuracy of the DNN model based on early epoch data. This method leverages machine learning to streamline the hyperparameter tuning process, making it more efficient and less time-consuming.

## File Structure
- **Data** : This folder contains data from Ciphar-10 model training and evaluation. It also contains test data used in this project.
- **src** : This folder contains entire logic of this project.
    - **data_ingestion.py** : Accepts input data.
    - **data_cleaning.py** : Handles primary data cleaning and transformation process
    - **train_test_valid.py** : Split data into train, validation and test split
    - **model_selection_training.py** : Trains regression models(accModel) on different early stop datasets.
    - **model_evaluation** : Provides predictions and r2-score generated by accModel.
- **run.py** : Contains driver code.
- **artifacts** : This folder contains saved accModels while training.
- **PreTrainedModels** : This folder contains previously trained models.
- **output** : This folder contains generated prediction files.
- **HPEarlyStopping.ipynb** : Python notebook which provides detailed explanation of entire project. Good if you want to first experiement with data and models.
- **README.md** : This markdown file you are reading.
- **requirements.txt** : Required imports to run the program successfully.


## Installation
(Optional) setup a virtual environment to install necessary packages using the following command:
``` commandline
virtualenv .venv
source .venv/bin/activate
```
Install the packages listed in Requirements.txt file
```shell
pip install -r requirements.txt
```
Train all the models from scratch without saving the models
```shell
python3 run.py 
```
Train all the models from scratch and save the models
```shell
python3 run.py -s True
```
OR
```shell
python3 run.py --save True
```
Use pretrained models to generate results directly. 
```shell
python3 run.py -p {early_epoch}
```
OR
```shell
python3 run.py --pretrained {early_epoch}
```
**Remember to replace early_epoch from [5,10,20,30,60]**


## Support
You may contact author for any support.
* __Email__: [sgandhi6@ncsu.edu](mailto:sgandhi6@ncsu.edu?subject=%20Algorithm%20Query)
* __Github__: [shyamal31](https://github.com/shyamal31)

## Acknowledgement 
I Would like to give credits to 
* [Build Command-Line Interfaces with python's argparse](https://realpython.com/command-line-interfaces-python-argparse/)
* [pandas documentation](https://pandas.pydata.org/docs/)
