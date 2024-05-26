from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#import linear regression model
from sklearn.linear_model import LinearRegression
import pickle

class accModel:

    def __init__(self):
        self.model = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('regressor', LinearRegression())           # The regression model
    ])

    def train(self, X, y):
        return self.model.fit(X,y)
    
    def save_model(self,model, model_name):
        with open(f'artifacts/{model_name}','wb') as f:
            pickle.dump(model, f)
