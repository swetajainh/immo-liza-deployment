import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from catboost import CatBoostRegressor

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.numerical_cols = ['zip_code', 'latitude', 'longitude', 'construction_year', 'total_area_sqm', 'surface_land_sqm', 'nbr_frontages', 'nbr_bedrooms', 'terrace_sqm', 'garden_sqm', 'primary_energy_consumption_sqm', 'cadastral_income']
        self.categorical_cols = ['region', 'province', 'equipped_kitchen', 'state_building', 'epc', 'heating_type']

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df_mod = self.df.drop(columns=['fl_floodzone', 'fl_swimming_pool', 'fl_garden', 'fl_terrace', 'fl_open_fire', 'fl_furnished', 'fl_double_glazing', 'locality'])
        self.df_houses = self.df_mod[self.df_mod['subproperty_type'] == 'HOUSE']

    def preprocess_data(self):
        X = self.df_houses.drop(columns=['price'])
        y = self.df_houses[['price']]

        # Initialize the KFold object
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)

        # Iterate over the splits and obtain the training and testing indices
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        self.X_train = X_train.drop(columns=['id', 'property_type', 'subproperty_type'])
        self.X_test = X_test.drop(columns=['id', 'property_type', 'subproperty_type'])
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='passthrough'
        )

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', CatBoostRegressor())
        ])

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)

        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)

        print("Model score on training set:", train_score)
        print("Model score on test set:", test_score)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)

    def predict_price(self, data):
        X_new = pd.DataFrame(data)
        predicted_price = self.model.predict(X_new)
        return predicted_price

# Example usage:
if __name__ == "__main__":
    # Initialize the HousePricePredictor object
    predictor = HousePricePredictor()

    # Load and preprocess the data
    predictor.load_data('properties.csv')
    predictor.preprocess_data()

    # Build and train the model
    predictor.build_model()
    predictor.train_model()

    # Evaluate the model
    predictor.evaluate_model()

    # Save the model
    predictor.save_model('cat_model.pkl')

    # Load the saved model
    predictor.load_model('cat_model.pkl')

    # Define new data for prediction
    new_data = {
        'region': ['Flanders'],
        'province': ['Antwerp'],
        'locality': ['Antwerp'],
        'zip_code': [2050],
        'latitude': [51.217172],
        'longitude': [4.379982],
        'construction_year': [2020],
        'total_area_sqm': [100],
        'surface_land_sqm': [200],
        'nbr_frontages': [2],
        'nbr_bedrooms': [3],
        'equipped_kitchen': ['HYPER_EQUIPPED'],
        'terrace_sqm': [20],
        'garden_sqm': [50],
        'state_building': ['New'],
        'primary_energy_consumption_sqm': [150],
        'epc': ['C'],
        'heating_type': ['GAS'],
        'cadastral_income': [1000]
    }

    # Make predictions
    predicted_price = predictor.predict_price(new_data)
    print("Predicted price:", predicted_price)
