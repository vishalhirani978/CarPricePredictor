from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)


def train_model():
    car = pd.read_csv('cleanedCar.csv')
    
    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']

    ohe = OneHotEncoder()
    ohe.fit(X[['name', 'company', 'fuel_type']])

    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
        remainder='passthrough'
    )

    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X, y)
    return pipe

car = pd.read_csv('cleanedCar.csv')
model = train_model()

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    ))
    return str(round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
