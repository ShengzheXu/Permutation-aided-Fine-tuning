import pandas as pd
from ctgan import CTGAN
import time

ctgan = CTGAN(epochs=10)

def generate(discrete_columns, filename):
    train = pd.read_csv(f'./train_data/{filename}.csv')
    train_tmp = train.copy().dropna()

    for i in range(5):
        start_train = time.time()
        ctgan.fit(train_tmp, discrete_columns)
        end_train = time.time()

        # Create synthetic data
        start_fit = time.time()
        synthetic_data = ctgan.sample(len(train))
        end_fit = time.time()

        print(f"    Time spent for {i}: training -> {end_train-start_train} sec, fitting -> {end_fit-start_fit} sec.")

        synthetic_data.to_csv(f'./ctgan/{filename}_{i}.csv', index=False)

print("Adult")
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]
generate(discrete_columns, 'adult')

print("Bejing")
discrete_columns = [
    'year',
    'month',
    'day',
    'hour',
    'DEWP',
    'cbwd',
    'Is',
    'Ir'
]
generate(discrete_columns, 'bejing')

print("California Housing")
discrete_columns = [
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'ocean_proximity'
]
generate(discrete_columns, 'california_housing')

print("US Location")
discrete_columns = [
    'state_code',
    'bird',
    'lat_zone'
]
generate(discrete_columns, 'us_location')

print("Seattle Housing")
discrete_columns = [
    'beds',
    'size',
    'size_units',
    'lot_size_units',
    'zip_code'
]
generate(discrete_columns, 'seattle_housing')

print("Travel")
discrete_columns = [
      'Age',
      'FrequentFlyer',
      'AnnualIncomeClass',
      'ServicesOpted',
      'AccountSyncedToSocialMedia',
      'BookedHotelOrNot',
      'Target']
generate(discrete_columns, 'travel')
