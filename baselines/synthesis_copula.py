import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import time
import warnings
warnings.filterwarnings("ignore")

def generate(filename):
  train = pd.read_csv(f'./train_data/{filename}.csv')
  train_tmp = train.copy().dropna()
  
  metadata = SingleTableMetadata()
  metadata.detect_from_dataframe(train_tmp)
  synthesizer = CopulaGANSynthesizer(metadata)

  for i in range(5):
    start_train = time.time()
    synthesizer.fit(train_tmp)
    end_train = time.time()

    start_fit = time.time()
    synthetic_data = synthesizer.sample(num_rows=len(train))
    end_fit = time.time()

    print(f"    Time spent for {i}: training -> {end_train-start_train} sec, fitting -> {end_fit-start_fit} sec.")

    synthetic_data.to_csv(f'./copulagan/{filename}_{i}.csv', index=False)

print("Adult")
generate('adult')

print("Bejing")
generate('bejing')

print("California Housing")
generate('california_housing')

print("US location")
generate('us_location')

print("Seattle Housing")
generate('seattle_housing')

print("Travel")
generate('travel')
