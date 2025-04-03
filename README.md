README.txt
==========

### 1. List of Unit Tests and Descriptions 

The unit tests are located in the folder "Unittest".

(a) Test_dataLoader:
- test_hug3874: Verifies all extracted features (returns, VWAP, volume, imbalance, arrival/terminal price) from TAQ data

(b) Test_dataProcessor:
- test_jug3478: Test the value-based features calculated from VWAP and volume-based features
- test_sig8234: Verifies feature DataFrames (stock * date) are melt to the final input dataset

(c) Test_impactModel:
- test_sfh1734: Verifies the Almgren Chriss Market Impact Model non-linear equation
- test_jrs1345: Test the non-linear fitting of the model
- test_owi2853: Test the non-linear fitting for volatile and normal regimes
- test_kwg8327: Test the non-linear fitting for active and inactive regimes
- test_wfb3875: Test the paired bootstrap (well conditioned data)
- test_ghy2764: Test the paired bootstrap (poorly conditioned data)
- test_isu2725: Test the While Test for heteroskedasticity

(d) Test_MidQuote:
- test_duh7246: Test the function to calculate the midquote return from Quote data

(e) Test_GetVolatility:
- test_shr8457: Test the calculation of volatility from 2-min returns

### 2. List of Files
- data/: Stores raw TAQ data files.
  - quotes/: TAQ quote data (yyyymmdd.tar.gz...)
  - trades/: TAQ trade data (yyyymmdd.tar.gz...)
- Helper/: Stores self-defined helper classes
  - MyDirectories.py: Defines base directory getters
  - MidQuote.py: Used to calculate the midquote return from Quote data
  - GetVolatility.py: Used to calculate volatility from 2-min returns
- scripts/: Core model logic
  - dataLoader.py: Load data from TAQ, stock selection, feature extraction, save to CSVs
  - dataPreprocessor.py: Process the raw features, calculate features, integrate to a CSV 
  - impactModel.py: Fit the non-linear model, result analysis
  - params_part1.txt: the estimated eta, beta and their t values
- Unittest/: Unit test files
- Utilities/: Given utility modules (TAQReader, VWAP, TickTest...)
- input/: CSVs generated from dataLoader.py, containing one matrix for each of the features (stock * dates)
- input.csv: final input data, one data point per stock per day
- stock_list.json: selected top 1500 stocks (generated if not exist)

### 3. Directories to be Created
Please create the data/ folder on the root directory. This should contain the quote data folder and trade data folder.

### 4. Python Packages
- Third Party Packages:
  - numpy
  - pandas
  - scipy
  - tqdm
- Standard Packages:
  - unittest
  - os
  - sys
  - json
  - ast
  - tarfile



