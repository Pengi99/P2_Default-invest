# --- 데이터 설정 ---
START_YEAR = 2013
END_YEAR = 2024
DATA_PATH = "./data"
API_KEY = "your_dart_api_key_here"

# --- 전처리 및 모델링 설정 ---
K1_SCORE_THRESHOLD = 5.0  # Adjusted to ensure a mix of bankrupt/non-bankrupt with synthetic data
TRAIN_END_YEAR = 2020
VALIDATION_END_YEAR = 2022

# --- 투자 전략 및 백테스팅 설정 ---
QUALITY_FACTOR = 'ROE'
QUALITY_QUANTILE = 0.2
TRANSACTION_COST = 0.002
REBALANCE_MONTH = 4
REBALANCE_DAY = 30

# --- Feature Engineering --- 
FEATURE_COLUMNS = ['X1', 'X2', 'X3', 'X4'] # Columns to be used as features for the model
