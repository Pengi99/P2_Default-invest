
# XGBoost 분류 모델링 템플릿

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

# 1. 데이터 로드: 사용자의 데이터 프레임(df)을 준비하고, 특성과 타깃으로 분리
# 예시:
# df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1)
# y = df['target_column']

# TODO: 아래 두 줄을 실제 데이터에 맞게 수정하세요
X = pd.DataFrame()  # 특성 행렬
y = pd.Series()     # 타깃 벡터

# 2. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 불균형 데이터일 경우 사용
)

# 3. DMatrix 생성 (XGBoost 전용)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 4. 하이퍼파라미터 설정
params = {
    'objective': 'binary:logistic',  # 이진 분류
    'eval_metric': 'auc',            # 평가 지표
    'eta': 0.1,                      # 학습률
    'max_depth': 4,                  # 트리 최대 깊이
    'subsample': 0.8,                # 샘플링 비율
    'colsample_bytree': 0.8,         # 특성 샘플링 비율
    'seed': 42
}
num_rounds = 100  # 부스팅 라운드 수

# 5. 모델 학습 및 평가
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=evals,
    early_stopping_rounds=10
)

# 6. 예측 및 성능 평가
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. 특성 중요도 확인 (옵션)
importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values(by='importance', ascending=False)
print("Feature Importances:\n", importance_df)

# 실행 예시:
# python xgboost_modeling.py
