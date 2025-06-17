
# 로지스틱 회귀 모델링 템플릿

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

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

# 3. 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 정의 및 학습
model = LogisticRegression(
    penalty='l2',        # 규제 종류: 'l1', 'l2', 'elasticnet', None
    C=1.0,                # 규제 강도: 작을수록 규제 강함
    solver='lbfgs',       # 'lbfgs', 'saga', 'liblinear' 등
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 5. 예측 및 평가
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. 모델 계수 확인 (옵션)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values(by='coefficient', key=abs, ascending=False)
print("Feature Coefficients:\n", coef_df)

# 실행 예시:
# python logistic_regression_modeling.py
