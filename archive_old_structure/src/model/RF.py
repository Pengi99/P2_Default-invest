
# Random Forest 분류 모델링 템플릿

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# 3. 모델 정의 및 학습
model = RandomForestClassifier(
    n_estimators=100,      # 트리 개수
    max_depth=None,        # 트리 최대 깊이
    min_samples_split=2,   # 내부 노드 분할 최소 샘플 수
    min_samples_leaf=1,    # 리프 노드 최소 샘플 수
    max_features='auto',   # 특성 선택 방식
    random_state=42,
    n_jobs=-1              # 병렬 처리
)
model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
# 이진 분류의 경우
try:
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
except Exception:
    auc = None

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
if auc is not None:
    print(f"ROC AUC: {auc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. 특성 중요도 확인 (옵션)
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
print("Feature Importances:\n", feat_df)

# 실행 예시:
# python random_forest_modeling.py
