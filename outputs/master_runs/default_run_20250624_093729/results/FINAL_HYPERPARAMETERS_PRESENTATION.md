# 🎯 모델 최적화 결과 종합 리포트

## 📊 실행 개요
- **실행 ID**: default_run_20250624_093729
- **최적화 방법**: Optuna Bayesian Optimization
- **시행 횟수**: 각 모델당 100회 trials
- **평가 방법**: 5-fold Cross Validation

## 🏆 성능 순위

🥇 **XGBOOST_NORMAL**: 0.9386
�� **XGBOOST_COMBINED**: 0.9318
🥉 **RANDOMFOREST_COMBINED**: 0.9317
4️⃣ **RANDOMFOREST_NORMAL**: 0.9309
5️⃣ **ENSEMBLE_MODEL**: 0.9269
6️⃣ **LOGISTICREGRESSION_NORMAL**: 0.9148
7️⃣ **LOGISTICREGRESSION_COMBINED**: 0.9139


## 🔧 최적 하이퍼파라미터

### XGBOOST_NORMAL
**CV Score: 0.9386**

- **n_estimators**: 291
- **max_depth**: 10
- **learning_rate**: 0.126540
- **subsample**: 0.740653
- **colsample_bytree**: 0.794929
- **reg_alpha**: 6.395776
- **reg_lambda**: 7.944301

### XGBOOST_COMBINED
**CV Score: 0.9318**

- **n_estimators**: 376
- **max_depth**: 8
- **learning_rate**: 0.019550
- **subsample**: 0.663520
- **colsample_bytree**: 0.745999
- **reg_alpha**: 7.991699
- **reg_lambda**: 3.427986

### RANDOMFOREST_COMBINED
**CV Score: 0.9317**

- **n_estimators**: 317
- **max_depth**: 6
- **min_samples_split**: 15
- **min_samples_leaf**: 6
- **max_features**: 0.378459

### RANDOMFOREST_NORMAL
**CV Score: 0.9309**

- **n_estimators**: 330
- **max_depth**: 4
- **min_samples_split**: 9
- **min_samples_leaf**: 7
- **max_features**: 0.154944

### ENSEMBLE_MODEL
**CV Score: 0.9269**


### LOGISTICREGRESSION_NORMAL
**CV Score: 0.9148**

- **C**: 0.001517
- **max_iter**: 979
- **penalty**: l2
- **solver**: liblinear

### LOGISTICREGRESSION_COMBINED
**CV Score: 0.9139**

- **C**: 0.006657
- **max_iter**: 544
- **penalty**: l2
- **solver**: liblinear


## 📈 최적화 인사이트

### 🥇 최고 성능 모델 특징
**XGBOOST_NORMAL** (CV: 0.9386)


### 🎛️ 하이퍼파라미터 패턴 분석

**Logistic Regression**:
- 강한 정규화 (낮은 C 값) → 과적합 방지
- L2 penalty 선호 → 안정적인 계수

**Random Forest**:
- 적당한 트리 개수 (300-330) → 성능과 효율의 균형
- 제한된 깊이 (4-6) → 개별 트리 복잡도 제어

**XGBoost**:
- 높은 정규화 (reg_alpha, reg_lambda) → 복잡도 억제
- 적응적 샘플링 → 다양성 확보

---
*P2_Default-invest 프로젝트*
