# 🎯 모델 하이퍼파라미터 최적화 결과

## 📊 실행 정보
- **실행 ID**: default_run_20250624_093729
- **최적화 방법**: Optuna
- **시행 횟수**: 100회
- **교차 검증**: 5-fold CV

## 🔧 최적화된 하이퍼파라미터

### 📈 Logistic Regression (Normal)

- **C**: 0.001517
- **penalty**: l2
- **solver**: liblinear
- **max_iter**: 979
- **l1_ratio**: None

### 📈 Logistic Regression (Combined)

- **C**: 0.006657
- **penalty**: l2
- **solver**: liblinear
- **max_iter**: 544
- **l1_ratio**: None

### 🌳 Random Forest (Normal)

- **n_estimators**: 330
- **max_depth**: 4
- **min_samples_split**: 9
- **min_samples_leaf**: 7
- **max_features**: 0.154944

### 🌳 Random Forest (Combined)

- **n_estimators**: 317
- **max_depth**: 6
- **min_samples_split**: 15
- **min_samples_leaf**: 6
- **max_features**: 0.378459

### 🚀 XGBoost (Normal)

- **n_estimators**: 291
- **max_depth**: 10
- **learning_rate**: 0.126540
- **subsample**: 0.740653
- **colsample_bytree**: 0.794929
- **reg_alpha**: 6.395776
- **reg_lambda**: 7.944301

### 🚀 XGBoost (Combined)

- **n_estimators**: 376
- **max_depth**: 8
- **learning_rate**: 0.019550
- **subsample**: 0.663520
- **colsample_bytree**: 0.745999
- **reg_alpha**: 7.991699
- **reg_lambda**: 3.427986


## 💡 주요 특징

### Logistic Regression
- **강한 정규화**: 낮은 C 값으로 과적합 방지
- **L2 정규화**: 안정적인 계수 추정
- **적절한 반복 횟수**: 수렴을 위한 충분한 iteration

### Random Forest  
- **적당한 트리 개수**: 과적합과 성능의 균형
- **제한된 깊이**: 개별 트리의 복잡도 조절
- **샘플링 제약**: min_samples 설정으로 일반화 성능 향상

### XGBoost
- **정교한 정규화**: L1/L2 정규화로 복잡도 제어
- **적응적 학습률**: 안정적인 학습을 위한 조정
- **부분 샘플링**: subsample/colsample로 다양성 확보

## 🎭 데이터 타입별 차이점

**Normal vs Combined 데이터**:
- Combined 데이터에서 일반적으로 더 복잡한 모델 구조
- 불균형 해소 후 더 높은 max_depth, 더 많은 estimators 경향
- 학습률과 정규화 강도의 미세 조정

---
*생성일시: 2025-01-24*
*프로젝트: P2_Default-invest*
