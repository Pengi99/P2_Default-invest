# 📊 모델링 (Modeling)

한국 기업 부실예측을 위한 머신러닝 모델링 파이프라인입니다.

## 🎯 주요 기능

### 1. **🆕 마스터 모델 러너** (통합 파이프라인)
- **자동화된 모델 실행**: LogisticRegression, RandomForest, XGBoost 일괄 실행
- **🔥 자동 Threshold 최적화**: 각 모델별 최적 임계값 자동 탐색
- **🎭 앙상블 모델**: 여러 모델을 결합한 앙상블 예측 (NEW!)
- **중앙 설정 관리**: JSON 기반 설정으로 모든 하이퍼파라미터 관리
- **Lasso 특성 선택**: 선택적 특성 선택 기능
- **체계적 저장**: 실행별 폴더 생성 및 결과 관리

### 2. 기본 모델링
- **로지스틱 회귀** (`logistic_regression_100.py`)
- **랜덤 포레스트** (`RF_100.py`) 
- **XGBoost** (`xgboost_100.py`)

### 3. 모델 비교 및 분석
- **`model_comparison.py`**: 일반 데이터 모델 비교
- **`model_comparison_normal.py`**: Normal vs SMOTE 비교

## 🚀 마스터 러너 사용법

### ⚡ 빠른 시작

```bash
# 빠른 테스트 (적은 trials, threshold 최적화 포함)
python src_new/modeling/run_master.py --template quick

# 프로덕션 실행 (많은 trials, 완전한 최적화)
python src_new/modeling/run_master.py --template production

# Lasso 집중 분석 (특성 선택 중심)
python src_new/modeling/run_master.py --template lasso
```

### 📋 설정 파일 사용

```bash
# 기본 설정 파일 사용
python src_new/modeling/run_master.py

# 커스텀 설정 파일 사용
python src_new/modeling/run_master.py --config my_config.json
```

## 🎭 앙상블 모델 (NEW!)

### 개요
개별 모델들을 결합하여 더 강력한 예측 성능을 달성합니다!

- **가중 평균**: 각 모델의 예측 확률을 가중치로 결합
- **자동 가중치**: 검증 성능 기반 최적 가중치 자동 계산
- **수동 가중치**: 사용자 정의 가중치 설정 가능
- **최적 Threshold**: 앙상블 결과에도 최적 임계값 적용

### 설정 방법

```json
{
  "ensemble": {
    "enabled": true,                                    // 앙상블 활성화
    "method": "weighted_average",                       // 앙상블 방법
    "auto_weight": true,                               // 자동 가중치 계산
    "models": ["logistic", "random_forest", "xgboost"], // 포함할 모델들
    "data_types": ["normal", "smote"],                 // 포함할 데이터 타입
    "weights": {                                       // 수동 가중치 (auto_weight=false시)
      "logisticregression_normal": 0.3,
      "randomforest_normal": 0.4,
      "xgboost_normal": 0.3,
      "logisticregression_smote": 0.2,
      "randomforest_smote": 0.3,
      "xgboost_smote": 0.2
    },
    "threshold_optimization": {
      "enabled": true,
      "metric_priority": "f1"
    }
  }
}
```

### 🎯 앙상블 방법

| 방법 | 설명 | 특징 |
|------|------|------|
| **weighted_average** | 가중 평균 | **권장** - 안정적이고 해석 가능 |
| **voting** | 가중 다수결 | 이진 투표 기반 |
| **stacking** | 메타 모델 | 고급 기법 (미래 확장) |

## 🔥 자동 Threshold 최적화 (핵심 기능)

### 개요
기존의 하드코딩된 `threshold: 0.5` 방식을 완전히 개선!

- **문제점**: 모든 모델에 동일한 threshold 적용 → 성능 제한
- **해결책**: 각 모델별로 Validation Set 기반 최적 threshold 자동 탐색
- **범위**: 0.1 ~ 0.85 (0.05 간격으로 16개 포인트 탐색)
- **메트릭**: F1, Precision, Recall, Balanced Accuracy 중 선택

### 설정 방법

```json
{
  "threshold_optimization": {
    "enabled": true,                                    // 활성화 여부
    "metric_priority": "f1",                           // 주 최적화 메트릭
    "alternatives": ["precision", "recall", "balanced_accuracy"]  // 대안 메트릭들
  }
}
```

### 💡 메트릭별 특징

| 메트릭 | 특징 | 권장 상황 |
|--------|------|-----------|
| **f1** | Precision과 Recall의 조화평균 | **일반적 권장** - 균형잡힌 성능 |
| **precision** | 부실 예측의 정확도 | 보수적 예측이 중요한 경우 |
| **recall** | 실제 부실의 탐지율 | 부실 기업을 놓치면 안 되는 경우 |
| **balanced_accuracy** | 클래스 불균형 고려 | 극심한 불균형 데이터 |

## 📊 설정 파일 구조

### 기본 설정 (`master_config.json`)
```json
{
  "run_name": "default_run",
  "random_state": 42,
  "data_path": "data_new/final",
  "output_base_dir": "outputs/master_runs",
  
  "threshold_optimization": {
    "enabled": true,
    "metric_priority": "f1",
    "alternatives": ["precision", "recall", "balanced_accuracy"]
  },
  
  "lasso": {
    "enabled": true,
    "alphas": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "cv_folds": 5,
    "threshold": "median"
  },
  
  "models": {
    "logistic": {
      "enabled": true,
      "n_trials": 50,
      "penalty": ["l1", "l2", "elasticnet"],
      "C_range": [1e-5, 1000],
      "max_iter_range": [100, 2000]
    },
    
    "random_forest": {
      "enabled": true,
      "n_trials": 50,
      "n_estimators_range": [50, 500],
      "max_depth_range": [3, 20],
      "min_samples_split_range": [2, 20],
      "min_samples_leaf_range": [1, 10],
      "max_features_range": [0.1, 1.0]
    },
    
    "xgboost": {
      "enabled": true,
      "n_trials": 50,
      "n_estimators_range": [50, 500],
      "max_depth_range": [3, 12],
      "learning_rate_range": [0.01, 0.3],
      "subsample_range": [0.7, 1.0],
      "colsample_bytree_range": [0.7, 1.0],
      "reg_alpha_range": [0, 5],
      "reg_lambda_range": [0, 5]
    }
  }
}
```

## 📁 출력 구조

```
outputs/master_runs/
└── {run_name}_{timestamp}/
    ├── config.json                          # 사용된 설정
    ├── models/                              # 훈련된 모델들
    │   ├── logisticregression_normal_model.joblib
    │   ├── logisticregression_smote_model.joblib
    │   └── ...
    ├── results/                             # 결과 파일들
    │   ├── all_results.json                # 전체 결과 (threshold 포함)
    │   ├── summary_table.csv               # 요약 테이블
    │   ├── lasso_selection_normal.json     # Lasso 결과
    │   └── lasso_selection_smote.json
    └── visualizations/                      # 시각화
        ├── threshold_optimization_analysis.png  # 🆕 Threshold 분석
        ├── precision_recall_curves.png         # 🆕 PR 곡선
        └── model_performance_comparison.png
```

## 🎯 템플릿 종류

### 1. **Quick Test** (`--template quick`)
```json
{
  "threshold_optimization": {"enabled": true, "metric_priority": "f1"},
  "n_trials": 10,   // 빠른 테스트
  "lasso": {"enabled": false}
}
```

### 2. **Production** (`--template production`)
```json
{
  "threshold_optimization": {"enabled": true, "metric_priority": "f1"},
  "n_trials": 100,  // 완전한 최적화
  "lasso": {"enabled": true}
}
```

### 3. **Lasso Focus** (`--template lasso`)
```json
{
  "threshold_optimization": {"enabled": true, "metric_priority": "precision"},
  "n_trials": 30,
  "lasso": {"enabled": true, "threshold": 0.001}  // 더 정밀한 특성 선택
}
```

## 📈 결과 해석

### 🆕 Threshold 최적화 결과
```json
{
  "threshold_optimization": {
    "LogisticRegression_normal": {
      "optimal_threshold": 0.15,
      "metric_scores": {
        "f1": 0.4567,
        "precision": 0.6123,
        "recall": 0.3654
      }
    }
  }
}
```

### 요약 테이블 (summary_table.csv)
| Model | Data_Type | **Optimal_Threshold** | CV_AUC | Test_AUC | **Test_F1** | Test_Precision | Test_Recall |
|-------|-----------|----------------------|--------|----------|-------------|----------------|-------------|
| LogisticRegression | normal | **0.15** | 0.823 | 0.816 | **0.457** | 0.612 | 0.365 |
| RandomForest | normal | **0.30** | 0.845 | 0.838 | **0.556** | 0.667 | 0.478 |

## 🔧 고급 사용법

### 1. 메트릭별 최적화 비교
```bash
# F1 최적화
python run_master.py --template quick  # F1이 기본값

# Precision 최적화 (보수적 예측)
# config에서 "metric_priority": "precision"으로 변경 후 실행

# Recall 최적화 (부실 기업 놓치지 않기)
# config에서 "metric_priority": "recall"로 변경 후 실행
```

### 2. Lasso 특성 선택과 함께
```bash
python run_master.py --template lasso  # Precision 우선 + Lasso 활성화
```

### 3. 커스텀 설정으로 실험
```json
{
  "threshold_optimization": {
    "enabled": true,
    "metric_priority": "balanced_accuracy",  // 클래스 불균형 고려
    "alternatives": ["f1", "precision", "recall"]
  }
}
```

## 🎨 시각화 기능

### 1. **Threshold 최적화 분석**
- 각 threshold별 메트릭 성능 곡선
- 최적 포인트 표시
- 모델별 비교

### 2. **Precision-Recall 곡선**
- 모델별 PR 곡선
- 최적 threshold 포인트
- AUC 점수 비교

### 3. **성능 비교 차트**
- 모델별 성능 레이더 차트
- Normal vs SMOTE 비교
- 메트릭별 성능 분포

## 💡 실무 활용 팁

### 1. **메트릭 선택 가이드**
- **금융기관**: `precision` 우선 (잘못된 부실 예측 비용 고려)
- **신용평가사**: `f1` 균형 (전반적 성능)
- **규제기관**: `recall` 우선 (부실 기업 놓치지 않기)

### 2. **Threshold 결과 해석**
- **낮은 threshold (0.1-0.3)**: 높은 Recall, 낮은 Precision
- **높은 threshold (0.6-0.8)**: 낮은 Recall, 높은 Precision
- **중간 threshold (0.3-0.5)**: 균형잡힌 성능

### 3. **성능 개선 전략**
1. **데이터 품질**: 결측치 처리, 특성 엔지니어링
2. **클래스 균형**: SMOTE vs Normal 데이터 비교
3. **특성 선택**: Lasso 활용한 차원 축소
4. **앙상블**: 여러 모델의 최적 threshold 조합

## 🚨 주의사항

1. **과적합 방지**: Validation Set 기반 threshold 선택으로 일반화 성능 확보
2. **클래스 불균형**: 극심한 불균형 시 `balanced_accuracy` 고려
3. **도메인 지식**: 금융 도메인 특성 고려하여 메트릭 선택
4. **계산 비용**: Threshold 최적화로 인한 추가 시간 소요

## 🔗 관련 파일

- **데이터**: `data_new/final/X_train_100_*.csv`
- **설정**: `src_new/modeling/config_templates/*.json`
- **결과**: `outputs/master_runs/{run_name}/`
- **시각화**: `outputs/master_runs/{run_name}/visualizations/`

---

## 📋 기존 개별 스크립트들

### 개별 모델 스크립트 (레거시)
- `logistic_regression_100.py`: 로지스틱 회귀 모델
- `RF_100.py`: 랜덤 포레스트 모델  
- `xgboost_100.py`: XGBoost 모델

### 사용법
```bash
# 개별 모델 실행 (권장하지 않음 - 마스터 러너 사용 권장)
python src_new/modeling/logistic_regression_100.py
python src_new/modeling/RF_100.py
python src_new/modeling/xgboost_100.py
```

## 🏆 권장 워크플로우

### 1단계: 빠른 프로토타이핑
```bash
python src_new/modeling/run_master.py --template quick
```

### 2단계: 본격 최적화
```bash
python src_new/modeling/run_master.py --template production
```

### 3단계: 특성 선택 분석
```bash
python src_new/modeling/run_master.py --template lasso
```

### 4단계: 커스텀 튜닝
1. 설정 파일 수정 (`master_config.json`)
2. 재실행 및 결과 비교

## 🎉 마스터 러너의 장점

1. **🎯 자동 최적화**: 각 모델별 최적 threshold 자동 탐색
2. **⚡ 효율성**: 한 번 실행으로 모든 모델 + 데이터 조합
3. **📊 풍부한 분석**: Threshold 곡선, PR 곡선 등 시각화
4. **🔧 유연성**: JSON 설정으로 모든 하이퍼파라미터 제어
5. **📁 체계성**: 실행별 독립적 결과 저장
6. **🔄 재현성**: 설정 파일 저장으로 완전한 재현 가능

## 🔧 **SMOTE Data Leakage 문제 해결**

### ❌ 기존 문제점
```python
# 잘못된 방법: SMOTE 먼저 적용 → CV 수행
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)  # ❌ Data Leakage!
```

### ✅ 올바른 해결책
```python
# 올바른 방법: CV 내부에서 SMOTE 적용
def proper_cv_with_smote(model, X, y, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        # 각 fold마다 별도로 분할
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 훈련 fold에만 SMOTE 적용 (Data Leakage 방지)
        smote = BorderlineSMOTE(sampling_strategy=0.1, random_state=42)
        X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
        
        # 모델 훈련 및 평가
        model.fit(X_fold_train_smote, y_fold_train_smote)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]  # 원본 데이터로 평가
        score = roc_auc_score(y_fold_val, y_pred_proba)
        scores.append(score)
    
    return np.array(scores)
```

### 🎯 핵심 개선사항
1. **각 CV fold마다 SMOTE 별도 적용**
2. **원본 데이터로 검증 수행**
3. **합성 데이터 간 오염 방지**
4. **정확한 일반화 성능 평가**

## 📊 결과 파일

### 모델 저장
- `outputs/master_runs/{run_name}/models/` - 훈련된 모델 파일들
- `.joblib` 형식으로 저장

### 결과 분석
- `outputs/master_runs/{run_name}/results/` - 성능 메트릭 및 분석 결과
- `all_results.json` - 전체 결과 종합
- `summary_table.csv` - 요약 테이블

### 시각화
- `outputs/master_runs/{run_name}/visualizations/` - 그래프 및 차트
- ROC 곡선, Precision-Recall 곡선
- 특성 중요도 비교
- Normal vs SMOTE 성능 비교

## 🔍 성능 메트릭

### 분류 메트릭
- **AUC-ROC**: 전체적인 분류 성능
- **Precision**: 부실 예측의 정확도
- **Recall**: 실제 부실기업 탐지율
- **F1-Score**: Precision과 Recall의 조화평균
- **Balanced Accuracy**: 클래스 불균형 고려 정확도

### 검증 방식
- **5-Fold Stratified Cross Validation**
- **Hold-out Test Set** 최종 평가
- **Validation Set** 기반 Threshold 최적화

## 💡 주요 특징

### 1. **클래스 불균형 처리**
- BorderlineSMOTE로 부실기업 데이터 증강
- 1:10 비율로 균형 조정
- 원본 데이터 보존

### 2. **과적합 방지**
- 3단계 검증 (CV → Validation → Test)
- Early Stopping 및 정규화
- 특성 선택으로 차원 축소

### 3. **재현 가능성**
- 모든 랜덤 시드 고정
- 설정 파일 기반 실험 관리
- 버전 관리 및 결과 추적

## 🚨 주의사항

1. **데이터 누수 방지**: SMOTE는 반드시 CV 내부에서 적용
2. **시계열 특성 고려**: 금융 데이터의 시간적 의존성 주의
3. **메모리 사용량**: 대용량 데이터 처리 시 메모리 모니터링 필요
4. **하이퍼파라미터 범위**: 과도한 탐색 범위는 최적화 시간 증가

## 📈 성능 향상 팁

1. **특성 엔지니어링**: 도메인 지식 기반 특성 생성
2. **앙상블 방법**: 여러 모델 조합으로 성능 향상
3. **임계값 조정**: 비즈니스 목적에 맞는 Precision/Recall 균형
4. **데이터 품질**: 이상치 처리 및 결측값 보완
