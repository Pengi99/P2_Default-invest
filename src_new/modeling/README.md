# 📊 모델링 (Modeling)

한국 기업 부실예측을 위한 머신러닝 모델링 파이프라인입니다.

## 🎯 주요 기능

### 1. **🆕 마스터 모델 러너** (통합 파이프라인)
- **자동화된 모델 실행**: LogisticRegression, RandomForest, XGBoost 일괄 실행
- **🔥 자동 Threshold 최적화**: 각 모델별 최적 임계값 자동 탐색
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
