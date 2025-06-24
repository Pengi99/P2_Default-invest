# 🤖 Modeling Module

**한국 기업 부실예측을 위한 머신러닝 모델링 시스템**

## 🎯 **개요**

이 모듈은 **두 가지 데이터 트랙**을 활용한 한국 기업 부실예측 모델링을 위한 통합 시스템입니다. 자동 임계값 최적화, 앙상블 모델링, Data Leakage 방지 등 고급 기능을 제공합니다.

## 📊 **지원 데이터 트랙**

### 🔥 **확장 트랙** (FS_ratio_flow_labeled.csv)
- **관측치**: 22,780개 × 36개 변수
- **부실기업**: 132개 (0.58%)
- **특징**: YoY 성장률, 변화량 지표, 발생액 등 고급 변수
- **용도**: 고급 특성공학, 복합 모델링

### ✅ **완전 트랙** (FS_100_complete.csv)  
- **관측치**: 16,197개 × 22개 변수
- **부실기업**: 104개 (0.64%)
- **특징**: 결측치 0%, 다중공선성 해결 완료
- **용도**: 안정적 운영, 기본 모델링

## 🏗️ **시스템 구조**

```
📦 src/modeling/
├── 🚀 master_model_runner.py          # 통합 모델링 엔진
├── 🎮 run_master.py                   # 마스터 러너 실행기
├── ⚙️ master_config.json             # 중앙 설정 파일
├── 🎭 ensemble_model.py               # 앙상블 모델 구현
├── 📁 config_templates/               # 설정 템플릿 모음
│   ├── 🏭 production_config.json      # 운영환경 설정
│   ├── ⚡ quick_test_config.json      # 빠른 테스트 설정
│   ├── 🎯 lasso_focus_config.json     # Lasso 특성선택 설정
│   └── 🔧 custom_config.json          # 사용자 정의 설정
├── 📊 개별 모델 파일들:
│   ├── 📈 logistic_regression_100.py  # 로지스틱 회귀
│   ├── 🌳 random_forest_100.py        # 랜덤 포레스트  
│   ├── 🚀 xgboost_100.py              # XGBoost
│   ├── 📊 model_comparison.py         # 모델 비교 분석
│   └── 🔍 threshold_optimization.py   # 임계값 최적화
└── 📄 README.md                       # 현재 파일
```

## 🚀 **핵심 기능**

### 1. 🎮 **마스터 러너 시스템**

통합 모델링 파이프라인으로 모든 모델을 자동화하여 실행합니다.

```python
# 기본 실행 (권장)
python run_master.py

# 템플릿 기반 실행
python run_master.py --template production     # 운영환경용
python run_master.py --template quick_test     # 빠른 테스트
python run_master.py --template lasso_focus    # Lasso 특성선택

# 사용자 정의 설정
python run_master.py --config custom_config.json
```

**주요 기능:**
- 🎯 **자동 임계값 최적화**: 각 모델별 F1-Score 기준 최적 threshold 탐색
- 🎭 **앙상블 모델링**: 9개 모델 조합으로 21.3% 성능 향상
- 🛡️ **Data Leakage 방지**: CV 내부 동적 SMOTE 적용
- 📊 **포괄적 평가**: 다양한 성능 지표로 모델 비교

### 2. 🎭 **앙상블 모델링**

**최고 성능 달성**: F1-Score 0.4096, AUC 0.9808

```python
from ensemble_model import EnsembleModel

# 앙상블 모델 초기화
ensemble = EnsembleModel(
    models=['logistic', 'randomforest', 'xgboost'],
    data_types=['normal', 'smote', 'combined'],
    weighting_strategy='equal'  # 균등 가중치
)

# 학습 및 예측
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

**앙상블 구성:**
- **모델**: LogisticRegression, RandomForest, XGBoost
- **데이터**: Normal, SMOTE, Combined (각 3개)
- **총 9개 모델** 균등 가중치 (각 11.11%)

### 3. ⚡ **자동 임계값 최적화**

각 모델별로 최적의 임계값을 자동으로 탐색합니다.

```python
from threshold_optimization import ThresholdOptimizer

optimizer = ThresholdOptimizer(
    metric='f1',           # 최적화 기준: f1, precision, recall
    search_range=(0.05, 0.95),  # 탐색 범위
    step_size=0.05         # 탐색 간격
)

optimal_threshold = optimizer.optimize(y_true, y_pred_proba)
```

**성과:**
- **평균 15% F1-Score 향상** (기본 0.5 대비)
- **모델별 최적 임계값**: 0.05~0.85 범위에서 자동 탐색
- **교차검증 기반**: 안정적이고 일반화된 성능

### 4. 🛡️ **Data Leakage 방지**

```python
# 동적 SMOTE 적용 (CV 내부에서만)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

def cv_with_smote(model, X, y, cv=5):
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # SMOTE는 훈련 데이터에만 적용
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        model.fit(X_train_smote, y_train_smote)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return scores
```

## 📊 **모델별 성능 비교**

### 🏆 **최고 성능 결과**

| 모델 | 데이터 트랙 | 최적 Threshold | AUC | F1-Score | Precision | Recall |
|------|------------|----------------|-----|----------|-----------|--------|
| **🎭 Ensemble** | **Mixed** | **0.10** | **0.9808** | **0.4096** | **0.2982** | **0.6538** |
| **🚀 XGBoost** | Normal | 0.10 | 0.9800 | 0.3380 | 0.2857 | 0.4103 |
| **🚀 XGBoost** | SMOTE | 0.15 | 0.9733 | 0.3121 | 0.2414 | 0.4359 |
| **🌳 RandomForest** | Normal | 0.15 | 0.9793 | 0.2381 | 0.2632 | 0.2179 |
| **🌳 RandomForest** | SMOTE | 0.20 | 0.9734 | 0.2222 | 0.2000 | 0.2500 |
| **📈 LogisticRegression** | Normal | 0.15 | 0.9508 | 0.2182 | 0.1875 | 0.2564 |
| **📈 LogisticRegression** | SMOTE | 0.20 | 0.9523 | 0.2105 | 0.1739 | 0.2564 |

### 📈 **성능 향상 분석**
- **앙상블 vs 최고 개별**: +21.3% F1-Score 향상
- **임계값 최적화**: 평균 +15% F1-Score 향상
- **SMOTE 효과**: Recall 향상, Precision 일부 하락
- **AUC 성능**: 모든 모델에서 0.95+ 달성

## ⚙️ **설정 관리**

### 📄 **master_config.json 구조**

```json
{
    "data_config": {
        "base_path": "../../data/final/",
        "train_file": "X_train_100_normal.csv",
        "target_file": "y_train_100_normal.csv",
        "test_file": "X_test_100_normal.csv",
        "scaler_type": "standard"
    },
    "model_config": {
        "models_to_run": ["logistic", "randomforest", "xgboost"],
        "use_smote": true,
        "smote_strategy": "minority",
        "cross_validation_folds": 5
    },
    "threshold_optimization": {
        "enabled": true,
        "metric": "f1",
        "search_range": [0.05, 0.95],
        "step_size": 0.05
    },
    "ensemble_config": {
        "enabled": true,
        "weighting_strategy": "equal",
        "models_to_ensemble": "all"
    },
    "output_config": {
        "save_models": true,
        "save_results": true,
        "create_visualizations": true,
        "output_dir": "../../outputs/master_runs/"
    }
}
```

### 🎯 **템플릿 설정**

#### 🏭 **production_config.json**
```json
{
    "model_config": {
        "cross_validation_folds": 10,
        "hyperparameter_tuning": true,
        "n_iter_search": 100
    },
    "threshold_optimization": {
        "step_size": 0.01,
        "search_range": [0.01, 0.99]
    }
}
```

#### ⚡ **quick_test_config.json**
```json
{
    "model_config": {
        "models_to_run": ["logistic", "randomforest"],
        "cross_validation_folds": 3,
        "hyperparameter_tuning": false
    },
    "threshold_optimization": {
        "step_size": 0.1,
        "search_range": [0.1, 0.9]
    }
}
```

## 🔍 **개별 모델 상세**

### 📈 **Logistic Regression**
```python
# logistic_regression_100.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 튜닝
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga']
}

model = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
```

**특징:**
- ✅ 해석 가능성 높음
- ✅ 빠른 학습 속도
- ✅ 선형 관계 모델링에 효과적
- ❌ 비선형 패턴 포착 한계

### 🌳 **Random Forest**
```python
# random_forest_100.py
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestClassifier(random_state=42, n_jobs=-1)
```

**특징:**
- ✅ 특성 중요도 제공
- ✅ 과적합 방지
- ✅ 결측치 처리 우수
- ❌ 메모리 사용량 많음

### 🚀 **XGBoost**
```python
# xgboost_100.py
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
```

**특징:**
- ✅ 최고 성능 (F1: 0.3380)
- ✅ 그래디언트 부스팅
- ✅ 조기 종료 지원
- ❌ 하이퍼파라미터 튜닝 복잡

## 📊 **결과 분석 및 시각화**

### 📈 **자동 생성 차트**

```python
# 실행 후 자동 생성되는 시각화
outputs/master_runs/default_run_YYYYMMDD_HHMMSS/visualizations/
├── 📊 ensemble_analysis.png           # 앙상블 성능 분석
├── ⚖️ ensemble_weights.png            # 앙상블 가중치 분포  
├── 🎯 threshold_optimization_analysis.png # 임계값 최적화 분석
├── 📈 cv_vs_test_comparison.png       # CV vs Test 성능 비교
├── 🔍 feature_importance_comparison.png # 특성 중요도 비교
├── 📊 normal_vs_smote_detailed.png    # Normal vs SMOTE 비교
└── 📋 performance_comparison.png      # 전체 성능 비교
```

### 📄 **결과 파일**

```python
outputs/master_runs/default_run_YYYYMMDD_HHMMSS/results/
├── 📊 all_results.json               # 전체 결과 JSON
├── 📋 summary_table.csv              # 성능 요약 테이블
├── 🎯 lasso_selection_normal.json    # Lasso 특성 선택 결과
└── 📈 threshold_analysis.json        # 임계값 분석 결과
```

### 🤖 **저장된 모델**

```python
outputs/master_runs/default_run_YYYYMMDD_HHMMSS/models/
├── 🎭 ensemble_model_model.joblib           # 앙상블 모델
├── 📈 logisticregression_normal_model.joblib # 로지스틱 회귀 (Normal)
├── 📈 logisticregression_smote_model.joblib  # 로지스틱 회귀 (SMOTE)
├── 🌳 randomforest_normal_model.joblib      # 랜덤 포레스트 (Normal)
├── 🌳 randomforest_smote_model.joblib       # 랜덤 포레스트 (SMOTE)
├── 🚀 xgboost_normal_model.joblib           # XGBoost (Normal)
└── 🚀 xgboost_smote_model.joblib            # XGBoost (SMOTE)
```

## 🚀 **실행 가이드**

### 1. **기본 실행** (권장)

```bash
cd src/modeling
python run_master.py
```

**실행 내용:**
- 3개 알고리즘 × 3개 데이터 타입 = 9개 모델 학습
- 자동 임계값 최적화
- 앙상블 모델 생성
- 포괄적 성능 평가 및 시각화

### 2. **템플릿 기반 실행**

```bash
# 운영환경용 (완전 최적화)
python run_master.py --template production

# 빠른 테스트용
python run_master.py --template quick_test

# Lasso 특성선택 포함
python run_master.py --template lasso_focus
```

### 3. **개별 모델 실행**

```bash
# 개별 모델 실행 (비교용)
python logistic_regression_100.py
python random_forest_100.py
python xgboost_100.py
```

### 4. **사용자 정의 실행**

```bash
# 사용자 정의 설정 파일 사용
python run_master.py --config my_custom_config.json

# 특정 모델만 실행
python run_master.py --models logistic,xgboost

# SMOTE 비활성화
python run_master.py --no-smote
```

## 🔧 **고급 기능**

### 1. **하이퍼파라미터 튜닝**

```python
# GridSearchCV 기반 자동 튜닝
hyperparameter_grids = {
    'logistic': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2']
    },
    'randomforest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None]
    },
    'xgboost': {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}
```

### 2. **특성 선택 (Lasso)**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Lasso 기반 특성 선택
lasso = LassoCV(cv=5, random_state=42)
selector = SelectFromModel(lasso)
X_selected = selector.fit_transform(X_train, y_train)

print(f"선택된 특성 수: {X_selected.shape[1]}")
print(f"선택된 특성: {X.columns[selector.get_support()].tolist()}")
```

### 3. **교차검증 전략**

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

# 시계열 고려 교차검증
tscv = TimeSeriesSplit(n_splits=5)

# 계층화 교차검증 (불균형 데이터)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## 📊 **성능 모니터링**

### 🎯 **핵심 지표 추적**

```python
# 자동 계산되는 성능 지표
metrics = {
    'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'probability': ['log_loss', 'brier_score'],
    'ranking': ['average_precision', 'roc_auc'],
    'threshold': ['optimal_threshold', 'threshold_range']
}
```

### 📈 **실시간 모니터링**

```python
# 학습 과정 모니터링
import wandb  # Weights & Biases (선택사항)

wandb.init(project="default-prediction")
wandb.log({
    "train_f1": train_f1,
    "val_f1": val_f1,
    "test_f1": test_f1,
    "optimal_threshold": optimal_threshold
})
```

## 🔍 **문제 해결 가이드**

### ❗ **자주 발생하는 오류**

#### 1. **메모리 부족 오류**
```bash
# 해결방법: 배치 크기 줄이기
python run_master.py --batch-size 1000

# 또는 모델 수 줄이기
python run_master.py --models logistic,xgboost
```

#### 2. **SMOTE 오류**
```bash
# 해결방법: SMOTE 비활성화
python run_master.py --no-smote

# 또는 SMOTE 전략 변경
python run_master.py --smote-strategy auto
```

#### 3. **수렴 경고**
```python
# LogisticRegression 수렴 문제
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 또는 max_iter 증가
LogisticRegression(max_iter=2000)
```

### 🔧 **성능 최적화 팁**

1. **병렬 처리 활용**
```python
# n_jobs=-1로 모든 CPU 코어 사용
RandomForestClassifier(n_jobs=-1)
GridSearchCV(n_jobs=-1)
```

2. **조기 종료 활용**
```python
# XGBoost 조기 종료
xgb.XGBClassifier(early_stopping_rounds=10)
```

3. **메모리 효율성**
```python
# 대용량 데이터 처리
pd.read_csv('data.csv', chunksize=10000)
```

## 📚 **관련 문서**

- **📄 [프로젝트 개요](../../README.md)**: 전체 프로젝트 설명
- **📊 [데이터 가이드](../../data/final/README.md)**: 데이터셋 상세 정보
- **📈 [시각화 가이드](../../outputs/visualizations/README.md)**: 분석 차트 해석
- **🎨 [대시보드 가이드](../../dashboard/README.md)**: 대화형 도구 사용법

## 🏆 **모델링 성과 요약**

✅ **앙상블 모델**: F1-Score 0.4096 (업계 최고 수준)  
✅ **자동 최적화**: 임계값 자동 탐색으로 15% 성능 향상  
✅ **Data Leakage 방지**: CV 내부 동적 SMOTE로 신뢰성 확보  
✅ **재현 가능성**: 설정 기반 관리로 실험 재현 가능  
✅ **운영 준비**: 자동화된 파이프라인으로 배포 준비 완료  

**🎯 한국 기업 부실예측을 위한 최고 성능의 ML 시스템!**

---

*모델 사용 시 성능 지표를 지속적으로 모니터링하시기 바랍니다.*  
*운영 환경 배포 전 충분한 검증을 권장합니다.*
