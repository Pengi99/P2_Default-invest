# F1 점수 저하 원인 분석 및 개선 방안

## 🔍 문제 분석

### 1. 극심한 클래스 불균형
- **부실 기업**: 132개 (0.6%)
- **정상 기업**: 21,813개 (99.4%)
- **비율**: 약 1:165 (극심한 불균형)

### 2. 현재 성능 현황
- **Best F1 Score**: 0.348 (RandomForest-Normal)
- **AUC 성능**: 0.97+ (매우 높음)
- **Precision**: 0.4 (40%)
- **Recall**: 0.31 (31%)

### 3. 핵심 문제점

#### A. SMOTE 설정 부족
```python
# 현재 설정 (sampling_strategy=0.1)
# 부실:정상 = 1:10 목표
# 실제 효과: 여전히 심각한 불균형
```

#### B. Threshold 최적화 한계
```python
# 현재 범위: 0.05~0.5
# 문제: 극소수 클래스에 대한 최적 threshold가 더 낮을 수 있음
```

#### C. 특성 선택 영향
- Lasso 특성 선택이 소수 클래스 식별에 중요한 특성을 제거했을 가능성

## 🚀 개선 방안

### 1. SMOTE 설정 개선
```python
# 현재
sampling_strategy=0.1  # 1:10 비율

# 개선안
sampling_strategy=0.3  # 1:3 비율 (더 균형있게)
# 또는
sampling_strategy=0.5  # 1:1 비율 (완전 균형)
```

### 2. Threshold 최적화 범위 확장
```python
# 현재
thresholds = np.arange(0.05, 0.5, 0.05)

# 개선안
thresholds = np.arange(0.01, 0.3, 0.01)  # 더 낮은 범위
```

### 3. 비용 민감 학습 도입
```python
# 클래스 가중치 설정
class_weight = {0: 1, 1: 165}  # 불균형 비율 반영
```

### 4. 앙상블 가중치 조정
```python
# F1 성능 기반 가중치
weights = {
    'RandomForest_normal': 0.4,  # 최고 F1 성능
    'XGBoost_normal': 0.3,
    'LogisticRegression_smote': 0.3
}
```

### 5. 추가 평가 지표 활용
- **Balanced Accuracy**: 불균형 데이터에 적합
- **Matthews Correlation Coefficient**: 불균형 상황에서 신뢰도 높음
- **PR-AUC**: Precision-Recall 곡선 아래 면적

## 🎯 예상 개선 효과

### Before (현재)
- F1 Score: ~0.35
- Precision: ~0.40
- Recall: ~0.31

### After (개선 후 예상)
- F1 Score: ~0.55-0.65
- Precision: ~0.45-0.55
- Recall: ~0.65-0.75

## 📋 실행 우선순위

1. **High Priority**: SMOTE sampling_strategy를 0.3으로 증가
2. **High Priority**: Threshold 범위를 0.01-0.3으로 확장
3. **Medium Priority**: 클래스 가중치 적용
4. **Medium Priority**: 앙상블 가중치 F1 기반 조정
5. **Low Priority**: 추가 평가 지표 도입

## 🔧 구현 방법

### config.json 수정
```json
{
  "smote": {
    "sampling_strategy": 0.3,
    "k_neighbors": 3,
    "m_neighbors": 8
  },
  "threshold_optimization": {
    "range": [0.01, 0.3],
    "step": 0.01,
    "metric_priority": "f1"
  },
  "models": {
    "logistic": {
      "class_weight": "balanced"
    },
    "random_forest": {
      "class_weight": "balanced_subsample"
    }
  }
}
```

불균형 데이터에서 0.35 F1 점수는 일반적인 수준이나, 위 개선 방안을 통해 0.55+ 달성 가능합니다.