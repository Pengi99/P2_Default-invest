# modeling

머신러닝 모델 구현 스크립트들

## 📄 스크립트별 상세 기능

### 🔧 logistic_regression.py
**로지스틱 회귀 모델링 템플릿**

**주요 기능:**
- 이진 분류 문제를 위한 로지스틱 회귀 구현
- 선형 결정 경계를 통한 부실예측
- 해석 가능한 계수(coefficient) 제공

**모델 특징:**
- **알고리즘**: Logistic Regression
- **장점**: 해석 가능성, 빠른 훈련, 확률 출력
- **단점**: 선형 관계 가정, 복잡한 패턴 학습 제한

**하이퍼파라미터:**
- **penalty**: 규제 종류 ('l1', 'l2', 'elasticnet', None)
- **C**: 규제 강도 (작을수록 규제 강함)
- **solver**: 최적화 알고리즘 ('lbfgs', 'saga', 'liblinear')
- **max_iter**: 최대 반복 횟수

**출력 결과:**
- 정확도 (Accuracy)
- ROC AUC 점수
- 분류 보고서 (Precision, Recall, F1-score)
- 특성별 계수 중요도

**사용법:**
```bash
# 데이터 경로 수정 후 실행
python src_new/modeling/logistic_regression.py
```

---

### 🔧 RF.py
**랜덤 포레스트 모델링 템플릿**

**주요 기능:**
- 앙상블 기반 결정 트리 모델
- 특성 중요도 자동 계산
- 오버피팅 방지 및 강건성 향상

**모델 특징:**
- **알고리즘**: Random Forest Classifier
- **장점**: 높은 정확도, 특성 중요도, 오버피팅 방지
- **단점**: 해석성 제한, 메모리 사용량 높음

**하이퍼파라미터:**
- **n_estimators**: 트리 개수 (100~1000)
- **max_depth**: 트리 최대 깊이
- **min_samples_split**: 노드 분할 최소 샘플 수
- **min_samples_leaf**: 리프 노드 최소 샘플 수
- **max_features**: 분할 시 고려할 특성 수

**출력 결과:**
- 모델 성능 지표
- 특성 중요도 순위
- Out-of-Bag (OOB) 점수
- 교차 검증 결과

**사용법:**
```bash
python src_new/modeling/RF.py
```

---

### 🔧 xgboost.py
**XGBoost 모델링 템플릿**

**주요 기능:**
- 그래디언트 부스팅 기반 고성능 모델
- 불균형 데이터 처리 최적화
- 조기 종료 및 정규화 기능

**모델 특징:**
- **알고리즘**: eXtreme Gradient Boosting
- **장점**: 최고 수준 성능, 불균형 데이터 처리, 빠른 훈련
- **단점**: 하이퍼파라미터 튜닝 복잡

**하이퍼파라미터:**
- **n_estimators**: 부스팅 라운드 수
- **learning_rate**: 학습률 (0.01~0.3)
- **max_depth**: 트리 최대 깊이
- **subsample**: 샘플링 비율
- **colsample_bytree**: 특성 샘플링 비율
- **scale_pos_weight**: 클래스 불균형 가중치

**출력 결과:**
- 성능 지표 (AUC, Precision, Recall)
- 특성 중요도 (gain, cover, frequency)
- 학습 곡선 시각화
- 최적 반복 횟수

**사용법:**
```bash
python src_new/modeling/xgboost.py
```

---

## 🎯 모델 선택 가이드

### 📊 부실예측 모델링 특성
- **클래스 불균형**: 부실 기업 < 1%
- **특성 수**: 17개 재무비율
- **시계열 데이터**: 2012-2023년
- **해석 필요성**: 금융 규제 요구사항

### 🏆 모델별 추천 용도

**1. Logistic Regression**
- ✅ **추천**: 베이스라인 모델, 해석 필요시
- 📈 **장점**: 빠른 훈련, 계수 해석 가능
- ⚠️ **주의**: 복잡한 패턴 학습 제한

**2. Random Forest**
- ✅ **추천**: 안정적인 성능, 특성 중요도 분석
- 📈 **장점**: 오버피팅 방지, 강건성
- ⚠️ **주의**: 메모리 사용량, 해석성 제한

**3. XGBoost**
- ✅ **추천**: 최고 성능 추구, 불균형 데이터
- 📈 **장점**: 최고 수준 성능, 불균형 처리
- ⚠️ **주의**: 하이퍼파라미터 튜닝 복잡

## 🔄 모델링 워크플로우

1. **데이터 준비**: `feature_engineering/` 결과 활용
2. **베이스라인**: Logistic Regression 실행
3. **성능 향상**: Random Forest 실행
4. **최적화**: XGBoost 하이퍼파라미터 튜닝
5. **앙상블**: 여러 모델 조합 고려

## 📊 평가 지표

- **AUC-ROC**: 주요 평가 지표 (불균형 데이터)
- **Precision**: 부실 예측 정확도 (False Positive 최소화)
- **Recall**: 부실 탐지율 (False Negative 최소화)
- **F1-Score**: Precision과 Recall의 조화평균

## 🎯 다음 단계
모델 성능을 `analysis/` 폴더에서 비교 분석
