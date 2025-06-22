# 📊 Outputs - 분석 결과 및 모델 저장소

이 디렉토리는 한국 기업 부실예측 모델링 프로젝트의 **모든 분석 결과, 학습된 모델, 시각화 자료**를 포함합니다.

## 📁 디렉토리 구조

```
outputs/
├── 📁 models/          # 학습된 머신러닝 모델들
├── 📁 reports/         # 분석 보고서 및 성능 결과
├── 📁 visualizations/  # 차트 및 그래프 시각화
└── 📁 analysis/        # 다중공선성 등 고급 분석 결과
```

---

## 🤖 Models - 학습된 모델

### 🏆 최종 추천 모델
- **`final_smote_randomforest_model.joblib`** (🥇 **최고 성능**)
  - F1-Score: 0.608, AUC: 0.986
  - Random Forest + SMOTE 조합
  - 실무 적용 권장 모델

- **`final_scaler.joblib`**
  - StandardScaler 전처리 객체
  - 새로운 데이터 예측 시 필수

### 📈 개별 모델별 결과

#### Logistic Regression
- **`logistic_regression_100_normal_*.joblib`**: 원본 데이터 학습
- **`logistic_regression_100_smote_*.joblib`**: SMOTE 적용 학습
- **`logistic_regression_results.json`**: 성능 지표 요약

#### Random Forest  
- **`random_forest_100_normal_*.joblib`**: 원본 데이터 학습
- **`random_forest_100_smote_*.joblib`**: SMOTE 적용 학습 (🥇)
- **`random_forest_results.json`**: 성능 지표 요약

#### XGBoost
- **`xgboost_100_normal_*.joblib`**: 원본 데이터 학습
- **`xgboost_100_smote_*.joblib`**: SMOTE 적용 학습
- **`xgboost_results.json`**: 성능 지표 요약

### 🔧 전처리 객체
- **`scaler_100_normal.joblib`**: Normal 데이터용 스케일러
- **`scaler_100_smote.joblib`**: SMOTE 데이터용 스케일러

### 📊 특성 선택 모델 (실험)
- **`k2_feature_*_model.joblib`**: K2 Score 기반 특성 선택 실험
- **`k2_feature_scaler_*.joblib`**: 해당 실험용 스케일러
- ⚠️ **주의**: K2 Score 관련 모델은 다중공선성 문제로 사용 비권장

---

## 📋 Reports - 분석 보고서

### 🏆 종합 성능 보고서
- **`100_complete_summary_report.json`**: 전체 모델 성능 요약
  - AUC, F1-Score, Precision, Recall 등 주요 지표
  - 모델별 비교 분석
  - 최고 성능 모델 식별

- **`100_complete_detailed_results.csv`**: 상세 성능 데이터
  - 모든 모델의 세부 성능 지표
  - 교차검증 결과 포함
  - 통계적 유의성 검증

### 📊 개별 모델 보고서
- **`logistic_regression_100_*_*.json`**: 로지스틱 회귀 상세 결과
- **`random_forest_100_*_*.json`**: 랜덤 포레스트 상세 결과
- **`xgboost_100_*_*.json`**: XGBoost 상세 결과

### 🔍 특별 분석 보고서
- **`k2_impact_analysis.json`**: K2 Score 영향 분석
  - K2 Score 추가 전후 성능 비교
  - 다중공선성 영향 분석
  - 정보 누출(Information Leakage) 검증

### 📈 스케일링 분석
- **`scaling_analysis.xlsx`**: 스케일링 필요성 분석
- **`scaling_analysis_detailed.xlsx`**: 상세 스케일링 연구

---

## 📊 Visualizations - 시각화 결과

### 🏆 종합 비교 차트
- **`100_complete_model_comparison.png`**: 모델별 성능 비교
- **`100_complete_auc_f1_scatter.png`**: AUC vs F1 산점도
- **`100_complete_radar_charts.png`**: 레이더 차트 (다각도 성능)

### 📈 성능 분석 차트
- **`model_metrics_comparison.png`**: 세부 지표 비교
- **`model_roc_comparison.png`**: ROC 곡선 비교
- **`feature_importance_comparison.png`**: 특성 중요도 비교

### 🎯 개별 모델 시각화
- **`logistic_regression_results.png`**: 로지스틱 회귀 성능 차트
- **`random_forest_results.png`**: 랜덤 포레스트 성능 차트

### 📊 데이터 분석 차트
- **`ratio_distributions.png`**: 재무비율 분포 분석
- **`ratio_boxplots_normalized.png`**: 정규화된 박스플롯
- **`scaling_comprehensive_analysis.png`**: 종합 스케일링 분석
- **`scaling_need_indicators.png`**: 스케일링 필요성 지표

### 🔗 상관관계 분석
- **`score_correlations.png`**: 스코어 간 상관관계
- **`k2_impact_analysis.png`**: K2 Score 영향 시각화

---

## 🔍 Analysis - 고급 분석 결과

### 🎯 다중공선성 분석 (⭐ **최신**)
- **`comprehensive_multicollinearity_analysis_*.json`**: 포괄적 다중공선성 분석
  - VIF 계산 결과
  - 상관관계 매트릭스
  - PCA 분석 결과
  - 조건지수 계산

- **`final_vif_results_*.csv`**: 최종 VIF 결과 (K2 Score 제거 후)
- **`final_correlation_matrix_*.csv`**: 상관계수 행렬

### 📊 다중공선성 시각화
- **`correlation_heatmap_*.png`**: 상관계수 히트맵
- **`vif_analysis_cleaned_*.png`**: 정리된 VIF 분석 차트
- **`variable_removal_process_*.png`**: 변수 제거 과정
- **`pca_variance_explained_*.png`**: PCA 설명 분산

### ⚠️ 주요 발견사항
1. **K2_Score_Original 완전한 다중공선성**: VIF = ∞
2. **해결책**: K2 Score 제거 → 모든 VIF < 5 달성
3. **높은 상관관계**: WC_TA ↔ CLCA (-0.823)
4. **PCA 결과**: 13개 성분으로 95% 분산 설명 가능

---

## 🚀 사용 방법

### 📖 모델 로드 및 예측
```python
import joblib
import pandas as pd

# 최고 성능 모델 로드
model = joblib.load('outputs/models/final_smote_randomforest_model.joblib')
scaler = joblib.load('outputs/models/final_scaler.joblib')

# 새로운 데이터 전처리 및 예측
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)[:, 1]
```

### 📊 성능 결과 확인
```python
import json

# 종합 보고서 로드
with open('outputs/reports/100_complete_summary_report.json', 'r') as f:
    summary = json.load(f)

# 최고 성능 모델 확인
best_model = summary['best_model']
print(f"최고 성능 모델: {best_model['name']}")
print(f"F1-Score: {best_model['f1_score']:.3f}")
print(f"AUC: {best_model['auc']:.3f}")
```

### 🔍 다중공선성 결과 확인
```python
# 다중공선성 분석 결과 로드
with open('outputs/analysis/comprehensive_multicollinearity_analysis_*.json', 'r') as f:
    multicollinearity = json.load(f)

# VIF 결과 확인
vif_results = pd.read_csv('outputs/analysis/final_vif_results_*.csv')
print("VIF > 5인 변수:")
print(vif_results[vif_results['VIF'] > 5])
```

---

## 📈 주요 성과 요약

### 🏆 최고 성능 달성
| 지표 | 값 | 모델 |
|------|----|----- |
| **F1-Score** | **0.608** | Random Forest (SMOTE) |
| **AUC** | **0.986** | Random Forest (SMOTE) |
| **Precision** | **0.583** | Random Forest (SMOTE) |
| **Recall** | **0.635** | Random Forest (SMOTE) |

### 🔧 기술적 성과
- ✅ **다중공선성 해결**: VIF ∞ → 평균 2.34
- ✅ **완전한 데이터**: 100% Complete (16,197개)
- ✅ **불균형 해결**: SMOTE 적용으로 성능 향상
- ✅ **재현 가능성**: 모든 결과 저장 및 문서화

---

## 📞 파일 관련 문의

### 🔍 파일이 없는 경우
1. **모델 재학습**: `src_new/modeling/` 스크립트 실행
2. **분석 재실행**: `src_new/analysis/` 스크립트 실행
3. **시각화 재생성**: 각 스크립트의 시각화 함수 호출

### 🗂️ 파일 명명 규칙
- **모델**: `{algorithm}_{데이터타입}_{timestamp}.joblib`
- **보고서**: `{analysis_type}_{timestamp}.json`
- **시각화**: `{chart_type}_{timestamp}.png`

### ⚡ 빠른 복구
```bash
# 전체 파이프라인 재실행
python src_new/modeling/model_comparison.py
python src_new/analysis/multicollinearity_analysis_improved.py
```

---

**📊 모든 결과는 완전히 재현 가능하며, 필요시 언제든 재생성할 수 있습니다.**

**🚀 실무 적용을 위해 `final_smote_randomforest_model.joblib` 모델 사용을 권장합니다.**

---

*마지막 업데이트: 2025년 6월 22일 - 다중공선성 분석 결과 추가* 