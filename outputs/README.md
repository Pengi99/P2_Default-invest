# 📊 Outputs - 분석 결과 및 모델 저장소

이 디렉토리는 **한국 기업 부실예측 모델링 프로젝트**의 모든 분석 결과, 학습된 모델, 시각화 자료를 포함합니다.

## 📁 디렉토리 구조

```
outputs/
├── 📁 master_runs/         # Master Model Runner 실행 결과들
│   ├── default_run_*       # 기본 설정 실행 결과
│   └── quick_test_*        # 빠른 테스트 실행 결과
├── 📁 reports/             # CSV 형태 분석 보고서
├── 📁 visualizations/      # 체계적 시각화 자료
│   ├── distributions/      # 개별 변수 분포 차트
│   ├── boxplots/          # 개별 변수 박스플롯
│   ├── scaling_indicators/ # 스케일링 지표 차트
│   ├── comprehensive/     # 종합 분석 차트
│   ├── default_group_analysis/ # Default 그룹별 분석 차트
│   ├── missing_default_analysis/ # 결측치 임계값별 Default 분석 차트
│   └── missing_analysis/  # 결측치 분석 차트
└── 📁 analysis/           # 특별 분석 결과
```

---

## 🚀 Master Runs - 통합 모델링 결과

### 🏆 최신 실행 결과 (2025-06-23)

#### `default_run_20250623_104052/` (⭐ **최고 성능**)
**앙상블 모델 포함 완전한 실행**

**🎯 최고 성능 모델:**
- **앙상블 모델**: F1: 0.3514, AUC: 0.9480 (Threshold: 0.10)
- **XGBoost Normal**: F1: 0.3226, AUC: 0.9502 (Threshold: 0.15)
- **RandomForest Normal**: F1: 0.3226, AUC: 0.9479 (Threshold: 0.20)

**📁 포함 파일:**
```
├── config.json                    # 실행 설정
├── models/                        # 학습된 모델들
│   ├── ensemble_model_model.joblib           # 🏆 앙상블 모델
│   ├── logisticregression_normal_model.joblib
│   ├── logisticregression_smote_model.joblib
│   ├── randomforest_normal_model.joblib
│   ├── randomforest_smote_model.joblib
│   ├── xgboost_normal_model.joblib
│   └── xgboost_smote_model.joblib
├── results/
│   ├── all_results.json           # 전체 결과 요약
│   └── summary_table.csv          # 성능 비교 테이블
└── visualizations/
    ├── cv_vs_test_comparison.png  # CV vs Test 성능 비교
    ├── ensemble_analysis.png      # 앙상블 분석
    ├── ensemble_weights.png       # 앙상블 가중치
    ├── feature_importance_comparison.png
    ├── normal_vs_smote_detailed.png
    ├── performance_comparison.png
    ├── precision_recall_curves.png
    ├── roc_curves_comparison.png
    └── threshold_optimization_analysis.png
```

#### `quick_test_20250623_130246/` (⚡ **빠른 테스트**)
**빠른 검증용 실행**

**🎯 주요 결과:**
- 동일한 모델 구성으로 빠른 검증
- 앙상블 모델 성능 확인
- 임계값 최적화 검증

### 🔧 핵심 기술적 개선사항

#### ✅ **임계값 자동 최적화**
- **기존**: 하드코딩된 0.3 임계값
- **개선**: 각 모델별 최적 임계값 자동 탐색 (0.1~0.85)
- **결과**: F1-Score 기준 최적화로 성능 향상

#### ✅ **앙상블 모델 도입**
- **구성**: 6개 기본 모델의 가중평균
- **가중치**: 검증 성능 기반 자동 계산
- **성능**: 단일 모델 대비 안정적 향상

#### ✅ **완전한 시각화 파이프라인**
- CV vs Test 성능 비교
- ROC/PR 곡선 분석
- 특성 중요도 비교
- 임계값 최적화 분석

---

## 📋 Reports - CSV 분석 보고서

### 📊 스케일링 분석 (최신)
- **`missing_analysis.csv`**: 결측치 패턴 분석
  - 33개 변수별 결측치 비율
  - 결측치 패턴 식별
  - 데이터 완전성 평가

- **`basic_statistics.csv`**: 기초 통계량
  - 평균, 표준편차, 왜도, 첨도
  - 변동계수, 범위, IQR
  - 33개 재무변수 완전 분석

- **`scaling_scores.csv`**: 스케일링 필요성 점수
  - 0-10점 스케일링 점수
  - 우선순위 분류 (고/중/저)
  - 스케일링 방법 추천

- **`scaling_recommendations.csv`**: 스케일링 방법 추천
  - RobustScaler vs StandardScaler
  - 변수별 최적 스케일러 선택
  - 추천 근거 제시

### 📊 Default & Missing Threshold 분석 (신규)
- **`missing_threshold_default_analysis.csv`**: 임계값별 데이터·Default 보존율 분석
- **`column_missing_changes_by_threshold.csv`**: 임계값별 컬럼별 결측치 비율 변화
- **`normal_companies_statistics.csv`**, **`default_companies_statistics.csv`**: Default 그룹별 기초 통계
- **`mean_comparison_analysis.csv`**, **`std_comparison_analysis.csv`**, **`comprehensive_group_comparison.csv`**: 그룹별 평균·표준편차·종합 비교
- **`default_group_analysis_report.txt`**: Default 그룹 비교 상세 리포트

### 🏆 모델 성능 보고서
각 master_run 디렉토리 내:
- **`all_results.json`**: 전체 모델 성능 요약
- **`summary_table.csv`**: 성능 비교 테이블
- **`lasso_selection_*.json`**: Lasso 특성 선택 결과

---

## 📊 Visualizations - 체계적 시각화

### 🔍 **Missing Analysis** (결측치 분석)
```
missing_analysis/
├── missing_pattern_heatmap.png    # 결측치 패턴 히트맵
├── missing_distribution_pie.png   # 결측치 분포 파이차트
├── data_completeness_analysis.png # 데이터 완전성 분석
└── missing_correlation_analysis.png # 결측치 상관관계
```

### 📈 **Distributions** (개별 변수 분포)
```
distributions/
├── 01_ROA_hist.png                # ROA 히스토그램
├── 02_TLTA_hist.png               # TLTA 히스토그램
├── ...                            # 33개 변수 개별 히스토그램
└── 33_현금흐름_기반_ROA_hist.png
```

### 📦 **Boxplots** (개별 변수 박스플롯)
```
boxplots/
├── 01_ROA_box.png                 # ROA 박스플롯
├── 02_TLTA_box.png                # TLTA 박스플롯
├── ...                            # 33개 변수 개별 박스플롯
└── 33_현금흐름_기반_ROA_box.png
```

### 📏 **Scaling Indicators** (스케일링 지표)
```
scaling_indicators/
├── 01_cv_vs_skewness.png          # 변동계수 vs 왜도
├── 02_range_vs_kurtosis.png       # 범위 vs 첨도
├── 03_mean_abs_distribution.png   # 평균 절댓값 분포
└── 04_std_distribution.png        # 표준편차 분포
```

### 🎯 **Comprehensive** (종합 분석)
```
comprehensive/
├── 01_scaling_scores.png          # 스케일링 점수 막대그래프
├── 02_priority_distribution.png   # 우선순위 분포 파이차트
├── 03_correlation_heatmap.png     # 상관관계 히트맵
└── 04_outlier_counts.png          # 이상치 개수 분석
```

### 📋 **Summary Charts** (요약 차트)
- **`00_ratio_distributions_summary.png`**: 전체 분포 요약
- **`00_ratio_boxplots_summary.png`**: 전체 박스플롯 요약

### 🏷️ **Default Group Analysis** (Default 그룹별 분석)
```
default_group_analysis/
├── 01_mean_comparison_top15.png       # 평균값 비교 (발생액 제외)
├── 02_std_comparison_top15.png        # 표준편차 비교 (발생액 제외)
├── 03_boxplot_comparison_top12.png    # 박스플롯 비교
├── 04_histogram_comparison_top6.png   # 히스토그램 비교
├── 05_statistics_heatmap.png          # 통계량 히트맵
└── 06_comprehensive_dashboard.png     # 종합 대시보드
```

### 🏷️ **Missing Threshold Default Analysis** (결측치 임계값별 Default 분석)
```
missing_default_analysis/
├── 01_missing_threshold_analysis.png        # 임계값별 데이터/Default 보존율
├── 02_data_count_changes.png                # 데이터 행 변화
├── 03_default_rate_changes.png              # Default 비율 변화
├── 05_remaining_missing_analysis.png        # 남은 결측치 분석
├── 06_column_missing_changes_heatmap.png    # 컬럼별 결측치 변화 히트맵
└── 04_comprehensive_dashboard.png            # 종합 대시보드
```

---

## 🔍 Analysis - 특별 분석

### 📊 데이터 품질 분석
- **Enhanced Outlier Analysis**: 이상치 종합 분석
- **New Variables Correlation**: 신규 변수 상관관계
- **Final Results Summary**: 최종 결과 요약

---

## 🚀 사용 방법

### 📖 최고 성능 모델 사용
```python
import joblib
import pandas as pd

# 앙상블 모델 로드 (최고 성능)
model = joblib.load('outputs/master_runs/default_run_20250623_104052/models/ensemble_model_model.joblib')

# 새로운 데이터 예측
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

# 최적 임계값 사용 (0.10)
binary_predictions = (probabilities >= 0.10).astype(int)
```

### 📊 성능 결과 확인
```python
import json
import pandas as pd

# 최신 실행 결과 로드
with open('outputs/master_runs/default_run_20250623_104052/results/all_results.json', 'r') as f:
    results = json.load(f)

# 성능 비교 테이블 확인
summary_df = pd.read_csv('outputs/master_runs/default_run_20250623_104052/results/summary_table.csv')
print(summary_df)
```

### 🔍 스케일링 분석 결과 확인
```python
# 스케일링 점수 확인
scaling_scores = pd.read_csv('outputs/reports/scaling_scores.csv')
high_priority = scaling_scores[scaling_scores['스케일링_점수'] >= 7]
print("고우선순위 스케일링 변수:")
print(high_priority[['비율명', '스케일링_점수', '추천_스케일러']])

# 결측치 분석 확인
missing_analysis = pd.read_csv('outputs/reports/missing_analysis.csv')
high_missing = missing_analysis[missing_analysis['결측치비율(%)'] > 20]
print("높은 결측치 변수:")
print(high_missing[['비율명', '결측치비율(%)']])
```

---

## 📈 주요 성과 요약

### 🏆 최고 성능 달성
| 지표 | 값 | 모델 | 임계값 |
|------|----|----- |--------|
| **F1-Score** | **0.3514** | 앙상블 모델 | 0.10 |
| **AUC** | **0.9502** | XGBoost Normal | - |
| **Precision** | **0.2400** | 앙상블 모델 | 0.10 |
| **Recall** | **0.6000** | 앙상블 모델 | 0.10 |

### 🔧 기술적 혁신
- ✅ **임계값 자동 최적화**: 하드코딩 제거, 모델별 최적화
- ✅ **앙상블 모델**: 6개 모델 가중평균으로 안정성 향상
- ✅ **완전한 시각화**: 80개 차트로 포괄적 분석
- ✅ **CSV 기반 보고서**: Excel 없이도 접근 가능
- ✅ **결측치 심화 분석**: 패턴 기반 결측치 이해

### 📊 데이터 품질 개선
- ✅ **36개 재무변수**: 기존 20개에서 16개 추가
- ✅ **스케일링 체계화**: 변수별 맞춤 스케일러 추천
- ✅ **결측치 투명성**: 완전한 결측치 패턴 분석
- ✅ **이상치 관리**: 카테고리별 이상치 현황 파악

### 🎯 실무 적용성
- ✅ **재현 가능성**: 모든 설정과 결과 저장
- ✅ **확장성**: Master Runner로 쉬운 실험 추가
- ✅ **해석 가능성**: 풍부한 시각화와 분석
- ✅ **유지보수성**: 체계적 파일 구조

---

## 📞 파일 관련 안내

### 🔍 주요 파일 위치
- **최고 성능 모델**: `master_runs/default_run_20250623_104052/models/ensemble_model_model.joblib`
- **성능 비교**: `master_runs/default_run_20250623_104052/results/summary_table.csv`
- **스케일링 분석**: `reports/scaling_*.csv`
- **시각화 요약**: `visualizations/00_ratio_*_summary.png`

### 🗂️ 명명 규칙
- **Master Runs**: `{config_name}_run_{YYYYMMDD_HHMMSS}/`
- **모델 파일**: `{algorithm}_{data_type}_model.joblib`
- **시각화**: `{순번}_{변수명}_{차트타입}.png`
- **보고서**: `{분석타입}.csv`

### ⚡ 빠른 재실행
```bash
# 전체 파이프라인 재실행
cd src/modeling
python run_master.py

# 스케일링 분석만 재실행
cd src/analysis
python analyze_scaling_needs.py
```

---

## 🔮 향후 계획

### 📈 모델 개선
- [ ] 딥러닝 모델 실험 (LSTM, Transformer)
- [ ] 시계열 특성을 활용한 모델링
- [ ] 업종별 특화 모델 개발

### 📊 분석 확장
- [ ] 시계열 패턴 분석
- [ ] 거시경제 변수 통합
- [ ] 실시간 예측 시스템 구축

### 🛠️ 시스템 개선
- [ ] 웹 대시보드 개발
- [ ] API 서버 구축
- [ ] 자동화된 모델 재학습

---

**📊 모든 결과는 완전히 재현 가능하며, 체계적으로 관리됩니다.**

**🚀 실무 적용을 위해 `ensemble_model_model.joblib` 사용을 권장합니다.**

---

*마지막 업데이트: 2025년 6월 23일 - Master Model Runner 시스템 도입 및 앙상블 모델 개발* 