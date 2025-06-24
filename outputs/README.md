# 📊 Outputs - 분석 결과 및 모델 저장소

이 디렉토리는 **한국 기업 부실예측 모델링 프로젝트**의 모든 분석 결과, 학습된 모델, 시각화 자료를 포함합니다.

## 📁 디렉토리 구조

```
outputs/
├── 📁 master_runs/         # 🔥 Master Model Runner 실행 결과들
│   ├── 📁 default_run_20250624_013703/       # 🏆 최신 앙상블 결과 (최고 성능)
│   ├── 📁 default_run_20250624_011424/       # 이전 실행 결과들
│   ├── 📁 default_run_20250624_004824/
│   └── 📁 default_run_20250623_*/            # 기타 실행 결과들
├── 📁 reports/             # CSV 형태 분석 보고서
├── 📁 visualizations/      # 체계적 시각화 자료
│   ├── 📁 distributions/      # 개별 변수 분포 차트 (33개)
│   ├── 📁 boxplots/          # 개별 변수 박스플롯 (33개)
│   ├── 📁 scaling_indicators/ # 스케일링 지표 차트
│   ├── 📁 comprehensive/     # 종합 분석 차트
│   ├── 📁 default_group_analysis/ # Default 그룹별 분석 차트
│   ├── 📁 missing_default_analysis/ # 결측치 임계값별 Default 분석
│   └── 📁 missing_analysis/  # 결측치 분석 차트
└── 📁 analysis/           # 특별 분석 결과
```

---

## 🚀 Master Runs - 통합 모델링 결과

### 🏆 **최신 실행 결과 (2025-06-24)**

#### `default_run_20250624_013703/` (⭐ **최고 성능 - 앙상블 모델 포함**)
**🎭 앙상블 모델링 완료 - 혁신적 성능 향상**

**🏆 최고 성능 모델:**
- **🎭 Ensemble Model**: **F1: 0.4096** (**21.3% 성능 향상**), AUC: 0.9808 (Threshold: 0.1)
- **XGBoost Normal**: F1: 0.3380, AUC: 0.9755 (Threshold: 0.1)
- **RandomForest Normal**: F1: 0.2381, AUC: 0.9793 (Threshold: 0.15)
- **LogisticRegression Normal**: F1: 0.2182, AUC: 0.9763 (Threshold: 0.15)

**📁 포함 파일:**
```
default_run_20250624_013703/
├── 📄 config.json                    # 실행 설정
├── 📁 models/                        # 🎭 학습된 모델들 (7개)
│   ├── 🏆 ensemble_model_model.joblib           # 앙상블 모델 (최고 성능)
│   ├── 📊 logisticregression_normal_model.joblib
│   ├── 📊 logisticregression_combined_model.joblib
│   ├── 📊 randomforest_normal_model.joblib
│   ├── 📊 randomforest_combined_model.joblib
│   ├── 📊 xgboost_normal_model.joblib
│   └── 📊 xgboost_combined_model.joblib
├── 📁 results/                       # 성능 결과
│   ├── 📄 all_results.json          # 전체 결과 요약 (2MB+)
│   ├── 📄 summary_table.csv         # 성능 비교 테이블
│   └── 📄 lasso_selection_normal.json # Lasso 특성 선택 결과
└── 📁 visualizations/               # 🎨 시각화 결과 (9개)
    ├── 🎭 ensemble_analysis.png     # 앙상블 종합 분석
    ├── 🎭 ensemble_weights.png      # 앙상블 가중치 분포
    ├── ⚡ threshold_optimization_analysis.png # 임계값 최적화 분석
    ├── 📈 precision_recall_curves.png # Precision-Recall 곡선
    ├── 📈 roc_curves_comparison.png  # ROC 곡선 비교
    ├── 📊 feature_importance_comparison.png # 특성 중요도 비교
    ├── 📊 cv_vs_test_comparison.png  # CV vs Test 성능 비교
    ├── 📊 performance_comparison.png # 전체 성능 비교
    └── 📊 sampling_strategy_comparison.png # 샘플링 전략 비교
```

#### 🔥 **핵심 기술적 혁신**

##### ✅ **앙상블 모델링 도입**
- **구성**: 6개 기본 모델의 가중평균 (3개 알고리즘 × 2개 데이터 타입)
- **자동 가중치**: 검증 성능 기반 소프트맥스 가중치 계산
- **성능 향상**: 개별 모델 최고 대비 **21.3% F1 개선** (0.3380 → 0.4096)
- **안정성**: 예측 분산 감소로 일반화 성능 향상

##### ✅ **자동 임계값 최적화**
- **기존**: 하드코딩된 threshold=0.5
- **개선**: 각 모델별 최적 임계값 자동 탐색 (0.1~0.85, 16개 포인트)
- **최적화 지표**: F1-Score 기준 최적화
- **결과**: 각 모델의 진정한 잠재력 발휘

##### ✅ **Data Leakage 완전 방지**
- **기존 문제**: 전체 데이터에 SMOTE 적용 후 CV 수행
- **해결책**: CV 내부 동적 SMOTE 적용
- **효과**: 현실적이고 신뢰할 수 있는 성능 수치

##### ✅ **포괄적 시각화**
- **15개 이상** 분석 차트 자동 생성
- 앙상블 분석, 임계값 최적화, ROC/PR 곡선 등
- 비즈니스 의사결정 지원 시각화

### 🔍 **이전 실행 결과들**

#### `default_run_20250624_011424/` (완전한 파이프라인)
- 앙상블 모델 포함 전체 실행
- CV vs Test 성능 비교 추가
- 특성 중요도 상세 분석

#### `default_run_20250624_004824/` (성능 최적화)
- 앙상블 모델 첫 도입 실험
- 가중치 자동 계산 시스템 구현
- 성능 비교 프레임워크 구축

#### `default_run_20250623_*` (기졀 실행들)
- 개별 모델 성능 최적화
- 임계값 최적화 시스템 개발
- SMOTE Data Leakage 방지 구현

---

## 📋 Reports - CSV 분석 보고서

### 📊 **최신 스케일링 분석**
- **`missing_analysis.csv`**: 결측치 패턴 분석
  - 33개 변수별 결측치 비율 분석
  - 결측치 패턴 식별 및 데이터 완전성 평가
  - 임계값별 데이터 보존 전략 수립

- **`basic_statistics.csv`**: 기초 통계량 종합
  - 평균, 표준편차, 왜도, 첨도 완전 분석
  - 변동계수, 범위, IQR 계산
  - 33개 재무변수 통계적 특성 파악

- **`scaling_scores.csv`**: 스케일링 필요성 점수
  - 0-10점 스케일링 점수 체계
  - 우선순위 분류 (고/중/저)
  - 변수별 최적 스케일러 추천

### 📊 **Default & Missing Threshold 분석 (신규)**
- **`missing_threshold_default_analysis.csv`**: 임계값별 데이터·Default 보존율 분석
- **`column_missing_changes_by_threshold.csv`**: 임계값별 컬럼별 결측치 비율 변화
- **`normal_companies_statistics.csv`**, **`default_companies_statistics.csv`**: Default 그룹별 기초 통계
- **`comprehensive_group_comparison.csv`**: 정상·부실 기업 종합 비교 분석
- **`default_group_analysis_report.txt`**: Default 그룹 비교 상세 리포트

### 🏆 **모델 성능 보고서 (최신)**
각 master_run 디렉토리 내:
- **`all_results.json`**: 전체 모델 성능 요약 (앙상블 포함)
- **`summary_table.csv`**: 성능 비교 테이블 (임계값 최적화 반영)
- **`lasso_selection_*.json`**: Lasso 특성 선택 결과

---

## 📊 Visualizations - 체계적 시각화

### 🔍 **Missing Analysis** (결측치 분석)
```
missing_analysis/
├── 📄 01_missing_rates_by_variable.png    # 변수별 결측치 비율
├── 📄 02_missing_pattern_heatmap.png      # 결측치 패턴 히트맵
├── 📄 03_missing_level_distribution.png   # 결측치 수준 분포
└── 📄 04_missing_correlation_matrix.png   # 결측치 상관관계
```

### 📈 **Distributions** (개별 변수 분포) - 33개 차트
```
distributions/
├── 📄 01_총자산수익률_hist.png              # ROA 히스토그램
├── 📄 02_총부채_대_총자산_hist.png           # TLTA 히스토그램
├── 📄 03_운전자본_대_총자산_hist.png         # WC_TA 히스토그램
├── 📄 ...                                # 중간 변수들
└── 📄 33_현금흐름_기반_ROA_hist.png         # 마지막 변수
```

### 📦 **Boxplots** (개별 변수 박스플롯) - 33개 차트
```
boxplots/
├── 📄 01_총자산수익률_box.png               # ROA 박스플롯
├── 📄 02_총부채_대_총자산_box.png            # TLTA 박스플롯
├── 📄 03_운전자본_대_총자산_box.png          # WC_TA 박스플롯
├── 📄 ...                                # 중간 변수들
└── 📄 33_현금흐름_기반_ROA_box.png          # 마지막 변수
```

### 📏 **Scaling Indicators** (스케일링 지표)
```
scaling_indicators/
├── 📄 01_cv_vs_skewness.png          # 변동계수 vs 왜도 분석
├── 📄 02_range_vs_kurtosis.png       # 범위 vs 첨도 분석
├── 📄 03_mean_abs_distribution.png   # 평균 절댓값 분포
└── 📄 04_scaling_priority_scores.png # 스케일링 우선순위 점수
```

### 🎯 **Comprehensive** (종합 분석)
```
comprehensive/
├── 📄 01_scaling_scores.png          # 스케일링 점수 막대그래프
├── 📄 02_priority_distribution.png   # 우선순위 분포 파이차트
├── 📄 03_correlation_heatmap.png     # 상관관계 히트맵 (17×17)
└── 📄 04_outlier_analysis.png        # 이상치 분석 종합
```

### 📋 **Summary Charts** (요약 차트)
- **`00_ratio_distributions_summary.png`**: 전체 분포 요약 (4×8 서브플롯)
- **`00_ratio_boxplots_summary.png`**: 전체 박스플롯 요약 (4×8 서브플롯)

### 🏷️ **Default Group Analysis** (Default 그룹별 분석)
```
default_group_analysis/
├── 📄 01_mean_comparison_top15.png       # 평균값 비교 (상위 15개)
├── 📄 02_std_comparison_top15.png        # 표준편차 비교 (상위 15개)
├── 📄 03_boxplot_comparison_top12.png    # 박스플롯 비교 (상위 12개)
├── 📄 04_histogram_comparison_top6.png   # 히스토그램 비교 (상위 6개)
├── 📄 05_statistics_heatmap.png          # 통계량 히트맵
└── 📄 06_comprehensive_dashboard.png     # 종합 대시보드
```

### 🏷️ **Missing Threshold Default Analysis** (결측치 임계값별 Default 분석)
```
missing_default_analysis/
├── 📄 01_missing_threshold_analysis.png        # 임계값별 데이터/Default 보존율
├── 📄 02_data_count_changes.png                # 데이터 행 변화 추이
├── 📄 03_default_rate_changes.png              # Default 비율 변화 추이
├── 📄 04_comprehensive_dashboard.png           # 종합 대시보드
├── 📄 05_remaining_missing_analysis.png        # 남은 결측치 분석
└── 📄 06_column_missing_changes_heatmap.png    # 컬럼별 결측치 변화 히트맵
```

---

## 🏆 **핵심 성과 요약**

### 🎭 **앙상블 모델 성과** (Test Set)
| 모델 | 데이터 타입 | **최적 Threshold** | **Test AUC** | **Test F1** | **Test Precision** | **Test Recall** |
|-----|-------------|------------------|--------------|------------|------------------|----------------|
| **🎭 Ensemble** | **Mixed** | **0.1** | **0.9808** | **🔥 0.4096** | **0.2982** | **0.6538** |

### 🥇 **개별 모델 최고 성과** (자동 임계값 최적화)
| 모델 | 데이터 | **최적 Threshold** | **Test AUC** | **Test F1** | **개선율** |
|-----|--------|------------------|--------------|------------|------------|
| **XGBoost** | Normal | **0.1** | **0.9755** | **0.3380** | +84% |
| **RandomForest** | Normal | **0.15** | **0.9793** | **0.2381** | +67% |
| **LogisticRegression** | Normal | **0.15** | **0.9763** | **0.2182** | +53% |

### 📊 **데이터 확장 성과**
- **관측치**: 16,197개 → **22,780개** (40% 증가)
- **부실기업**: 132개 → **2,922개** (22배 증가)
- **데이터 품질**: 100% Complete (결측치 없음)

## 🔧 **기술적 혁신 사항**

### 1. **마스터 러너 시스템**
- **통합 파이프라인**: 6개 모델 + 앙상블 한 번에 실행
- **템플릿 시스템**: production/quick_test/lasso_focus
- **중앙 설정 관리**: JSON 기반 하이퍼파라미터 통합

### 2. **앙상블 모델링**
- **가중 평균**: 검증 성능 기반 자동 가중치 계산
- **성능 향상**: 21.3% F1 개선 (0.3380 → 0.4096)
- **안정성**: 예측 분산 감소 및 일반화 성능 향상

### 3. **자동 임계값 최적화**
- **Grid Search**: 0.1~0.85 범위 16개 포인트 탐색
- **최적화 지표**: F1/Precision/Recall 중 선택
- **성능 혁신**: 최대 84% F1 성능 향상

### 4. **Data Leakage 방지**
- **동적 SMOTE**: CV 내부에서 각 fold별 적용
- **원본 검증**: 합성 데이터 오염 완전 방지
- **신뢰성**: 현실적 일반화 성능 평가

## 📈 **사용 방법**

### 🚀 **최신 결과 확인**
```bash
# 최신 앙상블 결과 확인
ls outputs/master_runs/default_run_20250624_013703/

# 성능 요약 테이블 확인
cat outputs/master_runs/default_run_20250624_013703/results/summary_table.csv

# 앙상블 분석 시각화 확인
open outputs/master_runs/default_run_20250624_013703/visualizations/ensemble_analysis.png
```

### 📊 **성능 비교 분석**
```python
import pandas as pd
import json

# 성능 요약 로드
summary = pd.read_csv('outputs/master_runs/default_run_20250624_013703/results/summary_table.csv')

# 최고 성능 모델 확인
best_model = summary.loc[summary['Test_F1'].idxmax()]
print(f"최고 성능: {best_model['Model']} - F1: {best_model['Test_F1']:.4f}")

# 전체 결과 상세 분석
with open('outputs/master_runs/default_run_20250624_013703/results/all_results.json', 'r') as f:
    all_results = json.load(f)
```

### 🎨 **시각화 분석**
```python
# 앙상블 가중치 확인
ensemble_weights_chart = 'outputs/master_runs/default_run_20250624_013703/visualizations/ensemble_weights.png'

# 임계값 최적화 과정 확인
threshold_analysis = 'outputs/master_runs/default_run_20250624_013703/visualizations/threshold_optimization_analysis.png'

# 성능 비교 확인
performance_comparison = 'outputs/master_runs/default_run_20250624_013703/visualizations/performance_comparison.png'
```

## 📋 **품질 보증 및 검증**

### ✅ **모델 검증 완료**
- [x] 앙상블 모델 성능 검증 (F1: 0.4096)
- [x] 자동 임계값 최적화 검증
- [x] Data Leakage 방지 메커니즘 구현
- [x] CV vs Test 성능 일관성 확인
- [x] 특성 중요도 안정성 검증

### ✅ **데이터 품질 검증**
- [x] 100% Complete 데이터 사용
- [x] 다중공선성 해결 (VIF < 5)
- [x] 시계열 정렬 및 Look-ahead Bias 방지
- [x] 부실기업 비율 적정성 (12.83%)

### ✅ **시각화 품질 검증**
- [x] 모든 차트 한글 폰트 지원
- [x] 고해상도 PNG 생성 (300 DPI)
- [x] 색맹 친화적 색상 팔레트 사용
- [x] 비즈니스 해석 가능한 시각화

## 🚀 **향후 개발 계획**

### 📈 **성능 개선**
- [ ] Stacking 앙상블 모델 추가
- [ ] 딥러닝 모델 실험 (Neural Network, LSTM)
- [ ] AutoML 파이프라인 구축
- [ ] 하이퍼파라미터 자동 최적화 고도화

### 🔧 **시스템 개선**
- [ ] 실시간 예측 API 개발
- [ ] 모델 드리프트 모니터링 시스템
- [ ] A/B 테스트 프레임워크
- [ ] MLOps 파이프라인 구축

### 📊 **분석 도구 확장**
- [ ] SHAP 해석 도구 추가
- [ ] LIME 로컬 해석 기능
- [ ] 특성 상호작용 분석
- [ ] 비즈니스 임팩트 분석

## 💡 **사용 팁**

### 🎯 **결과 해석 가이드**
1. **F1-Score 우선**: 불균형 데이터에서 가장 신뢰할 수 있는 지표
2. **AUC 참고**: 전체적인 분류 성능 평가
3. **Precision vs Recall**: 비즈니스 목적에 따른 선택
4. **앙상블 우선**: 개별 모델 대비 안정적 고성능

### 📊 **비교 분석 방법**
- **시계열 비교**: 여러 실행 결과 간 성능 추이 분석
- **모델 간 비교**: 알고리즘별 특성 및 장단점 파악
- **데이터 타입 비교**: Normal vs Combined 데이터 효과 분석
- **임계값 영향 분석**: Threshold 변화에 따른 성능 변화

---

## 🔗 **관련 링크**

- **📊 최종 결과 보고서**: [FINAL_RESULTS_100_COMPLETE.md](../FINAL_RESULTS_100_COMPLETE.md)
- **🏗️ 모델링 가이드**: [src/modeling/README.md](../src/modeling/README.md)
- **📈 대시보드**: [dashboard/README.md](../dashboard/README.md)
- **🔧 데이터 가이드**: [data/final/README.md](../data/final/README.md)

---

**프로젝트 상태**: ✅ **완료** (앙상블 모델링 포함)  
**최고 성능**: 🏆 **F1: 0.4096** (앙상블 모델)  
**데이터 품질**: 🏆 **A+** (100% Complete, VIF < 5)  
**최종 업데이트**: 2025-06-24  
**분석팀**: 금융 ML 전문팀 