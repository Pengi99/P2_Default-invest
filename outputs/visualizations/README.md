# 📊 Visualizations - 체계적 시각화 자료

이 디렉토리는 한국 기업 부실예측 모델링 프로젝트의 **모든 시각화 자료**를 체계적으로 정리한 곳입니다.

## 🎯 시각화 개요

- **📈 총 시각화 차트**: **90개 이상**
- **🎨 카테고리**: 8개 주요 분석 영역
- **📊 차트 유형**: 히스토그램, 박스플롯, 히트맵, 대시보드 등
- **🔍 분석 범위**: 변수별 분포부터 종합 대시보드까지

## 📁 디렉토리 구조

```
outputs/visualizations/
├── 📄 00_ratio_distributions_summary.png     # 📊 전체 분포 요약 (4×8 그리드)
├── 📄 00_ratio_boxplots_summary.png          # 📦 전체 박스플롯 요약 (4×8 그리드)
├── 📁 distributions/                          # 📈 개별 변수 분포 (33개)
├── 📁 boxplots/                              # 📦 개별 변수 박스플롯 (33개)
├── 📁 scaling_indicators/                     # 📏 스케일링 지표 분석 (4개)
├── 📁 comprehensive/                          # 🎯 종합 분석 차트 (4개)
├── 📁 default_group_analysis/                 # 🏷️ Default 그룹별 분석 (6개)
├── 📁 missing_default_analysis/               # 🔍 결측치-Default 관계 분석 (6개)
├── 📁 missing_analysis/                       # ❓ 결측치 패턴 분석 (4개)
└── 📄 README.md                              # 현재 파일
```

---

## 📊 **요약 시각화** (Overview Charts)

### 🎨 **00_ratio_distributions_summary.png**
**전체 재무비율 분포 한눈에 보기**

**특징:**
- **4×8 서브플롯 구성**: 32개 재무변수 히스토그램
- **일관된 스타일**: 통일된 색상 및 폰트
- **밀도 곡선 오버레이**: 분포 형태 명확히 표시
- **한글 라벨**: 변수명 한글 표시

**활용법:**
- 프레젠테이션용 고해상도 이미지
- 전체 데이터 분포 패턴 빠른 파악
- 이상 분포 변수 즉시 식별

### 📦 **00_ratio_boxplots_summary.png**
**전체 재무비율 박스플롯 한눈에 보기**

**특징:**
- **4×8 서브플롯 구성**: 32개 재무변수 박스플롯
- **이상치 표시**: 아웃라이어 점으로 표시
- **사분위수 정보**: Q1, Median, Q3 명확히 표시
- **스케일 정규화**: 변수 간 비교 용이

**활용법:**
- 이상치 분포 한눈에 파악
- 변수별 분산 정도 비교
- 데이터 품질 체크

---

## 📈 **개별 변수 분포** (Individual Distributions)

### 📁 distributions/ (33개 차트)
**각 재무변수별 상세 히스토그램**

**파일 구조:**
```
distributions/
├── 📄 01_총자산수익률_hist.png                    # ROA 히스토그램
├── 📄 02_총부채_대_총자산_hist.png                 # TLTA 히스토그램
├── 📄 03_운전자본_대_총자산_hist.png               # WC_TA 히스토그램
├── 📄 04_영업현금흐름_대_총부채_hist.png           # CFO_TD 히스토그램
├── 📄 05_주가변동성_hist.png                      # SIGMA 히스토그램
├── 📄 06_이익잉여금_대_총자산_hist.png             # RE_TA 히스토그램
├── 📄 07_EBIT_대_총자산_hist.png                  # EBIT_TA 히스토그램
├── 📄 08_시장가치_대_총부채_hist.png               # MVE_TL 히스토그램
├── 📄 09_매출_대_총자산_hist.png                  # S_TA 히스토그램
├── 📄 10_유동부채_대_유동자산_hist.png             # CLCA 히스토그램
├── 📄 11_영업이익음수여부_hist.png                 # OENEG 히스토그램
├── 📄 12_유동비율_hist.png                        # CR 히스토그램
├── 📄 13_영업현금흐름_대_총자산_hist.png           # CFO_TA 히스토그램
├── 📄 14_총부채_대_시장가치총자산_hist.png         # TLMTA 히스토그램
├── 📄 15_3개월수익률_hist.png                     # RET_3M 히스토그램
├── 📄 16_9개월수익률_hist.png                     # RET_9M 히스토그램
├── 📄 17_시장가대장부가_hist.png                  # MB 히스토그램
├── 📄 18_...                                      # 추가 변수들
└── 📄 33_현금흐름_기반_ROA_hist.png               # 마지막 변수
```

**차트 특징:**
- **고해상도**: 300 DPI PNG 형식
- **한글 폰트**: NanumGothic 적용
- **통계 정보**: 평균, 표준편차, 왜도, 첨도 표시
- **밀도 곡선**: KDE 오버레이로 분포 형태 강조

### 📁 boxplots/ (33개 차트)
**각 재무변수별 상세 박스플롯**

**파일 구조:**
```
boxplots/
├── 📄 01_총자산수익률_box.png                     # ROA 박스플롯
├── 📄 02_총부채_대_총자산_box.png                  # TLTA 박스플롯
├── 📄 03_운전자본_대_총자산_box.png                # WC_TA 박스플롯
├── 📄 ...                                        # 중간 변수들
└── 📄 33_현금흐름_기반_ROA_box.png                # 마지막 변수
```

**차트 특징:**
- **이상치 강조**: 빨간 점으로 outlier 표시
- **사분위수 라벨**: Q1, Q2, Q3 값 텍스트로 표시
- **통계 정보**: Min, Max, IQR 정보 포함
- **색상 코딩**: 변수 특성에 따른 색상 분류

---

## 📏 **스케일링 지표 분석** (Scaling Indicators)

### 📁 scaling_indicators/ (4개 차트)
**변수별 스케일링 필요성 분석**

**파일 구조:**
```
scaling_indicators/
├── 📄 01_cv_vs_skewness.png                # 변동계수 vs 왜도 분석
├── 📄 02_range_vs_kurtosis.png             # 범위 vs 첨도 분석
├── 📄 03_mean_abs_distribution.png         # 평균 절댓값 분포
└── 📄 04_scaling_priority_scores.png       # 스케일링 우선순위 점수
```

### 🎯 **01_cv_vs_skewness.png**
**변동계수와 왜도의 관계 분석**

**분석 내용:**
- **X축**: 변동계수 (CV = σ/μ)
- **Y축**: 왜도 (Skewness)
- **포인트 크기**: 데이터 범위 반영
- **색상**: 스케일링 우선순위 반영

**해석:**
- 우상단 변수들이 스케일링 필요성 높음
- 대칭 분포일수록 왜도 0에 가까움
- 높은 CV는 큰 변동성 의미

### 📊 **04_scaling_priority_scores.png**
**스케일링 우선순위 점수 (0-10점)**

**점수 계산:**
```python
scaling_score = (
    cv_normalized * 3.0 +           # 변동계수 (30%)
    range_normalized * 2.5 +        # 범위 (25%)
    abs_skew_normalized * 2.0 +     # 왜도 절댓값 (20%)
    abs_kurt_normalized * 1.5 +     # 첨도 절댓값 (15%)
    mean_abs_normalized * 1.0       # 평균 절댓값 (10%)
)
```

**우선순위 분류:**
- **고우선순위 (7-10점)**: 즉시 스케일링 필요
- **중우선순위 (4-7점)**: 스케일링 권장
- **저우선순위 (0-4점)**: 스케일링 선택적

---

## 🎯 **종합 분석** (Comprehensive Analysis)

### 📁 comprehensive/ (4개 차트)
**다차원 종합 분석 차트**

**파일 구조:**
```
comprehensive/
├── 📄 01_scaling_scores.png                # 스케일링 점수 막대그래프
├── 📄 02_priority_distribution.png         # 우선순위 분포 파이차트
├── 📄 03_correlation_heatmap.png           # 상관관계 히트맵 (17×17)
└── 📄 04_outlier_analysis.png              # 이상치 분석 종합
```

### 🔥 **03_correlation_heatmap.png**
**17개 최종 변수 상관관계 히트맵**

**특징:**
- **17×17 매트릭스**: 최종 선택된 변수들만
- **색상 범위**: -1(완전 음의 상관) ~ +1(완전 양의 상관)
- **수치 표시**: 상관계수 값 텍스트로 표시
- **다중공선성 확인**: 높은 상관관계(>0.8) 쌍 식별

**핵심 발견:**
- **WC_TA ↔ CLCA**: -0.823 (높은 음의 상관)
- **대부분 변수**: |r| < 0.5 (양호한 독립성)
- **VIF < 5**: 다중공선성 해결 확인

---

## 🏷️ **Default 그룹별 분석** (Default Group Analysis)

### 📁 default_group_analysis/ (6개 차트)
**정상 vs 부실 기업 그룹 비교 분석**

**파일 구조:**
```
default_group_analysis/
├── 📄 01_mean_comparison_top15.png          # 평균값 비교 (상위 15개)
├── 📄 02_std_comparison_top15.png           # 표준편차 비교 (상위 15개)
├── 📄 03_boxplot_comparison_top12.png       # 박스플롯 비교 (상위 12개)
├── 📄 04_histogram_comparison_top6.png      # 히스토그램 비교 (상위 6개)
├── 📄 05_statistics_heatmap.png             # 통계량 히트맵
└── 📄 06_comprehensive_dashboard.png        # 종합 대시보드
```

### 🎯 **01_mean_comparison_top15.png**
**정상 vs 부실 기업 평균값 비교**

**분석 기준:**
- **선별 기준**: |정상_평균 - 부실_평균| 큰 순서
- **상위 15개 변수**: 가장 차이 나는 변수들
- **시각화**: 그룹별 막대 차트
- **통계적 유의성**: t-test 결과 표시

**핵심 발견:**
- **ROA**: 정상(0.051) vs 부실(-0.089) - 큰 차이
- **TLTA**: 정상(0.523) vs 부실(0.789) - 부실기업 부채 많음
- **WC_TA**: 정상(0.078) vs 부실(-0.021) - 유동성 차이

### 🎨 **06_comprehensive_dashboard.png**
**Default 그룹 분석 종합 대시보드**

**구성 요소:**
```
┌─────────────────┬─────────────────┐
│  📊 평균 비교    │  📈 분포 비교    │
│  (막대 차트)     │  (히스토그램)    │
├─────────────────┼─────────────────┤
│  📦 박스플롯     │  🎯 통계 요약    │
│  (상위 6개)      │  (테이블)        │
└─────────────────┴─────────────────┘
```

---

## 🔍 **결측치-Default 관계 분석** (Missing-Default Analysis)

### 📁 missing_default_analysis/ (6개 차트)
**결측치 임계값별 Default 영향 분석**

**파일 구조:**
```
missing_default_analysis/
├── 📄 01_missing_threshold_analysis.png         # 임계값별 데이터/Default 보존율
├── 📄 02_data_count_changes.png                 # 데이터 행 변화 추이
├── 📄 03_default_rate_changes.png               # Default 비율 변화 추이
├── 📄 04_comprehensive_dashboard.png            # 종합 대시보드
├── 📄 05_remaining_missing_analysis.png         # 남은 결측치 분석
└── 📄 06_column_missing_changes_heatmap.png     # 컬럼별 결측치 변화 히트맵
```

### 📊 **01_missing_threshold_analysis.png**
**결측치 임계값별 영향도 분석**

**분석 내용:**
- **X축**: 결측치 임계값 (0% ~ 50%)
- **Y축 (왼쪽)**: 남은 데이터 행 수
- **Y축 (오른쪽)**: Default 기업 수
- **이중 Y축**: 데이터 보존 vs Default 보존 trade-off

**핵심 인사이트:**
- **20% 임계값**: 데이터 80% 보존, Default 90% 보존
- **30% 임계값**: 데이터 60% 보존, Default 75% 보존
- **최적점**: 25% 임계값에서 균형

---

## ❓ **결측치 패턴 분석** (Missing Pattern Analysis)

### 📁 missing_analysis/ (4개 차트)
**결측치 분포 및 패턴 상세 분석**

**파일 구조:**
```
missing_analysis/
├── 📄 01_missing_rates_by_variable.png         # 변수별 결측치 비율
├── 📄 02_missing_pattern_heatmap.png           # 결측치 패턴 히트맵
├── 📄 03_missing_level_distribution.png        # 결측치 수준 분포
└── 📄 04_missing_correlation_matrix.png        # 결측치 상관관계
```

### 🔥 **02_missing_pattern_heatmap.png**
**결측치 패턴 히트맵**

**특징:**
- **X축**: 33개 변수
- **Y축**: 샘플 (랜덤 선택 1,000개)
- **색상**: 결측(빨강) vs 관측(파랑)
- **패턴 식별**: 함께 결측되는 변수 그룹 확인

**발견 패턴:**
- **재무제표 그룹**: 함께 결측되는 경향
- **시장 지표 그룹**: 독립적 결측 패턴
- **무작위성**: 대부분 MCAR (Missing Completely at Random)

---

## 🎨 **시각화 품질 관리**

### ✅ **품질 표준**
- **해상도**: 300 DPI (출판 품질)
- **파일 형식**: PNG (투명도 지원)
- **색상 팔레트**: 색맹 친화적 색상
- **폰트**: NanumGothic (한글 지원)
- **크기**: 최소 1200×800 픽셀

### 🔧 **기술 스택**
- **Matplotlib**: 기본 차트 엔진
- **Seaborn**: 통계 차트 전문
- **Plotly**: 일부 대화형 차트
- **Pandas**: 데이터 전처리
- **NumPy**: 수치 계산

### 📏 **스타일 가이드**
```python
# 표준 시각화 설정
plt.rcParams.update({
    'font.family': 'NanumGothic',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
```

## 🚀 **활용 방법**

### 📊 **프레젠테이션용**
```bash
# 요약 차트 사용
outputs/visualizations/00_ratio_distributions_summary.png
outputs/visualizations/00_ratio_boxplots_summary.png

# 종합 대시보드 사용
outputs/visualizations/comprehensive/
outputs/visualizations/default_group_analysis/06_comprehensive_dashboard.png
```

### 🔍 **상세 분석용**
```bash
# 특정 변수 심화 분석
outputs/visualizations/distributions/01_총자산수익률_hist.png
outputs/visualizations/boxplots/01_총자산수익률_box.png

# 그룹 간 비교 분석
outputs/visualizations/default_group_analysis/
```

### 📋 **보고서용**
```bash
# 결측치 분석 섹션
outputs/visualizations/missing_analysis/
outputs/visualizations/missing_default_analysis/

# 데이터 품질 섹션
outputs/visualizations/scaling_indicators/
outputs/visualizations/comprehensive/
```

## 💡 **해석 가이드**

### 📈 **분포 해석**
- **정규분포**: 대칭적 종 모양
- **왜도 > 0**: 우측 꼬리 긴 분포
- **왜도 < 0**: 좌측 꼬리 긴 분포
- **첨도 > 3**: 뾰족한 분포 (leptokurtic)

### 📦 **박스플롯 해석**
- **상자 높이**: IQR (Q3-Q1) 크기
- **중앙선 위치**: 중앙값 위치
- **수염 길이**: 1.5×IQR 범위
- **점들**: 이상치 (outliers)

### 🎯 **상관관계 해석**
- **|r| > 0.8**: 매우 강한 상관관계
- **0.6 < |r| < 0.8**: 강한 상관관계
- **0.4 < |r| < 0.6**: 중간 상관관계
- **|r| < 0.4**: 약한 상관관계

---

## 🔗 **관련 자료**

- **📊 분석 결과**: [outputs/README.md](../README.md)
- **📈 모델링**: [src/modeling/README.md](../../src/modeling/README.md)
- **📋 최종 보고서**: [FINAL_RESULTS_100_COMPLETE.md](../../FINAL_RESULTS_100_COMPLETE.md)
- **🔧 데이터**: [data/final/README.md](../../data/final/README.md)

---

**시각화 상태**: ✅ **완료** (90개 이상 차트)  
**품질 수준**: 🏆 **출판 품질** (300 DPI)  
**커버리지**: 📊 **100%** (모든 분석 영역)  
**최종 업데이트**: 2025-06-24  
**제작팀**: 데이터 시각화 전문팀
