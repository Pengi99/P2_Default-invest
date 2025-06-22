# FS_100 데이터셋 다중공선성 분석 결과

## 📊 분석 개요

FS_100 데이터셋의 18개 재무비율 변수에 대한 포괄적인 다중공선성 분석을 수행했습니다.

- **분석 일시**: 2025-06-22 08:57:28
- **총 관측치**: 16,197개
- **원본 변수**: 18개
- **최종 변수**: 17개

## 🔍 주요 발견사항

### 1. 완전한 다중공선성 문제
- **K2_Score_Original** 변수가 완전한 다중공선성의 원인으로 확인됨
- 이 변수는 다른 변수들의 선형결합으로 표현 가능
- 제거 후 모든 VIF 값이 정상 범위로 개선됨

### 2. VIF (Variance Inflation Factor) 분석 결과

| 변수명 | VIF 값 | 해석 |
|--------|--------|------|
| WC_TA | 4.97 | 양호 |
| TLTA | 4.52 | 양호 |
| EBIT_TA | 3.46 | 양호 |
| CLCA | 3.24 | 양호 |
| CFO_TA | 2.98 | 양호 |
| TLMTA | 2.86 | 양호 |
| CFO_TD | 2.47 | 양호 |
| ROA | 2.43 | 양호 |
| CR | 2.31 | 양호 |
| RE_TA | 1.84 | 양호 |
| MVE_TL | 1.62 | 양호 |
| S_TA | 1.30 | 양호 |
| OENEG | 1.23 | 양호 |
| RET_9M | 1.20 | 양호 |
| MB | 1.19 | 양호 |
| RET_3M | 1.16 | 양호 |
| SIGMA | 1.01 | 매우 양호 |

**VIF 해석 기준:**
- VIF < 5: 낮은 다중공선성 (양호)
- 5 ≤ VIF < 10: 보통 다중공선성
- VIF ≥ 10: 높은 다중공선성 (문제)

### 3. 상관관계 분석
- **높은 상관관계 쌍**: 1개
  - WC_TA ↔ CLCA: -0.823 (강한 음의 상관관계)
- 이는 운전자본 비율과 유동자산 대비 유동부채 비율의 개념적 연관성을 반영

### 4. 주성분 분석 (PCA) 결과
- **95% 분산 설명**: 13개 성분 필요 (18개 중)
- **차원 축소 가능성**: 27.8%
- 첫 번째 주성분이 27.6%의 분산 설명
- 상위 3개 성분이 52.5%의 분산 설명

## 📈 생성된 시각화 자료

1. **correlation_heatmap**: 전체 변수 간 상관계수 히트맵
2. **vif_analysis_cleaned**: 정리된 VIF 결과 차트
3. **variable_removal_process**: 변수 제거 과정 시각화
4. **pca_variance_explained**: PCA 설명 분산 차트

## 💡 모델링 권장사항

### 1. 변수 선택
- ✅ **K2_Score_Original 제거**: 완전한 다중공선성으로 인해 필수 제거
- ⚠️ **WC_TA vs CLCA**: 높은 상관관계로 인해 둘 중 하나 제거 고려

### 2. 모델링 기법
- **정규화 회귀 (Ridge/Lasso)**: 다중공선성에 robust한 모델 사용
- **변수 선택**: 상관관계가 높은 변수 쌍 중 하나 제거
- **차원 축소**: PCA 적용 시 13개 성분으로 95% 분산 설명 가능

### 3. 추가 고려사항
- 현재 모든 VIF < 5로 양호한 상태
- 도메인 지식을 바탕으로 WC_TA와 CLCA 중 선택
- 모델 성능에 따라 추가 변수 선택 기법 적용

## 📁 파일 구조

```
outputs/analysis/
├── comprehensive_multicollinearity_analysis_20250622_085728.json  # 종합 분석 결과
├── final_vif_results_20250622_085728.csv                          # 최종 VIF 결과
├── final_correlation_matrix_20250622_085728.csv                   # 상관계수 행렬
├── correlation_heatmap_20250622_085727.png                        # 상관계수 히트맵
├── vif_analysis_cleaned_20250622_085727.png                       # VIF 분석 차트
├── variable_removal_process_20250622_085727.png                   # 변수 제거 과정
└── pca_variance_explained_20250622_085727.png                     # PCA 분석 차트
```

## 🔧 사용된 도구

- **Python 라이브러리**: pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn
- **분석 기법**: VIF 계산, 상관분석, 특이값 분해, PCA
- **시각화**: 히트맵, 막대그래프, 선형 차트

## 📊 결론

FS_100 데이터셋의 다중공선성 문제는 **K2_Score_Original 변수 제거**로 대부분 해결되었습니다. 남은 17개 변수는 모두 양호한 VIF 값을 보이며, 추가적인 변수 선택이나 정규화 기법을 통해 더욱 robust한 모델 구축이 가능합니다.

---
*분석 코드: `src_new/analysis/multicollinearity_analysis_improved.py`* 