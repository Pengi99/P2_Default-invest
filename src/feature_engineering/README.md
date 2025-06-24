# 🔧 Feature Engineering - 특성 공학 모듈

이 디렉토리는 한국 기업 부실예측 모델링을 위한 **특성 공학 및 변수 생성** 모듈을 포함합니다.

## 🎯 주요 기능

- **재무변수 추가**: 고급 재무지표 계산 및 파생변수 생성
- **성장률 보정**: 2011년 기준 데이터를 활용한 정확한 성장률 계산
- **시계열 특성**: Look-ahead Bias 방지를 위한 시계열 데이터 처리
- **도메인 지식 적용**: 한국 회계기준(K-IFRS) 기반 변수 설계

## 📁 파일 구조

```
src/feature_engineering/
├── 📄 add_financial_variables.py      # 재무변수 추가 및 파생변수 생성
├── 📄 fix_growth_rates_with_2011.py   # 2011년 기준 성장률 보정
└── 📄 README.md                       # 현재 파일
```

## 🔧 핵심 스크립트

### 📊 add_financial_variables.py
**재무변수 추가 및 고급 지표 계산**

**주요 기능:**
- **고급 재무비율**: 기본 비율을 활용한 복합 지표 계산
- **업종별 정규화**: 업종 특성을 고려한 변수 표준화
- **시장기반 지표**: 주가와 재무데이터 결합 변수
- **리스크 지표**: 변동성 및 안정성 측정 변수

**생성 변수 예시:**
- **수익성 복합지표**: ROA_EBIT_Combined, ROE_Adjusted
- **안정성 지표**: Debt_to_Equity_Modified, Interest_Coverage
- **효율성 지표**: Asset_Turnover_Enhanced, Working_Capital_Efficiency
- **성장성 지표**: Revenue_Growth_3Y, Asset_Growth_Stable

**입력 데이터:**
- `data/processed/FS_ratio_flow.csv` (기본 재무비율)
- 업종 분류 데이터 (필요시)

**출력 데이터:**
- `data/processed/FS_ratio_flow_enhanced.csv` (확장된 재무비율)

**사용법:**
```bash
python src/feature_engineering/add_financial_variables.py
```

### 📈 fix_growth_rates_with_2011.py
**2011년 기준 성장률 정확한 계산**

**주요 기능:**
- **기준연도 설정**: 2011년 데이터를 기준으로 성장률 계산
- **Look-ahead Bias 방지**: 미래 정보 누출 완전 차단
- **연도별 성장률**: 1년, 2년, 3년 성장률 각각 계산
- **안정성 고려**: 이상치 및 특이값 처리

**계산 성장률:**
- **매출 성장률**: Revenue_Growth_1Y, Revenue_Growth_2Y, Revenue_Growth_3Y
- **자산 성장률**: Asset_Growth_1Y, Asset_Growth_2Y, Asset_Growth_3Y
- **이익 성장률**: Profit_Growth_1Y, Profit_Growth_2Y, Profit_Growth_3Y
- **부채 성장률**: Debt_Growth_1Y, Debt_Growth_2Y, Debt_Growth_3Y

**시계열 처리 방식:**
```python
# 올바른 성장률 계산 (Look-ahead Bias 방지)
def calculate_growth_rate(df, base_year=2011):
    # 2011년 기준값 사용
    base_values = df[df['year'] == base_year]
    
    # 각 연도별 성장률 계산 (미래 정보 사용 금지)
    for year in range(2012, 2024):
        current_values = df[df['year'] == year]
        growth_rate = (current_values - base_values) / base_values
        df.loc[df['year'] == year, 'growth_rate'] = growth_rate
```

**입력 데이터:**
- `data/raw/GAAP_2011.csv`, `data/raw/IFRS_2011.csv` (2011년 기준 데이터)
- `data/processed/FS_ratio_flow_enhanced.csv`

**출력 데이터:**
- `data/processed/FS_ratio_flow_with_growth.csv`

**사용법:**
```bash
python src/feature_engineering/fix_growth_rates_with_2011.py
```

## 🎯 특성 공학 전략

### 1. **재무 도메인 전문성**
- **한국 회계기준 준수**: K-IFRS 기반 변수 설계
- **업종별 특성 고려**: 제조업, 서비스업, 금융업 등 업종별 맞춤 지표
- **규제 환경 반영**: 한국 금융감독원 기준 고려

### 2. **시계열 데이터 특성**
- **Look-ahead Bias 완전 방지**: 미래 정보 절대 사용 금지
- **시계열 정렬**: 연도별 순차적 처리
- **기준연도 설정**: 2011년을 일관된 기준으로 사용

### 3. **통계적 엄밀성**
- **이상치 처리**: Winsorization, IQR 기반 이상치 제거
- **정규화 전략**: 업종 내 표준화, Z-score 정규화
- **결측치 전략**: 도메인 지식 기반 합리적 대체

### 4. **모델링 최적화**
- **다중공선성 방지**: VIF 계산 후 상관관계 높은 변수 제거
- **특성 선택 준비**: Lasso, Random Forest 중요도 활용 준비
- **스케일링 고려**: 변수별 적절한 스케일러 적용

## 📊 생성 변수 카테고리

### 🏆 **수익성 지표 (Profitability)**
| 변수명 | 설명 | 계산 방식 |
|--------|------|-----------|
| **ROA_Adjusted** | 조정 총자산수익률 | (순이익 + 이자비용) / 평균총자산 |
| **ROE_Sustainable** | 지속가능 자기자본수익률 | ROA × Equity_Multiplier × Retention_Rate |
| **Gross_Margin_Trend** | 매출총이익률 추세 | 3년 이동평균 기울기 |
| **Operating_Efficiency** | 영업효율성 | 영업이익 / (매출원가 + 판관비) |

### 🛡️ **안정성 지표 (Stability)**
| 변수명 | 설명 | 계산 방식 |
|--------|------|-----------|
| **Debt_Service_Coverage** | 부채상환능력 | 영업현금흐름 / (이자비용 + 원금상환) |
| **Equity_Buffer** | 자본완충비율 | (자기자본 - 최소자본) / 총자산 |
| **Liquidity_Index** | 유동성지수 | (현금성자산 + 단기투자) / 유동부채 |
| **Financial_Flexibility** | 재무유연성 | 미사용 신용한도 / 총자산 |

### ⚡ **효율성 지표 (Efficiency)**
| 변수명 | 설명 | 계산 방식 |
|--------|------|-----------|
| **Asset_Utilization** | 자산활용도 | 매출 / 평균총자산 |
| **Working_Capital_Cycle** | 운전자본 회전기간 | DSO + DIO - DPO |
| **Capital_Intensity** | 자본집약도 | 유형자산 / 매출 |
| **Productivity_Index** | 생산성지수 | 부가가치 / 종업원수 |

### 📈 **성장성 지표 (Growth)**
| 변수명 | 설명 | 계산 방식 |
|--------|------|-----------|
| **Sustainable_Growth** | 지속가능성장률 | ROE × (1 - 배당성향) |
| **Market_Share_Growth** | 시장점유율 성장 | (매출성장률 - 업종평균성장률) |
| **Innovation_Investment** | 혁신투자율 | (R&D비용 + 광고비) / 매출 |
| **Expansion_Capacity** | 확장능력 | (현금흐름 - 배당) / 자본적지출 |

### 🔍 **리스크 지표 (Risk)**
| 변수명 | 설명 | 계산 방식 |
|--------|------|-----------|
| **Earnings_Volatility** | 이익변동성 | 5년 영업이익 표준편차 / 평균 |
| **Cash_Flow_Stability** | 현금흐름안정성 | 영업현금흐름 / 순이익 |
| **Leverage_Risk** | 레버리지위험 | 부채비율 × 이자율변동성 |
| **Industry_Beta** | 업종베타 | 개별기업수익률 / 업종평균수익률 |

## 🚀 사용 방법

### 1. **순차적 실행 (권장)**
```bash
# 1단계: 재무변수 추가
python src/feature_engineering/add_financial_variables.py

# 2단계: 성장률 보정
python src/feature_engineering/fix_growth_rates_with_2011.py

# 확인: 결과 데이터 검증
ls data/processed/FS_ratio_flow_*.csv
```

### 2. **개별 실행**
```bash
# 재무변수만 추가
python src/feature_engineering/add_financial_variables.py

# 성장률만 보정
python src/feature_engineering/fix_growth_rates_with_2011.py
```

### 3. **결과 확인**
```python
import pandas as pd

# 원본 데이터
original = pd.read_csv('data/processed/FS_ratio_flow.csv')
print(f"원본 변수 수: {original.shape[1]}")

# 추가 변수 후
enhanced = pd.read_csv('data/processed/FS_ratio_flow_enhanced.csv')
print(f"확장 변수 수: {enhanced.shape[1]}")

# 성장률 추가 후
final = pd.read_csv('data/processed/FS_ratio_flow_with_growth.csv')
print(f"최종 변수 수: {final.shape[1]}")

# 새로 추가된 변수 확인
new_variables = set(final.columns) - set(original.columns)
print(f"새로 추가된 변수: {new_variables}")
```

## ⚠️ 주의사항

### 🛡️ **Look-ahead Bias 방지**
```python
# ❌ 잘못된 방법 (미래 정보 사용)
df['growth_rate'] = df.groupby('company')['revenue'].pct_change()

# ✅ 올바른 방법 (과거 정보만 사용)
def safe_growth_calculation(group):
    # 시계열 순서 확인
    group = group.sort_values('year')
    # 이전 연도 대비 성장률 계산
    group['growth_rate'] = group['revenue'].pct_change()
    return group

df = df.groupby('company').apply(safe_growth_calculation)
```

### 📊 **다중공선성 관리**
- **VIF 계산**: 새로 생성된 변수 간 다중공선성 확인
- **상관관계 분석**: 높은 상관관계(>0.8) 변수 식별
- **도메인 우선**: 비즈니스 해석 가능한 변수 우선 선택

### 🔧 **데이터 품질 관리**
- **이상치 처리**: 99% 분위수 기준 Winsorization
- **결측치 대체**: 업종 중앙값 또는 시계열 보간
- **단위 통일**: 모든 비율 변수 소수점 단위 통일

## 📋 품질 검증

### ✅ **변수 품질 체크리스트**
- [ ] Look-ahead Bias 없음 확인
- [ ] 다중공선성 검사 (VIF < 5)
- [ ] 이상치 처리 완료
- [ ] 결측치 비율 < 10%
- [ ] 비즈니스 해석 가능성
- [ ] 시계열 일관성 확인

### 📊 **통계적 검증**
```python
# VIF 계산
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data

# 상관관계 분석
correlation_matrix = df.corr()
high_corr = correlation_matrix[correlation_matrix.abs() > 0.8]
```

## 🚀 향후 개발 계획

### 📈 **고급 특성 공학**
- [ ] 딥러닝 기반 특성 추출 (Autoencoder)
- [ ] 시계열 특성 (LSTM 기반 hidden states)
- [ ] 그래프 기반 특성 (기업 관계망)
- [ ] 텍스트 특성 (연차보고서 NLP)

### 🔧 **자동화 개선**
- [ ] 특성 생성 파이프라인 자동화
- [ ] 실시간 특성 업데이트 시스템
- [ ] A/B 테스트 기반 특성 검증
- [ ] 특성 중요도 자동 추적

### 📊 **도메인 확장**
- [ ] ESG 지표 통합
- [ ] 거시경제 변수 연동
- [ ] 뉴스 감성 지표
- [ ] 소셜미디어 지표

## 💡 **Best Practices**

### 🎯 **특성 설계 원칙**
1. **비즈니스 해석 가능성**: 실무진이 이해할 수 있는 변수
2. **통계적 유의성**: 타겟과 유의한 관계
3. **안정성**: 시간에 따른 일관된 패턴
4. **확장 가능성**: 새로운 데이터에도 적용 가능

### 📋 **개발 가이드라인**
1. **함수 모듈화**: 재사용 가능한 함수 단위 개발
2. **문서화**: 각 변수의 비즈니스 의미 명확히 기록
3. **테스트**: 단위 테스트로 계산 로직 검증
4. **버전 관리**: 변수 생성 로직 변경 시 버전 기록

---

## 🔗 **연관 모듈**

- **📊 데이터 처리**: [src/data_processing/README.md](../data_processing/README.md)
- **🤖 모델링**: [src/modeling/README.md](../modeling/README.md)
- **📈 분석**: [src/analysis/README.md](../analysis/README.md)
- **📋 전처리**: [src/preprocessing/README.md](../preprocessing/README.md)

---

**모듈 상태**: ✅ **완료**  
**핵심 기능**: 🎯 **재무변수 확장** + **성장률 보정**  
**품질 수준**: 🏆 **Production Ready**  
**최종 업데이트**: 2025-06-24  
**개발팀**: 금융 특성공학 전문팀
