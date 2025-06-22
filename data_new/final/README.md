# 📊 Final - 최종 모델링 데이터셋

이 디렉토리는 한국 기업 부실예측 모델링을 위한 **최종 완성된 데이터셋**을 포함합니다.

## 🎯 데이터셋 특징

- **100% Complete**: 결측치가 전혀 없는 완전한 데이터
- **12년간 데이터**: 2012-2023년 한국 상장기업 재무정보
- **16,197개 관측치**: 충분한 샘플 사이즈 확보
- **18개 재무지표**: 주요 재무비율 및 시장지표 포함

## 📁 주요 파일 설명

### 🎯 핵심 데이터셋
- **`FS_100_complete.csv`** (6.7MB, 16,197 × 21)
  - 완전한 재무 데이터셋 (회사명, 거래소코드, 회계년도 포함)
  - 18개 재무지표 + default 라벨
  - 다중공선성 분석 완료 (K2_Score_Original 제거 권장)

### 📋 메타데이터
- **`dataset_info_100_complete.json`**
  - 데이터셋 상세 정보 및 통계
  - 분할 정보, SMOTE 설정, 파일 구조 등

### 🎲 모델링용 분할 데이터

#### Normal Dataset (원본 비율 유지)
- **`X_train_100_normal.csv`** (9,718 × 18): 훈련용 특성
- **`X_valid_100_normal.csv`** (3,239 × 18): 검증용 특성  
- **`X_test_100_normal.csv`** (3,240 × 18): 테스트용 특성
- **`y_train_100_normal.csv`** (9,718 × 1): 훈련용 라벨
- **`y_valid_100_normal.csv`** (3,239 × 1): 검증용 라벨
- **`y_test_100_normal.csv`** (3,240 × 1): 테스트용 라벨

#### SMOTE Dataset (불균형 해결)
- **`X_train_100_smote.csv`** (10,621 × 18): SMOTE 적용 훈련 특성
- **`X_valid_100_smote.csv`** (3,239 × 18): 검증용 특성
- **`X_test_100_smote.csv`** (3,240 × 18): 테스트용 특성
- **`y_train_100_smote.csv`** (10,621 × 1): SMOTE 적용 훈련 라벨
- **`y_valid_100_smote.csv`** (3,239 × 1): 검증용 라벨
- **`y_test_100_smote.csv`** (3,240 × 1): 테스트용 라벨

### 📊 메타데이터 파일
- **`meta_train_100.csv`**: 훈련 데이터 메타정보 (회사명, 코드, 연도)
- **`meta_valid_100.csv`**: 검증 데이터 메타정보
- **`meta_test_100.csv`**: 테스트 데이터 메타정보

### 🔧 전처리 파일
- **`FS_ratio_flow_labeled.csv`**: 라벨링된 재무비율 데이터
- **`FS_ratio_flow_with_scores.csv`**: 스코어 포함 데이터
- **`FS_ratio_flow.csv`**: 기본 재무비율 플로우 데이터
- **`FS.csv`**: 원본 재무제표 데이터

## 📈 데이터 상세 정보

### 🏢 기업 현황
- **총 관측치**: 16,197개
- **정상 기업**: 16,093개 (99.36%)
- **부실 기업**: 104개 (0.64%)
- **관측 기간**: 2012년 12월 ~ 2023년 12월

### 📊 데이터 분할 (6:2:2)
- **훈련**: 9,718개 (60%)
- **검증**: 3,239개 (20%)  
- **테스트**: 3,240개 (20%)

### ⚖️ SMOTE 적용
- **원본 훈련 샘플**: 9,718개
- **SMOTE 후 샘플**: 10,621개
- **목표 비율**: 10% (부실기업)
- **기법**: BorderlineSMOTE

## 🎯 재무지표 설명

| 변수명 | 설명 | 분류 |
|--------|------|------|
| **ROA** | Return on Assets (총자산수익률) | 수익성 |
| **TLTA** | Total Liabilities to Total Assets (총부채비율) | 안전성 |
| **WC_TA** | Working Capital to Total Assets (운전자본비율) | 유동성 |
| **CFO_TD** | Cash Flow from Operations to Total Debt | 현금흐름 |
| **SIGMA** | Stock Return Volatility (주가변동성) | 위험성 |
| **RE_TA** | Retained Earnings to Total Assets | 성장성 |
| **EBIT_TA** | EBIT to Total Assets | 수익성 |
| **MVE_TL** | Market Value of Equity to Total Liabilities | 시장평가 |
| **S_TA** | Sales to Total Assets (총자산회전율) | 활동성 |
| **CLCA** | Current Liabilities to Current Assets | 유동성 |
| **OENEG** | Operating Earnings Negative (영업이익 음수 여부) | 수익성 |
| **CR** | Current Ratio (유동비율) | 유동성 |
| **CFO_TA** | Cash Flow from Operations to Total Assets | 현금흐름 |
| **TLMTA** | Total Liabilities to Market Value of Assets | 레버리지 |
| **RET_3M** | 3-Month Stock Return (3개월 수익률) | 시장성과 |
| **RET_9M** | 9-Month Stock Return (9개월 수익률) | 시장성과 |
| **MB** | Market-to-Book Ratio (시장가/장부가) | 밸류에이션 |
| ~~**K2_Score_Original**~~ | ~~원본 K2 Score~~ | ⚠️ **제거 권장** |

## ⚠️ 다중공선성 주의사항

### 🚨 문제 변수
- **K2_Score_Original**: 완전한 다중공선성 (VIF = ∞)
- **해결방안**: 해당 변수 제거 후 모델링 수행

### ✅ 정리 후 상태
- **K2_Score_Original 제거** → 17개 변수로 모델링
- **모든 VIF < 5** 달성 (평균 VIF: 2.34)
- **WC_TA ↔ CLCA**: 높은 상관관계 (-0.823) 주의

## 🔧 사용 방법

### 📖 데이터 로드
```python
import pandas as pd

# 완전한 데이터셋 로드
df = pd.read_csv('FS_100_complete.csv', encoding='utf-8')

# 모델링용 데이터 로드 (Normal)
X_train = pd.read_csv('X_train_100_normal.csv')
y_train = pd.read_csv('y_train_100_normal.csv')['default']

# 모델링용 데이터 로드 (SMOTE)
X_train_smote = pd.read_csv('X_train_100_smote.csv')
y_train_smote = pd.read_csv('y_train_100_smote.csv')['default']
```

### 🎯 다중공선성 해결
```python
# K2_Score_Original 제거 (권장)
feature_columns = [col for col in df.columns 
                  if col not in ['회사명', '거래소코드', '회계년도', 'default', 'K2_Score_Original']]

X_clean = df[feature_columns]
```

### 📊 메타데이터 확인
```python
import json

# 데이터셋 정보 로드
with open('dataset_info_100_complete.json', 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)
    
print(f"총 샘플 수: {dataset_info['original_data']['total_samples']}")
print(f"부실 기업 비율: {dataset_info['original_data']['default_ratio']:.2%}")
```

## 📋 품질 보증

### ✅ 데이터 검증 완료
- [x] 결측치 0개 확인
- [x] 중복 레코드 제거
- [x] 이상치 탐지 및 처리
- [x] 다중공선성 진단 완료
- [x] 시계열 정렬 확인

### 📊 통계적 검증
- **평균**: 모든 변수 정상 범위
- **분산**: 안정적 분포 확인
- **왜도/첨도**: 허용 범위 내
- **상관관계**: 다중공선성 해결

## 🚀 모델링 권장사항

### 🎯 변수 선택
1. **K2_Score_Original 제거** (필수)
2. **WC_TA vs CLCA 중 선택** (상관관계 -0.823)
3. **도메인 지식 기반 선택**

### 📈 모델링 전략
- **Normal Dataset**: 원본 분포 유지
- **SMOTE Dataset**: 불균형 해결 (F1-Score 향상)
- **교차검증**: Time-series split 권장
- **평가지표**: AUC, F1-Score, Precision, Recall

### 🔧 전처리 완료 사항
- **표준화**: StandardScaler 적용
- **분할**: Stratified split (층화추출)
- **라벨링**: default (0: 정상, 1: 부실)

## 📞 문의 및 지원

**데이터 관련 문의**: 프로젝트 메인 README.md 참조  
**품질 이슈**: GitHub Issues에 보고  
**개선 제안**: Pull Request 환영

---

**⚡ 빠른 시작**: `src_new/modeling/` 디렉토리의 모델링 스크립트 참조  
**📊 분석 결과**: `outputs/` 디렉토리에서 확인  
**🎨 시각화**: `dashboard/` 디렉토리의 Streamlit 앱 실행

---

*마지막 업데이트: 2025년 6월 22일 - 다중공선성 분석 및 K2 Score 이슈 해결*
