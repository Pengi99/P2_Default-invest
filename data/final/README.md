# 📊 Final - 최종 모델링 데이터셋

이 디렉토리는 한국 기업 부실예측 모델링을 위한 **최종 완성된 데이터셋**을 포함합니다.

## 🎯 데이터셋 특징

- **100% Complete**: 결측치가 전혀 없는 완전한 데이터
- **12년간 데이터**: 2012-2023년 한국 상장기업 재무정보
- **16,197개 관측치**: 충분한 샘플 사이즈 확보
- **17개 재무지표**: 주요 재무비율 및 시장지표 포함 (다중공선성 해결)

## 📁 주요 파일 설명

### 🎯 핵심 데이터셋
- **`FS_100_complete.csv`** (6.7MB, 16,197 × 21)
  - 완전한 재무 데이터셋 (회사명, 거래소코드, 회계년도 포함)
  - 17개 재무지표 + default 라벨
  - 다중공선성 분석 완료 (K2_Score_Original 제거됨)

### 📋 메타데이터
- **`dataset_info_100_complete.json`**
  - 데이터셋 상세 정보 및 통계
  - 분할 정보, 동적 SMOTE 설정, 파일 구조 등

### 🎲 모델링용 분할 데이터

#### 원본 데이터 (Data Leakage 방지)
- **`X_train_100_normal.csv`** (2.1MB, 6,478 × 17): 훈련용 특성
- **`X_valid_100_normal.csv`** (1.6MB, 4,859 × 17): 검증용 특성  
- **`X_test_100_normal.csv`** (1.6MB, 4,860 × 17): 테스트용 특성
- **`y_train_100_normal.csv`** (13KB, 6,478 × 1): 훈련용 라벨
- **`y_valid_100_normal.csv`** (9.5KB, 4,859 × 1): 검증용 라벨
- **`y_test_100_normal.csv`** (9.5KB, 4,860 × 1): 테스트용 라벨

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

### 📊 데이터 분할 (4:3:3)
- **훈련**: 6,478개 (40%)
- **검증**: 4,859개 (30%)  
- **테스트**: 4,860개 (30%)

### ⚖️ **동적 SMOTE 적용 (Data Leakage 방지)**
- **방식**: Cross-Validation 내부에서 동적 적용
- **목표 비율**: 10% (부실:정상 = 1:10)
- **기법**: BorderlineSMOTE
- **적용 시점**: 각 CV fold 및 최종 훈련 시
- **검증 방식**: 원본 데이터로만 검증 수행

## 🔧 **Data Leakage 방지 메커니즘**

### ❌ 기존 문제점
- 미리 생성된 SMOTE 데이터로 Cross-Validation 수행
- 합성 데이터 간 오염으로 과도하게 낙관적인 성능

### ✅ 해결 방안
- **각 CV fold마다 SMOTE 별도 적용**
- **원본 데이터로만 검증 수행**
- **정확한 일반화 성능 평가**

```python
# 올바른 SMOTE 적용 방식
def proper_cv_with_smote(model, X, y, cv_folds=5):
    for train_idx, val_idx in skf.split(X, y):
        # 각 fold마다 별도 SMOTE 적용
        X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
        # 원본 데이터로 검증
        score = evaluate_on_original_validation_fold()
```

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

## ✅ 다중공선성 해결 완료

### 🎯 해결된 문제
- **K2_Score_Original 완전 제거**: VIF = ∞ 문제 해결
- **최종 17개 변수**: 모든 VIF < 5 달성
- **평균 VIF**: 2.34 (양호한 수준)

### ⚠️ 주의사항
- **WC_TA ↔ CLCA**: 높은 상관관계 (-0.823) 주의
- **도메인 지식 기반 변수 선택 권장**

## 🔧 사용 방법

### 📖 데이터 로드
```python
import pandas as pd

# 완전한 데이터셋 로드
df = pd.read_csv('FS_100_complete.csv', encoding='utf-8')

# 모델링용 데이터 로드
X_train = pd.read_csv('X_train_100_normal.csv')
X_valid = pd.read_csv('X_valid_100_normal.csv')
X_test = pd.read_csv('X_test_100_normal.csv')

y_train = pd.read_csv('y_train_100_normal.csv').iloc[:, 0]
y_valid = pd.read_csv('y_valid_100_normal.csv').iloc[:, 0]
y_test = pd.read_csv('y_test_100_normal.csv').iloc[:, 0]
```

### 🎯 올바른 SMOTE 적용
```python
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import StratifiedKFold

# ❌ 잘못된 방법 (Data Leakage)
# smote = BorderlineSMOTE()
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)

# ✅ 올바른 방법 (Data Leakage 방지)
def proper_cv_with_smote(model, X, y, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 각 fold마다 별도 SMOTE 적용
        smote = BorderlineSMOTE(sampling_strategy=0.1, random_state=42)
        X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
        
        # 모델 훈련 및 원본 데이터로 검증
        model.fit(X_fold_train_smote, y_fold_train_smote)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_pred_proba)
        scores.append(score)
    
    return np.array(scores)
```

### 📊 메타데이터 확인
```python
import json

# 데이터셋 정보 로드
with open('dataset_info_100_complete.json', 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)
    
print(f"총 샘플 수: {dataset_info['original_data']['total_samples']}")
print(f"부실 기업 비율: {dataset_info['original_data']['default_ratio']:.2%}")
print(f"SMOTE 방식: {dataset_info['smote_strategy']['method']}")
```

## 📋 품질 보증

### ✅ 데이터 검증 완료
- [x] 결측치 0개 확인
- [x] 중복 레코드 제거
- [x] 이상치 탐지 및 처리
- [x] 다중공선성 완전 해결
- [x] 시계열 정렬 확인
- [x] **Data Leakage 방지 메커니즘 구현**

### 📊 통계적 검증
- **평균**: 모든 변수 정상 범위
- **분산**: 안정적 분포 확인
- **왜도/첨도**: 허용 범위 내
- **상관관계**: 다중공선성 해결
- **VIF**: 모든 변수 < 5

## 🚀 모델링 권장사항

### 🎯 변수 선택
1. **17개 변수 모두 사용 가능** (다중공선성 해결)
2. **WC_TA vs CLCA 중 선택 고려** (상관관계 -0.823)
3. **도메인 지식 기반 추가 선택**

### 📈 모델링 전략
- **동적 SMOTE**: CV 내부에서 적용 (Data Leakage 방지)
- **교차검증**: Proper CV with SMOTE 사용
- **평가지표**: AUC, F1-Score, Precision, Recall
- **Threshold 최적화**: Validation set 기반

### 🔧 전처리 완료 사항
- **표준화**: StandardScaler 적용
- **분할**: Stratified split (층화추출)
- **라벨링**: default (0: 정상, 1: 부실)
- **Data Leakage 방지**: 동적 SMOTE 구현

## 📞 문의 및 지원

**데이터 관련 문의**: 프로젝트 메인 README.md 참조  
**품질 이슈**: GitHub Issues에 보고  
**개선 제안**: Pull Request 환영

---

**⚡ 빠른 시작**: `src_new/modeling/master_model_runner.py` 실행  
**📊 분석 결과**: `outputs/master_runs/` 디렉토리에서 확인  
**🎨 시각화**: `dashboard/` 디렉토리의 Streamlit 앱 실행

---

*마지막 업데이트: 2025년 6월 23일 - Data Leakage 방지를 위한 동적 SMOTE 구현*
