# Feature Engineering 디렉토리

이 디렉토리는 한국 기업 부실예측을 위한 **특성 엔지니어링** 관련 스크립트들을 포함합니다.

## 📁 파일 구조

### 🔧 특성 생성 스크립트
- `add_financial_variables.py` - 추가 재무변수 생성 및 통합

### 📊 데이터 처리 현황

#### ✅ 완료된 작업
1. **재무비율 계산** - 17개 핵심 재무지표 생성
2. **결측치 처리** - 100% 완성도 달성
3. **다중공선성 해결** - K2_Score_Original 제거
4. **데이터 분할** - 4:3:3 비율 (Train:Valid:Test)
5. **표준화** - StandardScaler 적용
6. **라벨링** - 부실기업 분류 (0: 정상, 1: 부실)

#### 🔄 동적 처리 (런타임)
- **SMOTE 적용** - Cross-Validation 내부에서 동적 적용 (Data Leakage 방지)

## 🎯 핵심 특성

### 📈 생성된 재무지표 (17개)
| 분류 | 지표 | 개수 |
|------|------|------|
| **수익성** | ROA, EBIT_TA, OENEG | 3개 |
| **안전성** | TLTA, TLMTA | 2개 |
| **유동성** | WC_TA, CLCA, CR, CFO_TA | 4개 |
| **활동성** | S_TA | 1개 |
| **성장성** | RE_TA | 1개 |
| **현금흐름** | CFO_TD | 1개 |
| **시장평가** | MVE_TL, RET_3M, RET_9M, MB | 4개 |
| **위험성** | SIGMA | 1개 |

### 🔧 Data Leakage 방지 메커니즘
- **동적 SMOTE**: CV 내부에서 각 fold마다 별도 적용
- **원본 데이터 검증**: 합성 데이터 오염 방지
- **정확한 성능 평가**: 실제 일반화 능력 측정

## 🚀 사용법

### 📖 특성 데이터 로드
```python
import pandas as pd

# 최종 완성된 데이터 로드
X_train = pd.read_csv('data_new/final/X_train_100_normal.csv')
X_valid = pd.read_csv('data_new/final/X_valid_100_normal.csv')
X_test = pd.read_csv('data_new/final/X_test_100_normal.csv')

y_train = pd.read_csv('data_new/final/y_train_100_normal.csv').iloc[:, 0]
y_valid = pd.read_csv('data_new/final/y_valid_100_normal.csv').iloc[:, 0]
y_test = pd.read_csv('data_new/final/y_test_100_normal.csv').iloc[:, 0]

print(f"특성 개수: {X_train.shape[1]}")
print(f"훈련 샘플: {len(X_train):,}개")
print(f"부실 비율: {y_train.mean():.2%}")
```

### 🎯 올바른 SMOTE 적용
```python
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def proper_cv_with_smote(model, X, y, cv_folds=5):
    """Data Leakage 방지를 위한 올바른 CV with SMOTE"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        # 각 fold마다 별도로 분할
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 훈련 fold에만 SMOTE 적용
        smote = BorderlineSMOTE(sampling_strategy=0.1, random_state=42)
        X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
        
        # 모델 훈련 및 검증 (원본 데이터로만)
        model.fit(X_fold_train_smote, y_fold_train_smote)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_pred_proba)
        scores.append(score)
    
    return np.array(scores)

# 사용 예시
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
cv_scores = proper_cv_with_smote(model, X_train, y_train)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

## 📊 특성 품질 보증

### ✅ 검증 완료 사항
- [x] **결측치 0개**: 100% 완성도
- [x] **다중공선성 해결**: 모든 VIF < 5
- [x] **이상치 처리**: 통계적 검증 완료
- [x] **정규화**: StandardScaler 적용
- [x] **Data Leakage 방지**: 동적 SMOTE 구현

### 📈 특성 통계
- **평균 VIF**: 2.34 (양호)
- **최대 상관관계**: 0.823 (WC_TA ↔ CLCA)
- **특성 개수**: 17개 (최적화 완료)
- **샘플 수**: 16,197개 (충분한 크기)

## 🔍 주요 개선사항

### 🚨 해결된 문제들
1. **K2_Score_Original 제거**: VIF = ∞ 문제 해결
2. **Data Leakage 방지**: SMOTE CV 내부 적용
3. **과적합 방지**: 원본 데이터 검증
4. **성능 최적화**: 각 모델별 threshold 최적화

### 🎯 핵심 특징
- **도메인 기반**: 금융 전문가 지식 반영
- **통계적 검증**: 엄격한 품질 관리
- **재현 가능성**: 완전한 버전 관리
- **확장 가능성**: 추가 특성 생성 용이

## 💡 향후 개선 방향

### 🔄 추가 특성 후보
1. **시계열 특성**: 추세, 계절성 반영
2. **거시경제 지표**: GDP, 금리, 환율 등
3. **산업별 특성**: 업종 더미 변수
4. **텍스트 특성**: 뉴스, 공시 감성 분석

### 📈 고급 기법 적용
- **특성 선택**: Lasso, RFE, Mutual Information
- **특성 변환**: PCA, ICA, Polynomial Features
- **특성 상호작용**: Cross-product, Ratios
- **앙상블 특성**: 모델 기반 특성 생성

## 📞 문의 및 지원

**특성 관련 문의**: 프로젝트 메인 README.md 참조  
**개선 제안**: GitHub Issues에 등록  
**기술 지원**: Pull Request 환영

---

**⚡ 빠른 시작**: `src_new/modeling/master_model_runner.py` 실행  
**📊 결과 확인**: `outputs/master_runs/` 디렉토리  
**🎨 시각화**: `dashboard/` Streamlit 앱

---

*마지막 업데이트: 2025년 6월 23일 - Data Leakage 방지 동적 SMOTE 구현*
