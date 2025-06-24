# 🏢 한국 기업 부실예측 프로젝트 대시보드

## 📋 개요

한국 기업 부실예측 모델링 프로젝트의 **간결한 워크플로우 중심 대시보드**입니다. 복잡한 ML 파이프라인을 5단계로 나누어 실제 코드와 함께 설명합니다.

## 🚀 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 대시보드 실행
streamlit run project_workflow_dashboard.py

# 3. 브라우저에서 확인
# http://localhost:8501
```

## 🔄 워크플로우 구성

### 🎯 프로젝트 개요
- 프로젝트 목표 및 데이터 현황
- 두 가지 데이터 트랙 비교
- 최고 성능 지표 요약

### 📊 1단계: 원본 데이터
- **DART 재무제표**: 2012-2023년 12개 파일
- **주가 데이터**: 일별 시장 데이터
- **실제 코드**: 데이터 로딩 및 매칭 코드

```python
# 핵심 코드 예시
def load_financial_data():
    file_pattern = "data/raw/20*.csv"
    files = glob.glob(file_pattern)
    # ... 실제 구현
```

### 🔧 2단계: 데이터 전처리
- **데이터 정제**: 결측치, 이상치, 중복 제거
- **코드 정규화**: 거래소코드 6자리 통일
- **품질 검증**: 25,847개 → 22,780개 관측치

```python
# 핵심 코드 예시
def preprocess_financial_data(df):
    # 필수 컬럼 결측치 제거
    essential_cols = ['매출액', '총자산', '총부채']
    df = df.dropna(subset=essential_cols)
    # ... 실제 구현
```

### 📈 3단계: EDA & 특성공학
- **탐색적 분석**: 부실기업 분포, 재무지표 비교
- **재무비율 계산**: 17개 핵심 지표
- **부실 라벨링**: 상장폐지 1년 전 = 부실

```python
# 핵심 코드 예시
def calculate_financial_ratios(df):
    ratios_df['ROA'] = df['순이익'] / df['총자산']
    ratios_df['TLTA'] = df['총부채'] / df['총자산']
    # ... 17개 지표 계산
```

### 🤖 4단계: 모델링
- **개별 모델**: Logistic, RandomForest, XGBoost
- **앙상블**: 9개 모델 균등 가중치
- **성능**: F1-Score 0.4096 (21.3% 향상)

```python
# 핵심 코드 예시
class EnsembleModel:
    def predict_proba(self, X):
        probas = [model.predict_proba(X)[:, 1] for model in self.models]
        return np.average(probas, weights=self.weights)
```

### 🏆 5단계: 결과 분석
- **성능 요약**: F1-Score, AUC, Precision, Recall
- **특성 중요도**: ROA, MVE_TL, EBIT_TA 상위
- **실무 활용**: 금융기관, 투자전략, 연구 활용

## 📊 주요 성과

| 지표 | 값 | 설명 |
|------|----|----|
| **F1-Score** | 0.4096 | 최고 성능 (앙상블) |
| **AUC** | 0.9808 | 거의 완벽한 분류 |
| **데이터** | 22,780개 | 기업-연도 관측치 |
| **부실기업** | 132개 | 0.58% 불균형 |

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn, XGBoost, Imbalanced-learn

## 📁 파일 구조

```
dashboard/
├── project_workflow_dashboard.py  # 메인 대시보드
├── requirements.txt               # 의존성 목록
└── README.md                     # 이 파일
```

## 🎯 특징

### ✅ 간결함
- 복잡한 ML 파이프라인을 5단계로 단순화
- 각 단계별 핵심 내용만 집중 설명

### 💻 실용성
- 모든 단계에 **실제 코드** 포함
- 복사-붙여넣기 가능한 코드 스니펫

### 📊 시각화
- Plotly 기반 인터랙티브 차트
- 성능 비교, 데이터 분포 등 직관적 표현

### 🔄 워크플로우 중심
- 순차적 단계별 진행
- 각 단계의 입력/출력 명확히 표시

## 📧 문의

- **GitHub**: Issues 탭 활용
- **목적**: 교육 및 연구용
- **라이선스**: 상업적 사용 시 관련 법규 준수 필요 