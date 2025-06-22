# 📊 한국 기업 부실예측 모델링 대시보드

## 🎯 개요
한국 기업 부실예측 모델링 프로젝트의 **전체 분석 과정과 결과**를 시각적으로 설명하는 인터랙티브 대시보드입니다.

### 🌟 주요 특징
- **포괄적 분석 결과**: 다중공선성 분석, 모델 성능, 특성 중요도 등
- **실시간 코드 리뷰**: 핵심 함수들의 실제 구현 코드 표시
- **인터랙티브 시각화**: Plotly 기반 동적 차트 및 그래프
- **완전한 한국어 지원**: 발표 및 설명에 최적화

## 🚀 실행 방법

### 1. 환경 준비
```bash
cd dashboard
pip install -r requirements.txt
```

### 2. 대시보드 실행
```bash
streamlit run code_review_dashboard.py
```

### 3. 브라우저 접속
자동으로 브라우저가 열리며, `http://localhost:8501`에서 대시보드를 확인할 수 있습니다.

## 📋 대시보드 구성

### 🏠 프로젝트 개요
- **프로젝트 목표 및 배경**: 한국 기업 부실예측의 중요성
- **워크플로우 시각화**: 데이터 수집부터 모델 배포까지
- **주요 성과 지표**: 모델 성능, 데이터 현황 등
- **기술 스택 소개**: Python, ML 라이브러리, 분석 도구

### 🏗️ 코드베이스 구조  
- **src_new 디렉토리**: 최신 소스코드 구조 설명
  - `analysis/`: 다중공선성 분석, Altman Z-Score 등
  - `data_processing/`: 재무비율 계산, 전처리 파이프라인
  - `modeling/`: 머신러닝 모델링 (LR, RF, XGBoost)
  - `feature_engineering/`: 특성 엔지니어링
- **데이터 플로우 다이어그램**: 처리 단계별 시각화
- **핵심 모듈 상세 설명**: 각 모듈의 역할과 기능

### 📁 데이터 파이프라인
- **원본 데이터**: 2012-2023년 DART 재무제표 (16,197개 관측치)
- **전처리 과정**: 
  - 결측치 처리 (100% Complete 데이터 사용)
  - 이상치 탐지 및 제거
  - 재무비율 계산 (18개 주요 지표)
- **특성 엔지니어링**: 시장기반 지표, 변동성 측정
- **최종 데이터 준비**: Train/Valid/Test 분할, SMOTE 적용

### 🔧 핵심 분석 기능

#### 📊 다중공선성 분석
- **VIF(Variance Inflation Factor)** 계산 및 해석
- **상관관계 분석**: 히트맵으로 변수 간 관계 시각화
- **문제 변수 식별**: K2_Score_Original의 완전한 다중공선성 발견
- **해결 방안**: 반복적 VIF 계산을 통한 변수 제거

#### 🎯 모델 성능 평가
- **3가지 알고리즘 비교**: Logistic Regression, Random Forest, XGBoost
- **SMOTE 효과 분석**: 불균형 데이터 처리 전후 비교
- **성능 지표**: AUC, F1-Score, Precision, Recall 등
- **특성 중요도**: 각 모델별 변수 기여도 분석

#### 📈 고급 분석 도구
- **PCA 분석**: 차원 축소 가능성 평가 (27.8% 축소 가능)
- **Altman Z-Score**: 부실예측 스코어 성능 분석
- **스케일링 전략**: 표준화 필요성 분석 및 적용

### 📈 분석 결과 현황

#### 모델 성능 요약
| 모델 | Normal Dataset | SMOTE Dataset |
|-----|----------------|---------------|
| **Logistic Regression** | AUC: 0.943, F1: 0.244 | AUC: 0.928, F1: 0.271 |
| **Random Forest** | AUC: 0.987, F1: 0.583 | AUC: 0.986, F1: 0.608 |
| **XGBoost** | AUC: 0.985, F1: 0.538 | AUC: 0.982, F1: 0.569 |

#### 데이터 현황
- **총 관측치**: 16,197개 (12년간)
- **부실기업**: 104개 (0.64%)
- **특성 개수**: 18개 (재무비율 + 시장지표)
- **데이터 품질**: 100% Complete (결측치 없음)

### 🎯 모델링 인사이트
- **최고 성능**: Random Forest (SMOTE) - F1: 0.608
- **안정성**: XGBoost 모델의 일관된 성능
- **특성 중요도**: ROA, MVE_TL, EBIT_TA 상위 랭크
- **다중공선성 해결**: K2 Score 제거로 VIF < 5 달성

## 🎨 주요 시각화 기능

### 📊 인터랙티브 차트
- **ROC 곡선**: 모델별 성능 비교
- **Precision-Recall 곡선**: 불균형 데이터 성능 평가
- **특성 중요도 차트**: 변수별 기여도 시각화
- **상관계수 히트맵**: 변수 간 관계 매트릭스

### 📈 분석 결과 시각화
- **VIF 분석 차트**: 다중공선성 진단 결과
- **PCA 설명 분산**: 차원 축소 효과 분석
- **모델 성능 비교**: 알고리즘별 지표 비교
- **데이터 분포**: 부실/정상 기업 특성 분포

### 🎯 코드 리뷰 기능
- **핵심 함수 하이라이팅**: 실제 구현 코드 표시
- **알고리즘 설명**: 각 단계별 처리 로직
- **매개변수 설명**: 하이퍼파라미터 및 설정값
- **결과 해석**: 비즈니스 관점에서의 의미

## 📊 데이터 요구사항

대시보드 완전 작동을 위한 필수 파일:
- `data_new/final/dataset_info_100_complete.json`: 데이터셋 메타정보
- `data_new/final/FS_100_complete.csv`: 완전한 재무데이터
- `outputs/reports/100_complete_*.json`: 모델 성능 결과
- `outputs/analysis/comprehensive_multicollinearity_analysis_*.json`: 다중공선성 분석

**파일 없음 대응**: 예시 데이터로 자동 대체하여 시각화 표시

## 🛠️ 커스터마이징 가이드

### 새로운 분석 추가
```python
# dashboard/code_review_dashboard.py에 새 섹션 추가
def new_analysis_section():
    st.header("🔍 새로운 분석")
    # 분석 코드 및 시각화
```

### 시각화 스타일 변경
```python
# Plotly 차트 테마 설정
config = {
    'displayModeBar': False,
    'staticPlot': False,
    'theme': 'plotly_white'  # 또는 'plotly_dark'
}
```

### 데이터 소스 변경
```python
# 데이터 로드 함수 수정
@st.cache_data
def load_updated_data():
    # 새로운 데이터 소스 연결
    pass
```

## 📱 발표 가이드

### 🎤 권장 발표 순서
1. **프로젝트 개요** (5분)
   - 부실예측의 중요성 및 프로젝트 목표
   - 데이터 현황 및 모델링 접근법

2. **코드베이스 구조** (5분)
   - src_new 디렉토리 구조 설명
   - 모듈별 역할과 책임 분담

3. **데이터 파이프라인** (10분)
   - 원본 데이터부터 최종 모델링 데이터까지
   - 전처리 과정 및 품질 관리

4. **핵심 분석 결과** (15분)
   - 다중공선성 분석 및 해결
   - 모델 성능 비교 및 특성 중요도
   - SMOTE 효과 분석

5. **결론 및 향후 계획** (5분)
   - 주요 성과 및 한계점
   - 실무 적용 방안 및 개선 과제

### 🎯 발표 팁
- **인터랙티브 시연**: 실시간으로 차트 조작하며 설명
- **코드 리뷰**: 핵심 로직을 직접 보여주며 기술적 깊이 어필
- **비즈니스 관점**: 재무 도메인 지식과 연결하여 실용성 강조
- **Q&A 준비**: 다중공선성, SMOTE, 모델 선택 근거 등 예상 질문 대비

## 🔧 문제 해결

### 한글 폰트 문제
```python
# macOS
plt.rcParams['font.family'] = 'AppleGothic'

# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'

# Linux
plt.rcParams['font.family'] = 'DejaVu Sans'
```

### 메모리 부족 문제
```bash
# 대용량 데이터 처리 시
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
streamlit run code_review_dashboard.py --server.maxUploadSize=1000
```

### 포트 충돌 해결
```bash
# 다른 포트 사용
streamlit run code_review_dashboard.py --server.port 8502
```

### 캐시 초기화
```bash
# Streamlit 캐시 클리어
streamlit cache clear
```

## 📊 성능 최적화

- **@st.cache_data**: 데이터 로딩 캐시 적용
- **lazy loading**: 필요한 섹션만 로드
- **이미지 압축**: PNG 파일 최적화
- **청크 처리**: 대용량 데이터 분할 처리

## 🤖 자동화 기능

### 실시간 업데이트
- 새로운 분석 결과 자동 반영
- 모델 재학습 시 성능 지표 업데이트
- 데이터 추가 시 통계 정보 갱신

### 보고서 생성
```python
# 분석 결과 자동 보고서 생성
def generate_report():
    report = create_analysis_report()
    st.download_button("📄 보고서 다운로드", report, "analysis_report.html")
```

## 📞 지원 및 문의

- **기술 지원**: 프로젝트 메인 README.md 참조
- **버그 리포트**: GitHub Issues 활용
- **기능 제안**: Pull Request 환영

---

**💡 팁**: 발표 전 `streamlit run code_review_dashboard.py --server.headless true`로 미리 테스트해보세요!

---

*이 대시보드는 한국 금융시장 분석을 위한 교육 및 연구 목적으로 제작되었습니다.* 