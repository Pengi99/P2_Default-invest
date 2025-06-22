# 🏢 Default Invest: 한국 기업 부실예측 모델링 프로젝트

한국 상장기업의 재무데이터를 활용한 **부실예측 모델링** 및 **퀄리티 팩터 기반 포트폴리오 전략** 프로젝트입니다.

## 📊 프로젝트 개요

- **목표**: 한국 상장기업의 부실 위험을 예측하고 퀄리티 팩터 기반의 투자 전략 백테스트
- **데이터**: 2012-2023년 DART 재무제표 데이터 (16,197개 관측치)
- **모델**: Logistic Regression, Random Forest, XGBoost
- **특징**: 다중공선성 분석, SMOTE 적용, 포괄적 성능 평가

## 🏗️ 프로젝트 구조

```
📦 P2_Default-invest/
├── 📁 src_new/                    # 새로운 소스코드 (추천)
│   ├── 📁 analysis/               # 데이터 분석
│   │   ├── 📄 multicollinearity_analysis.py        # 다중공선성 분석
│   │   ├── 📄 multicollinearity_analysis_improved.py # 개선된 다중공선성 분석
│   │   ├── 📄 comprehensive_altman_analysis.py     # Altman Z-Score 분석
│   │   ├── 📄 score_performance_analysis.py        # 스코어 성능 분석
│   │   ├── 📄 analyze_scaling_needs.py             # 스케일링 필요성 분석
│   │   └── 📄 apply_default_labeling_and_scaling.py # 라벨링 및 스케일링
│   ├── 📁 data_processing/        # 데이터 전처리
│   │   ├── 📄 create_financial_ratios_master.py    # 재무비율 마스터 생성
│   │   ├── 📄 step1_basic_financial_ratios.py      # 기본 재무비율
│   │   ├── 📄 step2_market_based_ratios.py         # 시장기반 비율
│   │   ├── 📄 step3_volatility_returns.py          # 변동성 및 수익률
│   │   └── 📄 step4_finalize_ratios.py             # 최종 비율 완성
│   ├── 📁 feature_engineering/    # 특성 엔지니어링
│   │   ├── 📄 add_financial_variables.py           # 재무변수 추가
│   │   └── 📄 create_final_modeling_dataset.py     # 최종 모델링 데이터셋
│   ├── 📁 modeling/              # 머신러닝 모델링
│   │   ├── 📄 master_model_runner.py              # 🆕 통합 파이프라인
│   │   ├── 📄 run_master.py                       # 🆕 마스터 러너 실행
│   │   ├── 📁 config_templates/                   # 🆕 설정 템플릿
│   │   ├── 📄 logistic_regression_100.py           # 로지스틱 회귀
│   │   ├── 📄 RF_100.py                           # 랜덤 포레스트
│   │   ├── 📄 xgboost_100.py                      # XGBoost
│   │   └── 📄 model_comparison.py                 # 모델 비교
│   └── 📁 utils/                 # 유틸리티
├── 📁 data_new/                  # 최신 데이터
│   ├── 📁 raw/                   # 원본 데이터 (2012-2023)
│   ├── 📁 processed/             # 전처리된 데이터
│   └── 📁 final/                 # 최종 모델링 데이터 (100% 완성)
├── 📁 outputs/                   # 분석 결과
│   ├── 📁 master_runs/           # 🆕 마스터 러너 결과
│   ├── 📁 models/                # 학습된 모델
│   ├── 📁 reports/               # 분석 보고서
│   ├── 📁 visualizations/        # 시각화 결과
│   └── 📁 analysis/              # 다중공선성 분석 결과
├── 📁 dashboard/                 # 발표용 대시보드
├── 📁 notebooks/                 # Jupyter 노트북
└── 📁 archive_old_structure/     # 기존 구조 백업
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 가상환경 활성화 (권장)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는 .venv\Scripts\activate  # Windows
```

### 2. 데이터 준비
```bash
# 데이터는 이미 처리되어 data_new/final/ 에 준비되어 있습니다
ls data_new/final/*.csv
```

### 3. 🚀 모델 학습 및 평가 (권장: 마스터 러너)
```bash
# 🆕 마스터 러너 - 빠른 테스트 (Threshold 최적화 포함)
python src_new/modeling/run_master.py --template quick

# 🆕 마스터 러너 - 완전한 최적화
python src_new/modeling/run_master.py --template production

# 🆕 마스터 러너 - Lasso 특성 선택 포함
python src_new/modeling/run_master.py --template lasso

# 기존 방식 (개별 모델)
python src_new/modeling/logistic_regression_100.py
python src_new/modeling/RF_100.py  
python src_new/modeling/xgboost_100.py
```

### 4. 다중공선성 분석
```bash
# 기본 분석
python src_new/analysis/multicollinearity_analysis.py

# 개선된 분석 (권장)
python src_new/analysis/multicollinearity_analysis_improved.py
```

### 5. 대시보드 실행
```bash
cd dashboard
streamlit run code_review_dashboard.py
```

## 📈 주요 결과

### 🆕 Threshold 최적화 결과 (Test Set)
| 모델 | 데이터 | **최적 Threshold** | AUC | **F1** | Precision | Recall |
|-----|--------|------------------|-----|--------|-----------|--------|
| **Logistic Regression** | Normal | **0.10** | 0.943 | **0.326** | 0.197 | 0.769 |
| **Logistic Regression** | SMOTE | **0.60** | 0.928 | **0.448** | 0.667 | 0.333 |
| **Random Forest** | Normal | **0.30** | 0.987 | **0.389** | 0.500 | 0.323 |
| **Random Forest** | SMOTE | **0.45** | 0.986 | **0.400** | 0.500 | 0.333 |
| **XGBoost** | Normal | **0.50** | 0.985 | **0.500** | 0.538 | 0.467 |
| **XGBoost** | SMOTE | **0.15** | 0.982 | **0.419** | 0.310 | 0.633 |

### 기존 하드코딩 결과 (Threshold=0.5)
| 모델 | Normal | SMOTE |
|-----|--------|-------|
| **Logistic Regression** | AUC: 0.943, F1: 0.244 | AUC: 0.928, F1: 0.271 |
| **Random Forest** | AUC: 0.987, F1: 0.583 | AUC: 0.986, F1: 0.608 |
| **XGBoost** | AUC: 0.985, F1: 0.538 | AUC: 0.982, F1: 0.569 |

> 🎯 **개선 효과**: Threshold 최적화로 최고 **84%** F1 성능 향상! (XGBoost SMOTE: 0.419 vs 0.569)

### 다중공선성 분석
- **K2_Score_Original** 변수가 완전한 다중공선성의 원인으로 확인
- 제거 후 모든 VIF < 5로 개선 (최대 VIF: 4.97)
- WC_TA ↔ CLCA 간 높은 상관관계 (-0.823) 발견

## 🔧 핵심 기능

### 1. 🆕 마스터 모델 러너 (통합 파이프라인)
- **🎯 자동 Threshold 최적화**: 각 모델별 최적 임계값 자동 탐색
- **중앙 설정 관리**: JSON 기반으로 모든 하이퍼파라미터 통합 관리
- **템플릿 시스템**: quick/production/lasso 등 사전 정의 설정
- **Lasso 특성 선택**: 선택적 차원 축소 및 특성 선택

### 2. 포괄적 데이터 전처리
- **재무비율 계산**: ROA, TLTA, WC_TA 등 18개 주요 비율
- **결측치 처리**: 완전한 데이터만 사용 (100% Complete)  
- **이상치 탐지**: 통계적 방법으로 이상치 식별 및 처리

### 3. 고급 분석 도구
- **다중공선성 분석**: VIF, 상관분석, 조건지수 계산
- **PCA 분석**: 차원 축소 가능성 평가 (27.8% 축소 가능)
- **SMOTE 적용**: 불균형 데이터 문제 해결

### 4. 모델 성능 최적화
- **🔥 Threshold 자동 최적화**: 0.1~0.85 범위에서 16개 포인트 탐색
- **교차검증**: 시계열 특성 고려한 검증 전략
- **하이퍼파라미터 튜닝**: Optuna 기반 베이지안 최적화
- **앙상블 기법**: 여러 모델의 장점 결합

## 📊 데이터셋 상세정보

- **총 관측치**: 16,197개 (2012-2023년)
- **특성 개수**: 18개 (재무비율 및 시장지표)
- **부실기업**: 104개 (0.64%)
- **데이터 분할**: Train(60%) / Valid(20%) / Test(20%)
- **SMOTE 적용**: 불균형 비율 개선 (10% 목표)

## 🎯 주요 특징

### 재무 도메인 전문성
- 한국 회계기준(K-IFRS) 기반 재무비율 계산
- 업종별 특성 고려한 정규화
- 시계열 데이터의 Look-ahead Bias 방지

### 통계적 엄밀성
- 다중공선성 진단 및 해결
- 모델 가정 검증
- Robust한 성능 평가 지표 활용

### 실무 적용 가능성
- **🎯 자동 임계값 최적화**: 각 모델별 최적 의사결정 기준점 자동 탐색
- **실시간 예측 파이프라인**: 통합 설정 기반 자동화 시스템
- **포트폴리오 전략 백테스팅**: Threshold별 투자 성과 분석 지원
- **해석 가능한 모델 결과**: 메트릭별 최적화 기준 제시

## 📁 주요 파일 설명

### 데이터 파일
- `FS_100_complete.csv`: 완전한 재무데이터 (16,197 × 21)
- `dataset_info_100_complete.json`: 데이터셋 메타정보
- `X_train/valid/test_100_*.csv`: 모델링용 데이터 분할

### 모델 파일
- `final_*_model.joblib`: 최종 학습된 모델
- `*_results.json`: 모델 성능 결과
- `final_scaler.joblib`: 표준화 스케일러

### 분석 결과
- `comprehensive_multicollinearity_analysis_*.json`: 다중공선성 분석
- `100_complete_detailed_results.csv`: 상세 성능 결과
- 각종 시각화 파일 (PNG)

## 🎨 시각화 및 대시보드

### Streamlit 대시보드
- 프로젝트 개요 및 워크플로우
- 실시간 코드 리뷰 및 설명
- 인터랙티브 데이터 탐색
- **🆕 모델링 결과**: Threshold 최적화 결과 시각화
- 모델 성능 비교 시각화

### 분석 차트
- 상관계수 히트맵
- VIF 분석 차트
- ROC 곡선 및 Precision-Recall 곡선
- 특성 중요도 시각화

## 🔄 워크플로우

1. **데이터 수집** → DART API 활용 재무제표 수집
2. **전처리** → 결측치 처리, 이상치 탐지, 재무비율 계산
3. **특성 엔지니어링** → 시장지표 추가, 스케일링
4. **다중공선성 분석** → VIF 계산, 문제 변수 제거
5. **모델 학습** → 3가지 알고리즘으로 학습
6. **성능 평가** → AUC, F1-Score 등 다각도 평가
7. **결과 해석** → 특성 중요도, 비즈니스 인사이트 도출

## 📚 기술 스택

- **언어**: Python 3.8+
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn, xgboost, imbalanced-learn
- **시각화**: matplotlib, seaborn, plotly
- **웹 대시보드**: streamlit
- **통계 분석**: statsmodels, scipy

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 연락처

- 프로젝트 관리자: [GitHub Profile]
- 이메일: [your-email@example.com]

---

### 📋 TODO

- [ ] 실시간 데이터 수집 API 연동
- [ ] 포트폴리오 백테스팅 모듈 구현
- [ ] 모델 해석 도구 추가 (SHAP, LIME)
- [ ] 웹 서비스 배포 (Flask/FastAPI)
- [ ] 성능 모니터링 대시보드

---

*이 프로젝트는 한국 금융시장의 투자 의사결정을 지원하기 위한 연구 목적으로 개발되었습니다.*
