# Default Invest

본 프로젝트는 한국 상장기업의 부실 위험을 예측하고 퀄리티 팩터 기반의 포트폴리오 전략을 백테스트하기 위한 파이프라인을 제공합니다.

## 프로젝트 구조

```
.
├── analysis_scripts/
│   ├── analyze_delisted_reasons.py
│   └── compare_tickers.py
├── data/
│   ├── raw/          # 수집한 원본 데이터
│   └── processed/    # 전처리 후 저장되는 데이터
├── models/           # 학습된 최적 모델 저장 위치
├── notebooks/
│   └── jongho.ipynb
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── backtester.py
│   ├── data_collector.py
│   ├── financial_ratios.py
│   ├── model_trainer.py
│   └── preprocess.py
├── main.py           # 파이프라인 실행 스크립트
├── config.py         # 설정 파일
└── requirements.txt
```

### 폴더 설명

- **analysis_scripts/**: 데이터 불일치 분석 등 별도 실험용 스크립트
- **data/raw/**: DART에서 받은 원본 CSV 파일을 저장
- **data/processed/**: 전처리 후 분석과 모델 학습에 사용하는 데이터
- **models/**: 학습된 최적 모델을 저장
- **notebooks/**: 초기 탐색과 아이디어 검증을 위한 Jupyter 노트북
- **src/**: 파이프라인을 구성하는 주요 모듈 (데이터 수집, 전처리, 모델 학습, 백테스트 등)
- **main.py**: 전체 파이프라인을 순차적으로 실행하는 진입점
- **config.py**: 파라미터와 경로 등을 한 곳에서 관리

## 빠른 시작

1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 데이터 수집부터 백테스트까지 전체 파이프라인 실행
   ```bash
   python main.py
   ```

각 모듈의 세부 기능은 주석과 TODO 항목을 참고해 채워 넣을 수 있습니다.

## 데이터 파일 설명

- **data/raw/** 폴더에는 다음과 같은 원본 CSV가 들어 있습니다.
  - `연결 재무제표(IFRS).csv`, `재무제표(IFRS).csv`: 기업의 재무상태표와 손익계산서
  - `연결 재무비율(IFRS).csv`, `재무비율(IFRS).csv`: 주요 재무비율 정보
  - `코스피_상장폐지.csv`, `코스닥_상장폐지.csv`: 상장폐지 기업 목록과 사유
- 전처리 과정에서 위 데이터를 결합하여 `data/processed/financial_ratios.csv` 파일을 생성하며, 향후 학습 및 백테스트에 활용합니다.

## 주요 기능

- **데이터 수집**: DART 등에서 재무제표와 주가 데이터를 가져와 `data/raw/`에 저장합니다.
- **전처리 및 피처 생성**: K-1 스코어 계산, 부도 라벨 생성, 퀄리티 팩터 산출 등을 수행합니다.
- **모델 학습**: 로지스틱 회귀, 랜덤 포레스트, LightGBM을 비교하여 최적 모델을 선택합니다.
- **백테스팅**: 예측 모델로 필터링한 포트폴리오와 비교군을 구성하여 전략 성과를 검증합니다.

## 노트북 활용

`notebooks/jongho.ipynb` 파일에서 데이터 탐색과 간단한 모델 실험을 진행할 수 있습니다. 본격적인 코드 작성 전 아이디어를 검증하는 데 활용하세요.
