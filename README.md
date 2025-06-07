# Default Invest

본 프로젝트는 한국 상장기업의 부실 위험을 예측하고 퀄리티 팩터 기반의 포트폴리오 전략을 백테스트하기 위한 파이프라인을 제공합니다.

## 프로젝트 구조

```
.
├── data/
│   ├── raw/          # 수집한 원본 데이터
│   └── processed/    # 전처리 후 저장되는 데이터
├── models/           # 학습된 최적 모델 저장 위치
├── notebooks/
│   └── 1_eda_and_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data_collector.py
│   ├── preprocess.py
│   ├── model_trainer.py
│   ├── backtester.py
│   └── analysis.py
├── main.py           # 파이프라인 실행 스크립트
├── config.py         # 설정 파일
└── requirements.txt
```

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

## 주요 기능

- **데이터 수집**: DART 등에서 재무제표와 주가 데이터를 가져와 `data/raw/`에 저장합니다.
- **전처리 및 피처 생성**: K-1 스코어 계산, 부도 라벨 생성, 퀄리티 팩터 산출 등을 수행합니다.
- **모델 학습**: 로지스틱 회귀, 랜덤 포레스트, LightGBM을 비교하여 최적 모델을 선택합니다.
- **백테스팅**: 예측 모델로 필터링한 포트폴리오와 비교군을 구성하여 전략 성과를 검증합니다.

## 노트북 활용

`notebooks/1_eda_and_prototyping.ipynb` 파일에서 데이터 탐색과 간단한 모델 실험을 진행할 수 있습니다. 본격적인 코드 작성 전 아이디어를 검증하는 데 활용하세요.
