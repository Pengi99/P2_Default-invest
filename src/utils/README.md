# utils

공통 유틸리티 함수 및 헬퍼 모듈들

## 📋 개요
이 폴더는 프로젝트 전반에서 재사용 가능한 유틸리티 함수들을 포함합니다.
현재는 비어있지만, 프로젝트 진행하면서 공통 기능들을 모듈화하여 추가할 예정입니다.

## 🎯 향후 추가 예정 모듈들

### 📊 data_utils.py
**데이터 처리 공통 함수들**
- 안전한 나눗셈 함수 (division by zero 방지)
- 데이터 타입 변환 및 검증
- 결측치 처리 표준화
- CSV 파일 읽기/쓰기 헬퍼

### 📈 financial_utils.py
**금융 계산 전용 함수들**
- 재무비율 계산 공통 함수
- 수익률 계산 (단순, 복리, 로그)
- 변동성 측정 (표준편차, VaR)
- 시계열 금융 지표 계산

### 📋 validation_utils.py
**데이터 검증 및 품질 관리**
- 데이터 품질 체크 함수
- 이상치 탐지 및 처리
- 시계열 데이터 일관성 검증
- Look-ahead Bias 검증

### 📊 visualization_utils.py
**시각화 공통 함수들**
- 한글 폰트 자동 설정
- 금융 차트 템플릿 (캔들스틱, 시계열)
- 모델 성능 시각화 (ROC, Confusion Matrix)
- 재무비율 분포 시각화

### 🤖 model_utils.py
**모델링 공통 함수들**
- 교차 검증 헬퍼
- 하이퍼파라미터 튜닝 템플릿
- 모델 성능 평가 함수
- 특성 중요도 시각화

### ⚙️ config_utils.py
**설정 관리 함수들**
- 설정 파일 로드/저장
- 환경별 설정 관리
- 로깅 설정 표준화
- 경로 관리 함수

## 📋 사용 예시 (향후)

```python
# 데이터 처리
from src_new.utils.data_utils import safe_division, load_csv_with_encoding
from src_new.utils.financial_utils import calculate_roa, calculate_volatility

# 시각화
from src_new.utils.visualization_utils import setup_korean_font, plot_financial_ratios

# 모델링
from src_new.utils.model_utils import evaluate_binary_classifier, plot_feature_importance
```

## 🚀 개발 가이드라인

### 함수 작성 원칙
1. **재사용성**: 여러 모듈에서 사용되는 공통 기능
2. **독립성**: 외부 의존성 최소화
3. **문서화**: 상세한 docstring 필수
4. **테스트**: 단위 테스트 작성 권장

### 네이밍 규칙
- 함수명: snake_case
- 클래스명: PascalCase
- 상수: UPPER_CASE
- 모듈명: 기능별 명확한 이름

### 코드 품질
- PEP 8 스타일 가이드 준수
- Type hints 사용 권장
- 예외 처리 포함
- 로깅 활용

## 🎯 기여 방법

1. **중복 코드 발견시**: 공통 함수로 추출하여 utils에 추가
2. **새로운 유틸리티**: 적절한 모듈에 추가 또는 새 모듈 생성
3. **문서화**: README 업데이트 및 함수별 docstring 작성
4. **테스트**: 가능한 경우 단위 테스트 추가

## 📝 참고사항
- 금융 도메인 특화 함수들은 financial_utils.py에 집중
- 범용적인 데이터 처리는 data_utils.py 활용
- 시각화 관련은 visualization_utils.py로 분리
- 프로젝트 진행하면서 필요에 따라 모듈 추가/수정 