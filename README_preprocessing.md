# 데이터 전처리 파이프라인

자동화된 데이터 전처리 파이프라인으로 다음 기능들을 지원합니다:

## 🚀 주요 기능

1. **데이터 분할**: 5:3:2 비율로 train/validation/test 분할
2. **결측치 처리**: 50% 이상 결측 행 삭제 + median 대체
3. **윈저라이징**: 양 옆 0.05% 이상치 제거
4. **스케일링**: Standard/Robust 스케일링 지원
5. **피처 선택**: 라소 회귀를 이용한 자동 피처 선택
6. **Config 기반**: YAML 설정으로 모든 매개변수 커스터마이징

## 📁 파일 구조

```
├── config/
│   └── preprocessing_config.yaml    # 전처리 설정 파일
├── src/preprocessing/
│   └── data_pipeline.py            # 메인 파이프라인 클래스
├── examples/
│   └── run_preprocessing.py        # 실행 예제
└── README_preprocessing.md         # 이 파일
```

## 🛠️ 사용 방법

### 1. 기본 실행

```bash
# 예제 스크립트로 실행
python examples/run_preprocessing.py

# 또는 직접 실행
python src/preprocessing/data_pipeline.py --config config/preprocessing_config.yaml
```

### 2. 커스텀 설정으로 실행

config 파일을 복사하여 수정한 후 사용:

```bash
cp config/preprocessing_config.yaml config/my_config.yaml
# my_config.yaml 편집 후
python src/preprocessing/data_pipeline.py --config config/my_config.yaml
```

### 3. Python 코드에서 직접 사용

```python
from src.preprocessing.data_pipeline import DataPreprocessingPipeline

# 파이프라인 초기화
pipeline = DataPreprocessingPipeline("config/preprocessing_config.yaml")

# 전체 과정 실행
experiment_dir = pipeline.run_pipeline()

# 결과 확인
print(f"결과 저장 위치: {experiment_dir}")
print(f"선택된 피처 수: {len(pipeline.results['selected_features'])}")
```

## ⚙️ 설정 옵션

### 데이터 경로 및 출력
```yaml
data:
  input_path: "data/final/FS_ratio_flow_labeled.csv"
  output_dir: "data/final"    # X_train.csv 등이 저장될 위치

experiment:
  create_subdirectory: false  # false면 data/final에 직접 저장
```

### 파일 저장 형식
```yaml
output:
  file_naming:
    separate_features_target: true      # X, y 분리 저장
    feature_format: "X_{split}.csv"    # X_train.csv, X_val.csv, X_test.csv
    target_format: "y_{split}.csv"     # y_train.csv, y_val.csv, y_test.csv
```

### 데이터 분할
```yaml
data_split:
  train_ratio: 0.5      # Train 비율 (50%)
  val_ratio: 0.3        # Validation 비율 (30%)
  test_ratio: 0.2       # Test 비율 (20%)
  stratify: true        # 층화 샘플링 여부
```

### 결측치 처리
```yaml
missing_data:
  row_missing_threshold: 0.5    # 행 삭제 임계값 (50% 이상 결측)
  imputation_method: "median"   # median, mean, mode, knn
```

### 윈저라이징
```yaml
outlier_treatment:
  enabled: true
  winsorization:
    lower_percentile: 0.05    # 하위 5%
    upper_percentile: 0.95    # 상위 5% (양 옆 0.05씩)
```

### 스케일링
```yaml
scaling:
  methods: ["standard", "robust"]    # 사용할 스케일링 방법들
  default_method: "standard"         # 기본 방법
```

### 라소 피처 선택
```yaml
feature_selection:
  enabled: false      # true로 변경하면 피처 선택 활성화
  lasso:
    alpha_range: [0.001, 0.01, 0.1, 1.0, 10.0]
    cv_folds: 5
    alpha_selection: "1se"    # "min" 또는 "1se"
```

## 📊 출력 결과

실행 완료 후 `data/final/` 디렉토리에 다음 파일들이 생성됩니다:

### 데이터 파일 (X, y 분리 저장)
- `X_train.csv`: 학습 피처 데이터
- `y_train.csv`: 학습 타겟 데이터
- `X_val.csv`: 검증 피처 데이터  
- `y_val.csv`: 검증 타겟 데이터
- `X_test.csv`: 테스트 피처 데이터
- `y_test.csv`: 테스트 타겟 데이터

### 모델 파일
- `scaler_standard.pkl`: Standard 스케일러
- `scaler_robust.pkl`: Robust 스케일러
- `feature_selector.pkl`: 라소 피처 선택 모델 (피처 선택 활성화 시)

### 결과 파일
- `preprocessing_report.txt`: 전처리 과정 상세 리포트
- `preprocessing_report.html`: HTML 형식 리포트

## 📈 결과 예시

```
✅ 전처리 파이프라인이 성공적으로 완료되었습니다!

📊 데이터 정보:
   원본 데이터: (22780, 36)
   Train: (11006, 36)
   Validation: (6559, 36)  
   Test: (4380, 36)

🎯 피처 선택:
   상태: 비활성화됨 (또는 활성화 시 피처 수 정보)
   모든 피처가 유지됨

📝 생성된 파일들:
   ✅ X_train.csv
   ✅ y_train.csv
   ✅ X_val.csv
   ✅ y_val.csv
   ✅ X_test.csv
   ✅ y_test.csv
   ✅ scaler_standard.pkl
   ✅ scaler_robust.pkl
   ✅ preprocessing_report.txt
```

## 🔧 고급 사용법

### 1. 새로운 전처리 단계 추가

`DataPreprocessingPipeline` 클래스를 상속받아 새로운 메서드 추가:

```python
class CustomPipeline(DataPreprocessingPipeline):
    def apply_custom_preprocessing(self, train_df, val_df, test_df):
        # 커스텀 전처리 로직
        return train_df, val_df, test_df
    
    def run_pipeline(self):
        # 기존 파이프라인에 커스텀 단계 추가
        # ... 기존 단계들 ...
        train_df, val_df, test_df = self.apply_custom_preprocessing(train_df, val_df, test_df)
        # ... 나머지 단계들 ...
```

### 2. 다른 피처 선택 방법 사용

config에서 `feature_selection.method`를 수정하여 다른 방법 구현 가능:

```yaml
feature_selection:
  method: "recursive_elimination"  # 새로운 방법
  # 해당 방법의 설정들...
```

### 3. 배치 처리

여러 설정으로 실험을 반복 실행:

```python
configs = ["config1.yaml", "config2.yaml", "config3.yaml"]
results = []

for config_path in configs:
    pipeline = DataPreprocessingPipeline(config_path)
    experiment_dir = pipeline.run_pipeline()
    results.append({
        'config': config_path,
        'experiment_dir': experiment_dir,
        'performance': pipeline.results['model_performance']
    })
```

## 🐛 문제 해결

### 메모리 부족
```yaml
performance:
  optimize_memory: true
  n_jobs: 1  # 병렬 처리 줄이기
```

### 실행 시간 단축
```yaml
feature_selection:
  lasso:
    alpha_range: [0.01, 0.1, 1.0]  # alpha 후보 줄이기
    cv_folds: 3                    # CV 폴드 줄이기
```

### 로그 레벨 조정
```yaml
logging:
  level: "WARNING"  # INFO 대신 WARNING으로 설정
```

## 💡 주요 특징

- **Config 기반 관리**: 모든 설정을 YAML 파일로 관리하여 쉬운 커스터마이징
- **모듈화된 구조**: 각 전처리 단계를 독립적으로 활성화/비활성화 가능
- **X, y 분리 저장**: 머신러닝 모델에서 바로 사용 가능한 형태로 저장
- **스케일러 저장**: 학습된 스케일러를 저장하여 새로운 데이터에 동일한 변환 적용 가능
- **상세한 로그**: 전체 과정이 로그로 기록되어 디버깅과 분석이 용이

## 📞 지원

문제가 발생하면 다음을 확인해주세요:
1. 데이터 파일 경로가 올바른지 확인 (`data/final/FS_ratio_flow_labeled.csv`)
2. 필요한 Python 패키지가 모두 설치되어 있는지 확인
3. Config 파일의 YAML 문법이 올바른지 확인
4. 로그 파일(`logs/preprocessing.log`)에서 자세한 오류 메시지 확인
5. `data/final/` 디렉토리에 쓰기 권한이 있는지 확인