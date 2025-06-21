# feature_engineering

특성 생성 및 변환 스크립트들

## 📄 스크립트별 상세 기능

### 🔧 add_financial_variables.py
**추가 재무변수 계산 및 컬럼명 영문화**

**주요 기능:**
- 기존 재무제표 데이터에서 추가적인 재무비율 계산
- 한글 컬럼명을 영문으로 변환하여 국제 표준 준수
- 시계열적 변화율 계산 (전년 대비 성장률 등)

**계산되는 재무변수 (13개):**
1. **debt_to_equity_ratio**: 부채자본비율 (부채/자본 × 100)
2. **return_on_assets (roa)**: 총자산수익률 (순이익/총자산 × 100)
3. **return_on_equity (roe)**: 자기자본수익률 (순이익/자본 × 100)
4. **cfo_to_debt_ratio**: 현금흐름부채비율 (영업현금흐름/부채)
5. **asset_growth_rate**: 자산성장률 (전년 대비 총자산 증가율)
6. **interest_coverage_ratio**: 이자보상비율 (영업이익/이자비용)
7. **retained_earnings_ratio**: 이익잉여금비율 (이익잉여금/총자산)
8. **ebit_to_assets_ratio**: EBIT자산비율 (영업이익/총자산)
9. **net_income_to_total_assets_ratio**: 순이익자산비율 (순이익/총자산)
10. **cfo_to_total_liabilities_ratio**: 영업현금흐름부채비율
11. **loss_dummy_intwo**: 연속손실더미 (2년 연속 순손실 = 1)
12. **insolvency_dummy_oeneg**: 부실더미 (부채 > 자산 = 1)
13. **net_income_change_ratio**: 순이익변화율 (전년 대비 순이익 변화)

**입력 데이터:**
- `data/processed/BS_ratio.csv` (대차대조표 비율 데이터)
- `data/processed/final.csv` (기존 통합 데이터)

**출력 데이터:**
- `data/processed/final.csv` (추가 변수가 포함된 통합 데이터)

**사용법:**
```bash
python src_new/feature_engineering/add_financial_variables.py
```

**특징:**
- 안전한 나눗셈 함수로 0으로 나누기 오류 방지
- 시계열 데이터 고려한 전년도 값 계산
- 생성 성공/실패 변수 목록 자동 출력
- UTF-8 인코딩으로 저장하여 호환성 향상

---

### 🔧 create_final_modeling_dataset.py
**최종 모델링용 데이터셋 생성 및 스케일링**

**주요 기능:**
- 부실 라벨링이 적용된 데이터에 스케일링 적용
- 훈련/검증/테스트 데이터 분할 (시계열 고려)
- 다양한 스케일링 방법 적용 및 비교

**스케일링 방법:**
1. **StandardScaler**: 평균 0, 표준편차 1로 정규화
   - 정규분포에 가까운 데이터에 적합
   - 이상치에 민감함
2. **RobustScaler**: 중앙값과 IQR 기반 정규화
   - 이상치에 강건함
   - 왜도가 높은 재무비율에 적합

**데이터 분할 전략:**
- **시간 기반 분할**: 시계열 특성 고려
- **계층 분할**: 부실/정상 비율 유지
- **비율**: Train 70% / Validation 15% / Test 15%

**입력 데이터:**
- `data/processed/FS_ratio_flow_labeled.csv` (라벨링된 재무비율)

**출력 데이터:**
- `data/processed/FS_ratio_flow_scaled.csv` (스케일링된 전체 데이터)
- `data/processed/X_train.csv`, `data/processed/y_train.csv` (훈련 데이터)
- `data/processed/X_val.csv`, `data/processed/y_val.csv` (검증 데이터)
- `data/processed/X_test.csv`, `data/processed/y_test.csv` (테스트 데이터)
- `data/processed/dataset_info.json` (데이터셋 정보)

**사용법:**
```bash
python src_new/feature_engineering/create_final_modeling_dataset.py
```

**특징:**
- 스케일링 방법별 성능 비교 가능
- 데이터 누수(Data Leakage) 방지
- 재현 가능한 랜덤 시드 설정
- 상세한 데이터셋 통계 정보 제공

---

## 🔄 워크플로우

1. **add_financial_variables.py** 실행
   - 기본 재무변수에 추가 변수 계산
   - 컬럼명 영문화

2. **create_final_modeling_dataset.py** 실행  
   - 부실 라벨링 적용
   - 스케일링 및 데이터 분할

## 📊 생성되는 특성 요약

- **기본 재무비율**: 17개 (ROA, TLTA, WC_TA 등)
- **추가 재무변수**: 13개 (ROE, 성장률, 더미변수 등)
- **총 특성 수**: 30개
- **타겟 변수**: default (0: 정상, 1: 부실)

## 🎯 다음 단계
생성된 데이터셋을 `modeling/` 폴더의 스크립트들로 모델 훈련
