# data_processing

데이터 전처리 및 정제 스크립트들

## 📄 스크립트별 상세 기능

### 🔧 create_financial_ratios_master.py
**재무비율 계산 마스터 프로세스 관리자**

**주요 기능:**
- 4단계 재무비율 계산 프로세스의 통합 실행 관리
- 각 단계별 실행 시간 측정 및 성공/실패 모니터링
- 오류 발생 시 프로세스 중단 및 상세 로그 제공

**실행 단계:**
1. **1단계**: 기본 재무비율 계산 (FS_flow 활용)
   - ROA, TLTA, WC_TA, CFO_TD, RE_TA, EBIT_TA 등
2. **2단계**: 시장기반 재무비율 계산
   - MVE_TL, TLMTA, MB (시가총액 데이터 활용)
3. **3단계**: 변동성과 수익률 계산
   - SIGMA (주가 변동성), RET_3M, RET_9M (수익률)
4. **4단계**: 최종 재무비율 정리 및 저장
   - 모든 비율 통합, 품질 검증, CSV 저장

**입력 데이터:**
- `data/processed/FS_flow_fixed.csv` (재무제표 데이터)
- `data/processed/1m_fixed.csv` (주가 데이터)

**출력 데이터:**
- `data/processed/FS_ratio_flow.csv` (최종 재무비율)

**사용법:**
```bash
python src_new/data_processing/create_financial_ratios_master.py
```

**특징:**
- 각 단계별 독립 실행 가능
- 실행 시간 및 성공률 통계 제공
- 오류 발생 시 상세 디버깅 정보 출력
- 최종 파일 크기 및 품질 검증

---

## 📋 의존성 스크립트들 (참조용)

### step1_basic_financial_ratios.py
- 기본 재무비율 11개 계산
- Altman Z-Score 구성 요소 포함

### step2_market_based_ratios.py  
- 시장 데이터 기반 비율 3개 계산
- 시가총액/부채, 장부가/시가 비율

### step3_volatility_returns.py
- 주가 변동성 및 수익률 3개 계산
- 3개월/9개월 수익률, 연간 변동성

### step4_finalize_ratios.py
- 모든 비율 통합 및 품질 검증
- 최종 데이터셋 생성

## 🚀 실행 순서
1. 원본 데이터 확인 (FS_flow_fixed.csv, 1m_fixed.csv)
2. create_financial_ratios_master.py 실행
3. 결과 파일 검증 (FS_ratio_flow.csv)
4. 다음 단계 (feature_engineering)로 진행
