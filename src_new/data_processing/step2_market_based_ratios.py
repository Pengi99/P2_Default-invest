import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== 2단계: 시장기반 재무비율 계산 ===")

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
basic_ratios = pd.read_csv('data/processed/temp_basic_ratios_flow.csv')
monthly_data = pd.read_csv('data/processed/1m_fixed.csv')
fs_flow = pd.read_csv('data/processed/FS_flow_fixed.csv')

print(f"기본 재무비율: {basic_ratios.shape}")
print(f"월별 데이터: {monthly_data.shape}")
print(f"FS_flow: {fs_flow.shape}")

# 2. 월별 데이터 전처리
print("\n2. 월별 데이터 전처리 중...")

# 연월 형식 확인 및 처리
if '-' in str(monthly_data['연월'].iloc[0]):
    monthly_data['연도'] = monthly_data['연월'].str[:4].astype(int)
    monthly_data['월'] = monthly_data['연월'].str[5:7].astype(int)
else:
    monthly_data['연도'] = monthly_data['연월'] // 100
    monthly_data['월'] = monthly_data['연월'] % 100

print(f"월별 데이터 연도 범위: {monthly_data['연도'].min()} ~ {monthly_data['연도'].max()}")

# 3. 12월 데이터 추출 (연말 기준)
print("\n3. 12월 데이터 추출 중...")
monthly_dec = monthly_data[monthly_data['월'] == 12].copy()
print(f"12월 데이터 개수: {len(monthly_dec)}")

if len(monthly_dec) == 0:
    print("12월 데이터가 없습니다. 가장 많은 월 데이터를 사용합니다.")
    most_common_month = monthly_data['월'].mode()[0]
    monthly_dec = monthly_data[monthly_data['월'] == most_common_month].copy()
    print(f"{most_common_month}월 데이터 사용: {len(monthly_dec)}개")

# 4. 매칭 키 생성
print("\n4. 매칭 키 생성 중...")
basic_ratios['매칭키'] = basic_ratios['거래소코드'].astype(str) + '_' + basic_ratios['연도'].astype(str)
monthly_dec['매칭키'] = monthly_dec['거래소코드'].astype(str) + '_' + monthly_dec['연도'].astype(str)

# 공통 키 확인
basic_keys = set(basic_ratios['매칭키'])
monthly_keys = set(monthly_dec['매칭키'])
common_keys = basic_keys & monthly_keys
print(f"기본비율 키: {len(basic_keys)}개")
print(f"월별 키: {len(monthly_keys)}개") 
print(f"공통 키: {len(common_keys)}개")

# 5. 월별 데이터와 매칭
print("\n5. 월별 데이터와 매칭 중...")
matched_data = basic_ratios.merge(
    monthly_dec[['매칭키', '월평균종가(원)', '월평균시가총액(원)']],
    on='매칭키',
    how='left'
)

matched_count = matched_data['월평균종가(원)'].notna().sum()
print(f"매칭 결과: {len(matched_data)}행 중 {matched_count}개 매칭 ({matched_count/len(matched_data)*100:.1f}%)")

# 6. FS_flow 데이터와 매칭 (부채, 자본 정보 필요)
print("\n6. FS_flow 데이터와 매칭 중...")
fs_subset = fs_flow[['거래소코드', '연도', '부채_평균', '자본_평균', '발행주식수_평균']].copy()
fs_subset['매칭키'] = fs_subset['거래소코드'].astype(str) + '_' + fs_subset['연도'].astype(str)

matched_data = matched_data.merge(
    fs_subset[['매칭키', '부채_평균', '자본_평균', '발행주식수_평균']],
    on='매칭키',
    how='left'
)

# 7. 시장기반 재무비율 계산
print("\n7. 시장기반 재무비율 계산 중...")

with np.errstate(divide='ignore', invalid='ignore'):
    # MVE/TL = 월평균시가총액 / 부채_평균
    # 또는 MVE/TL = (월평균종가 × 발행주식수_평균) / 부채_평균
    matched_data['MVE_TL_방법1'] = matched_data['월평균시가총액(원)'] / matched_data['부채_평균']
    matched_data['MVE_TL_방법2'] = (matched_data['월평균종가(원)'] * matched_data['발행주식수_평균']) / matched_data['부채_평균']
    
    # 방법1 우선, 없으면 방법2 사용
    matched_data['MVE_TL'] = matched_data['MVE_TL_방법1'].fillna(matched_data['MVE_TL_방법2'])
    
    # TLMTA = 부채_평균 / (월평균시가총액 + 부채_평균)
    matched_data['시가총액'] = matched_data['월평균시가총액(원)'].fillna(
        matched_data['월평균종가(원)'] * matched_data['발행주식수_평균']
    )
    matched_data['TLMTA'] = matched_data['부채_평균'] / (matched_data['시가총액'] + matched_data['부채_평균'])
    
    # MB = 월평균시가총액 / 자본_평균
    matched_data['MB'] = matched_data['시가총액'] / matched_data['자본_평균']

# 8. 불필요한 임시 컬럼 제거
print("\n8. 데이터 정리 중...")
columns_to_drop = ['MVE_TL_방법1', 'MVE_TL_방법2', '시가총액', '부채_평균', '자본_평균', '발행주식수_평균']
for col in columns_to_drop:
    if col in matched_data.columns:
        matched_data = matched_data.drop(columns=[col])

# 무한대 및 NaN 처리
matched_data = matched_data.replace([np.inf, -np.inf], np.nan)

# 9. 결과 저장
print("\n9. 결과 저장 중...")
matched_data.to_csv('data/processed/temp_with_market_ratios_flow.csv', index=False, encoding='utf-8-sig')

print(f"\n=== 2단계 완료 ===")
print(f"시장기반 비율 추가 완료: {matched_data.shape}")
print(f"저장 위치: data/processed/temp_with_market_ratios_flow.csv")

# 10. 결과 요약
print(f"\n=== 시장기반 재무비율 요약 ===")
market_ratios = ['MVE_TL', 'TLMTA', 'MB']
for ratio in market_ratios:
    valid_count = matched_data[ratio].notna().sum()
    valid_pct = valid_count / len(matched_data) * 100
    if valid_count > 0:
        mean_val = matched_data[ratio].mean()
        print(f"{ratio:8}: {valid_count:,}개 ({valid_pct:.1f}%) | 평균: {mean_val:.4f}")
    else:
        print(f"{ratio:8}: {valid_count:,}개 ({valid_pct:.1f}%)")

print(f"\n매칭된 데이터 샘플:")
sample_cols = ['회사명', '연도', 'MVE_TL', 'TLMTA', 'MB']
available_cols = [col for col in sample_cols if col in matched_data.columns]
print(matched_data[available_cols].dropna().head()) 