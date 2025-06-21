import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== 1단계: 기본 재무비율 계산 (FS_flow 활용) ===")

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
fs_flow = pd.read_csv('data/processed/FS_flow_fixed.csv')
print(f"FS_flow.csv: {fs_flow.shape}")
print(f"컬럼 확인: {list(fs_flow.columns)}")

# 2. 기본 재무비율 계산
print("\n2. 기본 재무비율 계산 중...")

# 결과 데이터프레임 생성
result_df = fs_flow[['회사명', '거래소코드', '회계년도', '연도']].copy()

with np.errstate(divide='ignore', invalid='ignore'):
    # ROA = 당기순이익 / 자산_평균
    result_df['ROA'] = fs_flow['당기순이익'] / fs_flow['자산_평균']
    
    # TLTA = 부채_평균 / 자산_평균  
    result_df['TLTA'] = fs_flow['부채_평균'] / fs_flow['자산_평균']
    
    # WC/TA = (유동자산_평균 - 유동부채_평균) / 자산_평균
    result_df['WC_TA'] = (fs_flow['유동자산_평균'] - fs_flow['유동부채_평균']) / fs_flow['자산_평균']
    
    # CFO/TD = 영업현금흐름 / 부채_평균
    result_df['CFO_TD'] = fs_flow['영업현금흐름'] / fs_flow['부채_평균']
    
    # RE/TA = 이익잉여금_평균 / 자산_평균
    result_df['RE_TA'] = fs_flow['이익잉여금_평균'] / fs_flow['자산_평균']
    
    # EBIT/TA = 영업손익 / 자산_평균
    result_df['EBIT_TA'] = fs_flow['영업손익'] / fs_flow['자산_평균']
    
    # S/TA = 매출액 / 자산_평균
    result_df['S_TA'] = fs_flow['매출액'] / fs_flow['자산_평균']
    
    # CLCA = 유동부채_평균 / 유동자산_평균
    result_df['CLCA'] = fs_flow['유동부채_평균'] / fs_flow['유동자산_평균']
    
    # OENEG = IF(자산_평균 < 부채_평균, 1, 0)
    result_df['OENEG'] = (fs_flow['자산_평균'] < fs_flow['부채_평균']).astype(int)
    
    # CR = 유동자산_평균 / 유동부채_평균
    result_df['CR'] = fs_flow['유동자산_평균'] / fs_flow['유동부채_평균']
    
    # CFO/TA = 영업현금흐름 / 자산_평균
    result_df['CFO_TA'] = fs_flow['영업현금흐름'] / fs_flow['자산_평균']

# 3. 무한대 및 NaN 처리
print("\n3. 데이터 정리 중...")
result_df = result_df.replace([np.inf, -np.inf], np.nan)

# 4. 결과 저장
print("\n4. 결과 저장 중...")
result_df.to_csv('data/processed/temp_basic_ratios_flow.csv', index=False, encoding='utf-8-sig')

print(f"\n=== 1단계 완료 ===")
print(f"기본 재무비율 계산 완료: {result_df.shape}")
print(f"저장 위치: data/processed/temp_basic_ratios_flow.csv")

# 5. 결과 요약
print(f"\n=== 기본 재무비율 요약 ===")
basic_ratios = ['ROA', 'TLTA', 'WC_TA', 'CFO_TD', 'RE_TA', 'EBIT_TA', 'S_TA', 'CLCA', 'OENEG', 'CR', 'CFO_TA']
for ratio in basic_ratios:
    valid_count = result_df[ratio].notna().sum()
    valid_pct = valid_count / len(result_df) * 100
    if valid_count > 0:
        mean_val = result_df[ratio].mean()
        print(f"{ratio:8}: {valid_count:,}개 ({valid_pct:.1f}%) | 평균: {mean_val:.4f}")
    else:
        print(f"{ratio:8}: {valid_count:,}개 ({valid_pct:.1f}%)")

print(f"\n💡 개선사항:")
print("- Stock 지표는 평균값 사용으로 더 정확한 비율 계산")
print("- Flow 지표와의 매칭이 개선됨")

print(f"\n샘플 데이터:")
print(result_df[['회사명', '연도', 'ROA', 'TLTA', 'WC_TA']].head()) 