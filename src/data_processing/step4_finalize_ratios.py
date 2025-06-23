import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== 4단계: 최종 재무비율 정리 및 저장 ===")

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
final_data = pd.read_csv('data/processed/temp_with_volatility_returns_flow.csv')
print(f"최종 데이터: {final_data.shape}")

# 2. 최종 컬럼 정리
print("\n2. 최종 컬럼 정리 중...")

# 필요한 컬럼만 선택 (Altman Z-Score 및 확장 모델용)
final_columns = [
    '회사명', '거래소코드', '회계년도',
    'ROA', 'TLTA', 'WC_TA', 'CFO_TD', 'SIGMA', 'RE_TA', 'EBIT_TA', 
    'MVE_TL', 'S_TA', 'CLCA', 'OENEG', 'CR', 'CFO_TA', 'TLMTA', 
    'RET_3M', 'RET_9M', 'MB'
]

# 컬럼 존재 여부 확인
existing_columns = [col for col in final_columns if col in final_data.columns]
missing_columns = [col for col in final_columns if col not in final_data.columns]

print(f"존재하는 컬럼: {len(existing_columns)}개")
if missing_columns:
    print(f"누락된 컬럼: {missing_columns}")

# 존재하는 컬럼만으로 최종 데이터 생성
final_df = final_data[existing_columns].copy()

# 3. 데이터 품질 검사
print("\n3. 데이터 품질 검사 중...")

print(f"총 데이터 개수: {len(final_df):,}개")
print(f"중복 데이터: {final_df.duplicated().sum()}개")

# 중복 제거
if final_df.duplicated().sum() > 0:
    final_df = final_df.drop_duplicates()
    print(f"중복 제거 후: {len(final_df):,}개")

# 4. 각 비율별 유효 데이터 통계
print("\n4. 각 비율별 유효 데이터 통계:")
ratio_columns = [col for col in existing_columns if col not in ['회사명', '거래소코드', '회계년도']]

for col in ratio_columns:
    valid_count = final_df[col].notna().sum()
    valid_pct = valid_count / len(final_df) * 100
    
    if valid_count > 0:
        mean_val = final_df[col].mean()
        std_val = final_df[col].std()
        print(f"{col:8}: {valid_count:,}개 ({valid_pct:.1f}%) | 평균: {mean_val:.4f} | 표준편차: {std_val:.4f}")
    else:
        print(f"{col:8}: {valid_count:,}개 ({valid_pct:.1f}%)")

# 5. 최종 저장
print("\n5. 최종 저장 중...")
final_df.to_csv('data/processed/FS_ratio_flow.csv', index=False, encoding='utf-8-sig')

print(f"\n=== 최종 완료 ===")
print(f"최종 데이터 크기: {final_df.shape}")
print(f"저장 위치: data/processed/FS_ratio_flow.csv")

# 6. 임시 파일 정리
print("\n6. 임시 파일 정리 중...")
import os

temp_files = [
    'data/processed/temp_basic_ratios_flow.csv',
    'data/processed/temp_with_market_ratios_flow.csv', 
    'data/processed/temp_with_volatility_returns_flow.csv'
]

for temp_file in temp_files:
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"삭제: {temp_file}")
    except Exception as e:
        print(f"삭제 실패: {temp_file} - {e}")

# 7. 최종 결과 요약
print(f"\n=== 최종 결과 요약 ===")
print(f"처리된 기업-연도 데이터: {len(final_df):,}개")
print(f"계산된 재무비율: {len(ratio_columns)}개")

print(f"\n📊 재무비율 설명 (FS_flow 기반 개선):")
ratio_descriptions = {
    'ROA': '당기순이익 / 자산_평균 (총자산수익률)',
    'TLTA': '부채_평균 / 자산_평균 (부채비율)', 
    'WC_TA': '(유동자산_평균 - 유동부채_평균) / 자산_평균 (운전자본비율)',
    'CFO_TD': '영업현금흐름 / 부채_평균 (현금흐름 대 부채비율)',
    'SIGMA': '주가 변동성 (일별 수익률 표준편차, 최근 3개월)',
    'RE_TA': '이익잉여금_평균 / 자산_평균 (유보이익비율)',
    'EBIT_TA': '영업손익 / 자산_평균 (영업이익비율)',
    'MVE_TL': '시가총액 / 부채_평균 (시장가치 대 부채비율)',
    'S_TA': '매출액 / 자산_평균 (자산회전율)',
    'CLCA': '유동부채_평균 / 유동자산_평균 (유동비율 역수)',
    'OENEG': '자산_평균 < 부채_평균 여부 (음의 자기자본 더미)',
    'CR': '유동자산_평균 / 유동부채_평균 (유동비율)',
    'CFO_TA': '영업현금흐름 / 자산_평균 (현금흐름비율)',
    'TLMTA': '부채_평균 / (시가총액 + 부채_평균) (시장가치 기준 부채비율)',
    'RET_3M': '최근 3개월 누적수익률',
    'RET_9M': '최근 9개월 누적수익률',
    'MB': '시가총액 / 자본_평균 (시장가치 대 장부가치 비율)'
}

for ratio in ratio_columns:
    if ratio in ratio_descriptions:
        print(f"{ratio:8} = {ratio_descriptions[ratio]}")

# 8. 기존 FS_ratio.csv와 비교
print(f"\n8. 기존 FS_ratio.csv와 비교:")
try:
    old_ratio = pd.read_csv('data/processed/FS_ratio.csv')
    print(f"기존 FS_ratio.csv: {old_ratio.shape}")
    print(f"개선된 FS_ratio_flow.csv: {final_df.shape}")
    
    # 공통 컬럼에 대한 비교
    common_cols = ['ROA', 'TLTA', 'WC_TA']
    for col in common_cols:
        if col in old_ratio.columns and col in final_df.columns:
            old_mean = old_ratio[col].mean()
            new_mean = final_df[col].mean()
            print(f"{col}: 기존 평균 {old_mean:.4f} → 개선 평균 {new_mean:.4f}")
    
except Exception as e:
    print(f"기존 파일 비교 실패: {e}")

print(f"\n샘플 데이터:")
print(final_df.head())

print(f"\n파일 크기: {os.path.getsize('data/processed/FS_ratio_flow.csv') / 1024 / 1024:.2f} MB")

print(f"\n🎯 FS_ratio_flow.csv 생성 완료!")
print("💡 주요 개선사항:")
print("- Stock 지표는 평균값 사용으로 더 정확한 비율 계산")
print("- Flow 지표와의 매칭 개선")
print("- 시계열적 일관성 향상")
print("- 재무비율의 경제적 의미 정확성 향상") 