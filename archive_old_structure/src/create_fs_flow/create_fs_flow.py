import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== Stock 개념 지표를 Flow 개념으로 변환 ===")

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
fs_data = pd.read_csv('data/processed/FS_filtered.csv')
print(f"FS_filtered.csv: {fs_data.shape}")
print(f"컬럼: {list(fs_data.columns)}")

# 2. Stock 개념 지표 정의
print("\n2. Stock 개념 지표 정의...")
stock_indicators = [
    '자산',           # 총자산
    '자본',           # 자기자본  
    '부채',           # 총부채
    '유동부채',       # 유동부채
    '유동자산',       # 유동자산
    '발행주식수',     # 발행주식수
    '자본금',         # 자본금
    '이익잉여금'      # 이익잉여금
]

# Flow 개념 지표 (그대로 유지)
flow_indicators = [
    '매출액',         # 매출액
    '영업손익',       # 영업손익
    '당기순이익',     # 당기순이익
    '영업현금흐름',   # 영업현금흐름
    '매출액증가율',   # 매출액증가율
    '매출액총이익률', # 매출액총이익률
    '매출액정상영업이익률', # 매출액정상영업이익률
    '매출액순이익률', # 매출액순이익률
    '총자본순이익률', # 총자본순이익률
    '자기자본순이익률', # 자기자본순이익률
    '유동비율',       # 유동비율
    '부채비율',       # 부채비율
    '이자보상배율',   # 이자보상배율
    '총자본회전률',   # 총자본회전률
    '기업가치',       # 기업가치
    'EBITDA',         # EBITDA
    'EV_EBITDA_비율'  # EV/EBITDA 비율
]

print(f"Stock 개념 지표: {len(stock_indicators)}개")
for indicator in stock_indicators:
    print(f"  - {indicator}")

print(f"\nFlow 개념 지표: {len(flow_indicators)}개 (그대로 유지)")

# 3. 데이터 전처리
print("\n3. 데이터 전처리 중...")

# 회계년도에서 연도 추출
fs_data['연도'] = fs_data['회계년도'].str[:4].astype(int)
print(f"연도 범위: {fs_data['연도'].min()} ~ {fs_data['연도'].max()}")

# 회사별 정렬 (거래소코드, 연도 순)
fs_data = fs_data.sort_values(['거래소코드', '연도']).reset_index(drop=True)
print(f"정렬 완료: {fs_data.shape}")

# 4. Stock 지표 평균값 계산
print("\n4. Stock 지표 평균값 계산 중...")

# 결과 데이터프레임 생성 (기본 정보 + Flow 지표는 그대로)
result_df = fs_data[['회사명', '거래소코드', '회계년도', '연도'] + flow_indicators].copy()

# Stock 지표들을 평균값으로 변환
for stock_col in stock_indicators:
    if stock_col in fs_data.columns:
        print(f"  처리 중: {stock_col}")
        
        # 전기 데이터와 매칭을 위한 준비
        fs_data[f'{stock_col}_전기'] = fs_data.groupby('거래소코드')[stock_col].shift(1)
        
        # 평균값 계산: (전기말 + 당기말) / 2
        result_df[f'{stock_col}_평균'] = (fs_data[f'{stock_col}_전기'] + fs_data[stock_col]) / 2
        
        # 첫 해 데이터는 당기말 값 그대로 사용 (전기 데이터가 없으므로)
        first_year_mask = fs_data.groupby('거래소코드')['연도'].transform('min') == fs_data['연도']
        result_df.loc[first_year_mask, f'{stock_col}_평균'] = fs_data.loc[first_year_mask, stock_col]
        
        # 원본 당기말 값도 보존
        result_df[f'{stock_col}_당기말'] = fs_data[stock_col]
    else:
        print(f"  ⚠️ 컬럼 없음: {stock_col}")

# 5. 데이터 품질 확인
print("\n5. 데이터 품질 확인 중...")

# 평균값 계산 결과 통계
print("Stock 지표별 평균값 계산 결과:")
print(f"{'지표명':<12} {'유효개수':>8} {'평균':>12} {'표준편차':>12}")
print("-" * 50)

for stock_col in stock_indicators:
    if stock_col in fs_data.columns:
        avg_col = f'{stock_col}_평균'
        if avg_col in result_df.columns:
            valid_count = result_df[avg_col].notna().sum()
            mean_val = result_df[avg_col].mean()
            std_val = result_df[avg_col].std()
            print(f"{stock_col:<12} {valid_count:>8,} {mean_val:>12,.0f} {std_val:>12,.0f}")

# 6. 연도별 데이터 개수 확인
print("\n6. 연도별 데이터 개수 확인:")
yearly_counts = result_df.groupby('연도').size()
print(yearly_counts)

# 7. 샘플 데이터 확인
print("\n7. 샘플 데이터 확인:")
print("첫 5개 기업의 자산 평균값 계산 예시:")
sample_cols = ['회사명', '거래소코드', '연도', '자산_당기말', '자산_평균']
if all(col in result_df.columns for col in sample_cols):
    sample_data = result_df[sample_cols].head(10)
    print(sample_data.to_string(index=False))

# 8. 최종 저장
print("\n8. 최종 저장 중...")

# 컬럼 순서 정리
basic_info = ['회사명', '거래소코드', '회계년도', '연도']
stock_avg_cols = [f'{col}_평균' for col in stock_indicators if col in fs_data.columns]
stock_current_cols = [f'{col}_당기말' for col in stock_indicators if col in fs_data.columns]
flow_cols = [col for col in flow_indicators if col in result_df.columns]

final_columns = basic_info + stock_avg_cols + stock_current_cols + flow_cols
final_df = result_df[final_columns].copy()

# 저장
final_df.to_csv('data/processed/FS_flow.csv', index=False, encoding='utf-8-sig')

print(f"\n=== 완료 ===")
print(f"최종 데이터 크기: {final_df.shape}")
print(f"저장 위치: data/processed/FS_flow.csv")

# 9. 결과 요약
print(f"\n=== 결과 요약 ===")
print(f"✅ 처리된 Stock 지표: {len(stock_avg_cols)}개")
print(f"✅ 보존된 Flow 지표: {len(flow_cols)}개")
print(f"✅ 총 컬럼 수: {len(final_columns)}개")

print(f"\n📊 변환된 Stock 지표 목록:")
for i, col in enumerate(stock_avg_cols, 1):
    original_name = col.replace('_평균', '')
    print(f"{i:2d}. {original_name} → {col}")

print(f"\n💡 사용법:")
print("- Stock 지표 평균값: 전기말과 당기말의 평균 ((전기말 + 당기말) / 2)")
print("- Flow 지표와의 비율 계산에 적합")
print("- 예: ROA = 당기순이익 / 자산_평균")
print("- 예: 부채비율 = 부채_평균 / 자산_평균")

print(f"\n파일 크기: {pd.read_csv('data/processed/FS_flow.csv').memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# 10. 검증 예시
print(f"\n10. 검증 예시 - 특정 기업의 자산 변화:")
if '자산_평균' in final_df.columns and '자산_당기말' in final_df.columns:
    # 첫 번째 기업의 연도별 자산 변화
    first_company = final_df['거래소코드'].iloc[0]
    company_data = final_df[final_df['거래소코드'] == first_company].head(5)
    
    if len(company_data) > 1:
        print(f"기업: {company_data['회사명'].iloc[0]} ({first_company})")
        print(f"{'연도':<6} {'당기말자산':>12} {'평균자산':>12} {'차이':>12}")
        print("-" * 50)
        for _, row in company_data.iterrows():
            당기말 = row['자산_당기말']
            평균 = row['자산_평균']
            차이 = 평균 - 당기말 if pd.notna(평균) and pd.notna(당기말) else np.nan
            print(f"{row['연도']:<6} {당기말:>12,.0f} {평균:>12,.0f} {차이:>12,.0f}")

print(f"\n🎯 FS_flow.csv 생성 완료! Flow 개념 지표와의 비율 계산에 활용하세요.") 