import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def convert_thousand_to_won(value):
    """천원 단위를 원 단위로 변환"""
    if pd.isna(value):
        return np.nan
    return value * 1000

def calculate_altman_k2_score(total_assets, total_asset_turnover, retained_earnings_to_assets, market_equity_to_debt):
    """
    Altman K2 Score 계산 (한국형 부도예측모형)
    
    K2 = 4.30 * log(총자산) + 0.55 * (총자산회전율) + 0.78 * (이익잉여금/총자산) + 2.89 * (자기자본시장가치/총부채)
    
    판정기준:
    - K2 < -2.30: 부실가능성 심각
    - -2.30 ≤ K2 ≤ 0.75: 판정 유보 (회색지대)
    - K2 > 0.75: 부실가능성 낮음 (안전지대)
    
    Args:
        total_assets: 총자산 (원)
        total_asset_turnover: 총자산회전율 (매출액/총자산)
        retained_earnings_to_assets: 이익잉여금/총자산 비율
        market_equity_to_debt: 자기자본시장가치/총부채 비율
    
    Returns:
        k2_score: Altman K2 Score
    """
    if pd.isna(total_assets) or total_assets <= 0:
        return np.nan
    
    # 총자산의 자연로그 (억원 단위로 변환 후 로그)
    log_assets = np.log(total_assets / 100000000)  # 원을 억원으로 변환
    
    # K2 Score 계산
    k2_score = (4.30 * log_assets + 
                0.55 * total_asset_turnover + 
                0.78 * retained_earnings_to_assets + 
                2.89 * market_equity_to_debt)
    
    return k2_score

def classify_k2_score(k2_score):
    """
    K2 Score 기준으로 부실 위험도 분류
    
    Args:
        k2_score: Altman K2 Score
    
    Returns:
        classification: 분류 결과
    """
    if pd.isna(k2_score):
        return 'Unknown'
    elif k2_score < -2.30:
        return 'High Risk'
    elif k2_score <= 0.75:
        return 'Gray Zone'
    else:
        return 'Safe Zone'

def calculate_financial_ratios(df):
    """
    BS_ratio.csv에서 final.csv의 컬럼들을 계산하고 Altman K2 Score 추가
    
    Args:
        df: BS_ratio.csv 데이터프레임
    
    Returns:
        final_df: 계산된 재무비율이 포함된 데이터프레임
    """
    
    # 기본 컬럼들 (천원 단위를 원 단위로 변환)
    result_df = df[['회사명', '거래소코드', '회계년도']].copy()
    
    # 1. 총자산 (천원 -> 원)
    result_df['total_assets'] = df['[A100000000]자산(*)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 2. 총자본 (천원 -> 원)
    result_df['total_equity'] = df['[A600000000]자본(*)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 3. 총부채 (천원 -> 원)
    result_df['total_liabilities'] = df['[A800000000]부채(*)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 4. 영업이익 (계속영업이익 사용, 천원 -> 원)
    result_df['operating_income'] = df['[B800000000]계속영업이익(손실)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 5. 당기순이익 (천원 -> 원)
    result_df['net_income'] = df['[B840000000]당기순이익(손실)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 6. 영업현금흐름 (천원 -> 원)
    result_df['cfo'] = df['[D100000000]영업활동으로 인한 현금흐름(간접법)(*)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 7. 이자비용 (천원 -> 원) - 직접적인 이자비용 컬럼이 없으므로 0으로 설정
    result_df['interest_expense'] = 0
    
    # 8. 발행주식수 (주)
    result_df['shares_outstanding'] = df['[A600010300]   보통주(IFRS연결)(주)']
    
    # 9. 자본금 (천원 -> 원)
    result_df['capital'] = df['[A611000000]   자본금(*)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 10. 매출액 (영업손익을 매출액으로 사용, 천원 -> 원)
    result_df['revenue'] = df['[B420000000]* (정상)영업손익(보고서기재)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 11. 이익잉여금 (천원 -> 원) - K2 Score 계산용
    result_df['retained_earnings'] = df['[A615000000]      이익잉여금(결손금)(*)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
    # 재무비율 계산
    # 12. BPS (Book Value Per Share) = 총자본 / 발행주식수
    result_df['bps'] = np.where(
        result_df['shares_outstanding'] > 0,
        result_df['total_equity'] / result_df['shares_outstanding'],
        np.nan
    )
    
    # 13. EPS (Earnings Per Share) = 당기순이익 / 발행주식수
    result_df['eps'] = np.where(
        result_df['shares_outstanding'] > 0,
        result_df['net_income'] / result_df['shares_outstanding'],
        np.nan
    )
    
    # 14. 부채비율 (Debt to Equity Ratio) = 총부채 / 총자본
    result_df['debt_to_equity'] = np.where(
        result_df['total_equity'] > 0,
        (result_df['total_liabilities'] / result_df['total_equity']),
        np.nan
    )

    # 14. 총자산부채비율 (Debt to Assets Ratio) = 총부채 / 총자산
    result_df['debt_to_assets'] = np.where(
        result_df['total_assets'] > 0,
        (result_df['total_liabilities'] / result_df['total_assets']),
        np.nan
    )
    
    # 15. ROA (Return on Assets) = 당기순이익 / 총자산
    result_df['roa'] = np.where(
        result_df['total_assets'] > 0,
        (result_df['net_income'] / result_df['total_assets']),
        np.nan
    )
    
    # 16. ROE (Return on Equity) = 당기순이익 / 총자본
    result_df['roe'] = np.where(
        result_df['total_equity'] > 0,
        (result_df['net_income'] / result_df['total_equity']),
        np.nan
    )
    
    # 17. 영업이익률 (Operating Profit Margin) = 영업이익 / 매출액
    result_df['operating_profit_margin'] = np.where(
        result_df['revenue'] > 0,
        (result_df['operating_income'] / result_df['revenue']),
        0
    )
    
    # 18. 순이익률 (Net Profit Margin) = 당기순이익 / 매출액
    result_df['net_profit_margin'] = np.where(
        result_df['revenue'] > 0,
        (result_df['net_income'] / result_df['revenue']),
        0
    )
    
    # 19. 총자산회전율 (Total Asset Turnover) = 매출액 / 총자산
    result_df['total_asset_turnover'] = np.where(
        result_df['total_assets'] > 0,
        result_df['revenue'] / result_df['total_assets'],
        np.nan
    )
    
    # 20. 이익잉여금/총자산 비율
    result_df['retained_earnings_to_assets'] = np.where(
        result_df['total_assets'] > 0,
        result_df['retained_earnings'] / result_df['total_assets'],
        np.nan
    )
    
    # 21. 자기자본시장가치/총부채 비율 (PBR을 이용하여 근사치 계산)
    # 자기자본시장가치 = 자기자본장부가치 * PBR
    result_df['market_equity'] = result_df['total_equity'] * df['PBR(최저)(IFRS연결)'].fillna(1.0)
    result_df['market_equity_to_debt'] = np.where(
        result_df['total_liabilities'] > 0,
        result_df['market_equity'] / result_df['total_liabilities'],
        np.nan
    )
    
    # 22. Altman K2 Score 계산
    result_df['altman_k2_score'] = result_df.apply(
        lambda row: calculate_altman_k2_score(
            row['total_assets'],
            row['total_asset_turnover'],
            row['retained_earnings_to_assets'],
            row['market_equity_to_debt']
        ), axis=1
    )
    
    # 23. K2 Score 분류
    result_df['k2_risk_level'] = result_df['altman_k2_score'].apply(classify_k2_score)
    
    # 24. PBR (Price to Book Ratio) - 기존 데이터에서 가져오기
    result_df['pbr'] = df['PBR(최저)(IFRS연결)']
    
    # 25. PER (Price to Earnings Ratio) - 기존 데이터에서 가져오기
    result_df['per'] = df['PER(최저)(IFRS연결)']
    
    # 26. CFO to Interest Expense Ratio
    result_df['cfo_to_interest_expense'] = np.where(
        result_df['interest_expense'] != 0,
        result_df['cfo'] / result_df['interest_expense'],
        np.nan
    )
    
    # 27. CFO to Total Debt Ratio
    result_df['cfo_to_total_debt'] = np.where(
        result_df['total_liabilities'] > 0,
        result_df['cfo'] / result_df['total_liabilities'],
        np.nan
    )
    
    # 28. CFO to Total Assets Ratio
    result_df['cfo_to_total_assets'] = np.where(
        result_df['total_assets'] > 0,
        result_df['cfo'] / result_df['total_assets'],
        np.nan
    )
    
    # 컬럼 설정 딕셔너리 (활성화/비활성화 및 설명 포함)
    column_config = {
        # 기본 정보
        '회사명': {'active': True, 'category': 'basic', 'description': '회사명'},
        '거래소코드': {'active': True, 'category': 'basic', 'description': '거래소 코드'},
        '회계년도': {'active': True, 'category': 'basic', 'description': '회계년도'},
        
        # Altman K2 Score (부도예측모형)
        'altman_k2_score': {'active': True, 'category': 'bankruptcy_prediction', 'description': 'Altman K2 Score (한국형 부도예측모형)'},
        'k2_risk_level': {'active': True, 'category': 'bankruptcy_prediction', 'description': 'K2 Score 기반 위험도 분류'},
        
        # 주당 지표
        'bps': {'active': True, 'category': 'per_share', 'description': 'Book Value Per Share (주당순자산가치)'},
        'eps': {'active': True, 'category': 'per_share', 'description': 'Earnings Per Share (주당순이익)'},
        
        # 현금흐름
        'cfo': {'active': True, 'category': 'cash_flow', 'description': 'Cash Flow from Operations (영업현금흐름)'},
        
        # 레버리지 지표
        'debt_to_equity': {'active': True, 'category': 'leverage', 'description': 'Debt to Equity Ratio (부채비율)'},
        'debt_to_assets': {'active': True, 'category': 'leverage', 'description': 'Debt to Assets Ratio (총자산부채비율)'},
        
        # 수익성 지표
        'roa': {'active': True, 'category': 'profitability', 'description': 'Return on Assets (총자산수익률)'},
        'roe': {'active': True, 'category': 'profitability', 'description': 'Return on Equity (자기자본수익률)'},
        'operating_profit_margin': {'active': True, 'category': 'profitability', 'description': 'Operating Profit Margin (영업이익률)'},
        'net_profit_margin': {'active': True, 'category': 'profitability', 'description': 'Net Profit Margin (순이익률)'},
        
        # 활동성 지표
        'total_asset_turnover': {'active': True, 'category': 'activity', 'description': 'Total Asset Turnover (총자산회전율)'},
        'retained_earnings_to_assets': {'active': True, 'category': 'activity', 'description': 'Retained Earnings to Assets (이익잉여금/총자산)'},
        'market_equity_to_debt': {'active': True, 'category': 'activity', 'description': 'Market Equity to Debt (자기자본시장가치/총부채)'},
        
        # 시장 지표
        'pbr': {'active': True, 'category': 'market', 'description': 'Price to Book Ratio (주가순자산비율)'},
        'per': {'active': True, 'category': 'market', 'description': 'Price to Earnings Ratio (주가수익비율)'},
        
        # 절대값 지표
        'net_income': {'active': True, 'category': 'absolute', 'description': 'Net Income (당기순이익)'},
        'operating_income': {'active': True, 'category': 'absolute', 'description': 'Operating Income (영업이익)'},
        'total_assets': {'active': True, 'category': 'absolute', 'description': 'Total Assets (총자산)'},
        'total_equity': {'active': True, 'category': 'absolute', 'description': 'Total Equity (총자본)'},
        'total_liabilities': {'active': True, 'category': 'absolute', 'description': 'Total Liabilities (총부채)'},
        'retained_earnings': {'active': True, 'category': 'absolute', 'description': 'Retained Earnings (이익잉여금)'},
        'market_equity': {'active': True, 'category': 'absolute', 'description': 'Market Value of Equity (자기자본시장가치)'},
        
        # 현금흐름 비율 (문제가 있는 지표들)
        'interest_expense': {'active': False, 'category': 'problematic', 'description': 'Interest Expense (이자비용) - 데이터 없음'},
        'cfo_to_interest_expense': {'active': False, 'category': 'problematic', 'description': 'CFO to Interest Expense Ratio - 이자비용 데이터 없어 NaN'},
        'cfo_to_total_debt': {'active': True, 'category': 'cash_flow', 'description': 'CFO to Total Debt Ratio (현금흐름/총부채 비율)'},
        'cfo_to_total_assets': {'active': True, 'category': 'cash_flow', 'description': 'CFO to Total Assets Ratio (현금흐름/총자산 비율)'},
    }
    
    # 활성화된 컬럼만 선택
    final_columns = [col for col, config in column_config.items() if config['active']]
    
    # 활성화된 컬럼 정보 출력 (카테고리별로 정리)
    print("\n=== 활성화된 컬럼 정보 (카테고리별) ===")
    categories = {}
    for col in final_columns:
        if col in column_config:
            category = column_config[col]['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((col, column_config[col]['description']))
    
    for category, columns in categories.items():
        print(f"\n[{category.upper()}]")
        for col, desc in columns:
            print(f"  {col}: {desc}")
    
    # 비활성화된 컬럼 정보 출력
    inactive_columns = [col for col, config in column_config.items() if not config['active']]
    if inactive_columns:
        print("\n=== 비활성화된 컬럼 정보 ===")
        for col in inactive_columns:
            print(f"  {col}: {column_config[col]['description']}")
    
    result_df = result_df[final_columns]
    
    return result_df, column_config

def main():
    """메인 실행 함수"""
    print("BS_ratio.csv에서 재무비율 계산 및 Altman K2 Score 생성 중...")
    
    # BS_ratio.csv 읽기
    try:
        df = pd.read_csv('/Users/jojongho/KDT/P2_Default-invest/data/processed/BS_ratio.csv')
        print(f"데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}열")
    except FileNotFoundError:
        print("BS_ratio.csv 파일을 찾을 수 없습니다.")
        return
    
    # 재무비율 계산
    final_df, column_config = calculate_financial_ratios(df)
    
    # 결과 저장
    output_path = '/Users/jojongho/KDT/P2_Default-invest/data/processed/final_calculated.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"계산 완료! 결과가 {output_path}에 저장되었습니다.")
    print(f"최종 데이터 형태: {final_df.shape[0]}행, {final_df.shape[1]}열")
    
    # Altman K2 Score 통계 출력
    print("\n=== Altman K2 Score 통계 ===")
    k2_scores = final_df['altman_k2_score'].dropna()
    if len(k2_scores) > 0:
        print(f"K2 Score 평균: {k2_scores.mean():.3f}")
        print(f"K2 Score 표준편차: {k2_scores.std():.3f}")
        print(f"K2 Score 최소값: {k2_scores.min():.3f}")
        print(f"K2 Score 최대값: {k2_scores.max():.3f}")
        
        # 위험도별 분포
        print("\n=== K2 Score 위험도별 분포 ===")
        risk_distribution = final_df['k2_risk_level'].value_counts()
        for risk_level, count in risk_distribution.items():
            percentage = count / len(final_df) * 100
            print(f"{risk_level}: {count}개 ({percentage:.1f}%)")
    
    # 기본 통계 출력
    print("\n=== 기본 통계 ===")
    print(final_df.describe())
    
    # 결측치 확인
    print("\n=== 결측치 현황 ===")
    missing_data = final_df.isnull().sum()
    print(missing_data[missing_data > 0])

if __name__ == "__main__":
    main() 