import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def convert_thousand_to_won(value):
    """천원 단위를 원 단위로 변환"""
    if pd.isna(value):
        return np.nan
    return value * 1000

def calculate_financial_ratios(df):
    """
    BS_ratio.csv에서 final.csv의 컬럼들을 계산
    
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
    
    # 4. 영업이익 (천원 -> 원)
    result_df['operating_income'] = df['[B420000000]* (정상)영업손익(보고서기재)(IFRS연결)(천원)'].apply(convert_thousand_to_won)
    
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
    
    # 재무비율 계산
    # 11. BPS (Book Value Per Share) = 총자본 / 발행주식수
    result_df['bps'] = np.where(
        result_df['shares_outstanding'] > 0,
        result_df['total_equity'] / result_df['shares_outstanding'],
        np.nan
    )
    
    # 12. EPS (Earnings Per Share) = 당기순이익 / 발행주식수
    result_df['eps'] = np.where(
        result_df['shares_outstanding'] > 0,
        result_df['net_income'] / result_df['shares_outstanding'],
        np.nan
    )
    
    # 13. 부채비율 (Debt to Equity Ratio) = 총부채 / 총자본 * 100
    result_df['debt_to_equity'] = np.where(
        result_df['total_equity'] > 0,
        (result_df['total_liabilities'] / result_df['total_equity']) * 100,
        np.nan
    )
    
    # 14. ROA (Return on Assets) = 당기순이익 / 총자산 * 100
    result_df['roa'] = np.where(
        result_df['total_assets'] > 0,
        (result_df['net_income'] / result_df['total_assets']) * 100,
        np.nan
    )
    
    # 15. ROE (Return on Equity) = 당기순이익 / 총자본 * 100
    result_df['roe'] = np.where(
        result_df['total_equity'] > 0,
        (result_df['net_income'] / result_df['total_equity']) * 100,
        np.nan
    )
    
    # 16. 영업이익률 (Operating Profit Margin) = 영업이익 / 매출액 * 100
    result_df['operating_profit_margin'] = np.where(
        result_df['revenue'] > 0,
        (result_df['operating_income'] / result_df['revenue']) * 100,
        0
    )
    
    # 17. 순이익률 (Net Profit Margin) = 당기순이익 / 매출액 * 100
    result_df['net_profit_margin'] = np.where(
        result_df['revenue'] > 0,
        (result_df['net_income'] / result_df['revenue']) * 100,
        0
    )
    
    # 18. PBR (Price to Book Ratio) - 기존 데이터에서 가져오기
    result_df['pbr'] = df['PBR(최저)(IFRS연결)']
    
    # 19. PER (Price to Earnings Ratio) - 기존 데이터에서 가져오기
    result_df['per'] = df['PER(최저)(IFRS연결)']
    
    # 20. CFO to Interest Expense Ratio
    result_df['cfo_to_interest_expense'] = np.where(
        result_df['interest_expense'] != 0,
        result_df['cfo'] / result_df['interest_expense'],
        np.nan
    )
    
    # 21. CFO to Total Debt Ratio
    result_df['cfo_to_total_debt'] = np.where(
        result_df['total_liabilities'] > 0,
        result_df['cfo'] / result_df['total_liabilities'],
        np.nan
    )
    
    # 22. CFO to Total Assets Ratio
    result_df['cfo_to_total_assets'] = np.where(
        result_df['total_assets'] > 0,
        result_df['cfo'] / result_df['total_assets'],
        np.nan
    )
    
    # final.csv와 동일한 컬럼 순서로 재배열
    final_columns = [
        '회사명', '거래소코드', '회계년도', 'bps', 'cfo', 'debt_to_equity', 
        'eps', 'interest_expense', 'net_income', 'net_profit_margin',
        'operating_income', 'operating_profit_margin', 'pbr', 'per', 
        'roa', 'roe', 'total_assets', 'total_equity', 'total_liabilities',
        'cfo_to_interest_expense', 'cfo_to_total_debt', 'cfo_to_total_assets'
    ]
    
    result_df = result_df[final_columns]
    
    return result_df

def main():
    """메인 실행 함수"""
    print("BS_ratio.csv에서 재무비율 계산 중...")
    
    # BS_ratio.csv 읽기
    try:
        df = pd.read_csv('../data/processed/BS_ratio.csv')
        print(f"데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}열")
    except FileNotFoundError:
        print("BS_ratio.csv 파일을 찾을 수 없습니다.")
        return
    
    # 재무비율 계산
    final_df = calculate_financial_ratios(df)
    
    # 결과 저장
    output_path = '../data/processed/final_calculated.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"계산 완료! 결과가 {output_path}에 저장되었습니다.")
    print(f"최종 데이터 형태: {final_df.shape[0]}행, {final_df.shape[1]}열")
    
    # 기본 통계 출력
    print("\n=== 기본 통계 ===")
    print(final_df.describe())
    
    # 결측치 확인
    print("\n=== 결측치 현황 ===")
    missing_data = final_df.isnull().sum()
    print(missing_data[missing_data > 0])

if __name__ == "__main__":
    main() 