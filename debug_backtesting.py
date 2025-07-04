"""
백테스팅 문제 진단 스크립트
-97% 수익률 문제 원인 분석
"""

import pandas as pd
import numpy as np

def debug_factor_computation(df):
    """팩터 계산 단계 디버깅"""
    print("🔍 팩터 계산 진단 시작...")
    
    print("\n1️⃣ 데이터 기본 정보")
    print(f"   - 총 행 수: {len(df):,}")
    print(f"   - 고유 종목 수: {df['거래소코드'].nunique()}")
    print(f"   - 날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
    
    print("\n2️⃣ Magic Formula 진단")
    if 'earnings_yield' in df.columns:
        ey_stats = df['earnings_yield'].describe()
        print(f"   - EY 통계: 최소={ey_stats['min']:.4f}, 최대={ey_stats['max']:.4f}")
        print(f"   - EY 유효값: {df['earnings_yield'].notna().sum():,}개")
        
        # 극단값 확인
        extreme_ey = df[df['earnings_yield'].abs() > 10]
        if len(extreme_ey) > 0:
            print(f"   ⚠️ 극단 EY 값 {len(extreme_ey)}개 발견 (절댓값 > 10)")
    
    if 'roic' in df.columns:
        roic_stats = df['roic'].describe()
        print(f"   - ROIC 통계: 최소={roic_stats['min']:.4f}, 최대={roic_stats['max']:.4f}")
        print(f"   - ROIC 유효값: {df['roic'].notna().sum():,}개")
    
    print("\n3️⃣ F-Score 진단")
    if 'fscore' in df.columns:
        fscore_dist = df['fscore'].value_counts().sort_index()
        print(f"   - F-Score 분포: {dict(fscore_dist)}")
        print(f"   - 평균 F-Score: {df['fscore'].mean():.2f}")
    
    print("\n4️⃣ 모멘텀 진단")
    if 'momentum' in df.columns:
        mom_stats = df['momentum'].describe()
        print(f"   - 모멘텀 통계: 최소={mom_stats['min']:.4f}, 최대={mom_stats['max']:.4f}")
        print(f"   - 모멘텀 유효값: {df['momentum'].notna().sum():,}개")
        
        # 극단값 확인
        extreme_mom = df[df['momentum'].abs() > 5]  # 500% 이상 변화
        if len(extreme_mom) > 0:
            print(f"   ⚠️ 극단 모멘텀 값 {len(extreme_mom)}개 발견 (절댓값 > 5)")

def debug_portfolio_construction(portfolios):
    """포트폴리오 구성 단계 디버깅"""
    print("\n🔍 포트폴리오 구성 진단 시작...")
    
    for strategy_name, strategy_portfolios in portfolios.items():
        print(f"\n📊 {strategy_name} 전략:")
        
        for universe, portfolio_list in strategy_portfolios.items():
            print(f"   {universe} 유니버스:")
            print(f"   - 리밸런싱 횟수: {len(portfolio_list)}")
            
            if len(portfolio_list) > 0:
                # 첫 번째와 마지막 포트폴리오 확인
                first_portfolio = portfolio_list[0]
                last_portfolio = portfolio_list[-1]
                
                print(f"   - 첫 리밸런싱: {first_portfolio['date']} ({len(first_portfolio['stocks'])}개 종목)")
                print(f"   - 마지막 리밸런싱: {last_portfolio['date']} ({len(last_portfolio['stocks'])}개 종목)")
                
                # 포트폴리오 크기 분포
                portfolio_sizes = [len(p['stocks']) for p in portfolio_list]
                print(f"   - 평균 포트폴리오 크기: {np.mean(portfolio_sizes):.1f}개")
                print(f"   - 포트폴리오 크기 범위: {min(portfolio_sizes)} ~ {max(portfolio_sizes)}개")
                
                # 빈 포트폴리오 확인
                empty_portfolios = sum(1 for p in portfolio_list if len(p['stocks']) == 0)
                if empty_portfolios > 0:
                    print(f"   ⚠️ 빈 포트폴리오 {empty_portfolios}개 발견")

def debug_price_data(price_data):
    """가격 데이터 진단"""
    print("\n🔍 가격 데이터 진단 시작...")
    
    print(f"   - 가격 데이터 행 수: {len(price_data):,}")
    print(f"   - 고유 종목 수: {price_data['거래소코드'].nunique()}")
    print(f"   - 날짜 범위: {price_data['date'].min()} ~ {price_data['date'].max()}")
    
    # 가격 컬럼 확인
    price_col = '종가' if '종가' in price_data.columns else '일간_시가총액'
    print(f"   - 사용 가격 컬럼: {price_col}")
    
    # 가격 통계
    price_stats = price_data[price_col].describe()
    print(f"   - 가격 통계: 최소={price_stats['min']:,.0f}, 최대={price_stats['max']:,.0f}")
    
    # 0 또는 음수 가격 확인
    invalid_prices = price_data[price_data[price_col] <= 0]
    if len(invalid_prices) > 0:
        print(f"   ⚠️ 유효하지 않은 가격 {len(invalid_prices)}개 발견 (≤ 0)")
    
    # 극단적 가격 변화 확인
    price_data_sorted = price_data.sort_values(['거래소코드', 'date'])
    price_data_sorted['price_change'] = price_data_sorted.groupby('거래소코드')[price_col].pct_change()
    
    extreme_changes = price_data_sorted[price_data_sorted['price_change'].abs() > 0.5]  # 50% 이상 변화
    if len(extreme_changes) > 0:
        print(f"   ⚠️ 극단적 가격 변화 {len(extreme_changes)}개 발견 (일일 50% 이상)")

def debug_backtest_results(backtest_results):
    """백테스팅 결과 진단"""
    print("\n🔍 백테스팅 결과 진단 시작...")
    
    for strategy_name, strategy_results in backtest_results.items():
        print(f"\n📊 {strategy_name} 전략:")
        
        for universe, results in strategy_results.items():
            if results is None:
                print(f"   {universe}: 결과 없음")
                continue
                
            print(f"   {universe} 유니버스:")
            
            portfolio_values = results.get('portfolio_values', [])
            daily_returns = results.get('daily_returns', [])
            
            if len(portfolio_values) == 0:
                print(f"   ⚠️ 포트폴리오 가치 데이터 없음")
                continue
            
            print(f"   - 데이터 포인트 수: {len(portfolio_values)}")
            print(f"   - 시작 가치: {portfolio_values[0]:,.0f}")
            print(f"   - 최종 가치: {portfolio_values[-1]:,.0f}")
            
            # 총 수익률 계산
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            print(f"   - 총 수익률: {total_return:.2%}")
            
            if total_return < -0.9:  # -90% 이하
                print(f"   🚨 극단적 손실 발생!")
                
                # 가치 변화 패턴 분석
                value_changes = np.diff(portfolio_values)
                large_drops = np.where(value_changes < -portfolio_values[0] * 0.1)[0]  # 초기 자본의 10% 이상 손실
                
                if len(large_drops) > 0:
                    print(f"   - 큰 손실 발생 횟수: {len(large_drops)}")
                    print(f"   - 첫 번째 큰 손실 시점: {large_drops[0]}일째")
            
            # 일일 수익률 통계
            if len(daily_returns) > 0:
                returns_stats = pd.Series(daily_returns).describe()
                print(f"   - 일일 수익률 평균: {returns_stats['mean']:.4f}")
                print(f"   - 일일 수익률 최소: {returns_stats['min']:.4f}")
                print(f"   - 일일 수익률 최대: {returns_stats['max']:.4f}")

def run_comprehensive_debug():
    """종합 진단 실행"""
    print("🔬 백테스팅 종합 진단 시작")
    print("=" * 60)
    
    print("⚠️ 주요 확인 포인트:")
    print("1. 팩터 값의 극단성 (EY, ROIC, 모멘텀)")
    print("2. 포트폴리오 구성 오류 (빈 포트폴리오, 크기)")
    print("3. 가격 데이터 품질 (0원, 극단 변화)")
    print("4. 백테스팅 로직 오류 (포지션 계산, 현금 관리)")
    print("5. 거래비용 과다 적용")
    print("6. 리밸런싱 로직 오류")
    print("=" * 60)
    
    return {
        'debug_factors': debug_factor_computation,
        'debug_portfolios': debug_portfolio_construction, 
        'debug_prices': debug_price_data,
        'debug_results': debug_backtest_results
    }

if __name__ == "__main__":
    print("백테스팅 진단 도구가 준비되었습니다.")
    print("사용법:")
    print("1. debug_functions = run_comprehensive_debug()")
    print("2. debug_functions['debug_factors'](master_df)")
    print("3. debug_functions['debug_portfolios'](portfolios)")
    print("4. debug_functions['debug_prices'](price_data)")
    print("5. debug_functions['debug_results'](backtest_results)")