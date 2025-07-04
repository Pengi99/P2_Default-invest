"""
빠른 백테스팅 테스트 - 작은 데이터셋으로 -97% 이슈 확인
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def quick_backtest_test():
    """작은 데이터셋으로 빠른 백테스팅 테스트"""
    print("🚀 빠른 백테스팅 테스트 시작...")
    
    try:
        # Import after environment setup
        sys.path.append('.')
        os.chdir('/Users/jojongho/KDT/P2_Default-invest')
        
        # Load libraries with fallbacks
        try:
            import polars as pl
            print("✅ Polars 사용 가능")
        except ImportError:
            print("⚠️ Polars 미사용 - Pandas fallback")
        
        try:
            from numba import jit
            print("✅ Numba 사용 가능")
        except ImportError:
            print("⚠️ Numba 미사용 - Python fallback")
        
        # Load config
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Override config for quick test
        config['start_date'] = '2022-01-01'  # Test with recent 2 years only
        config['end_date'] = '2023-12-31'
        config['portfolio_params']['portfolio_size'] = 5  # Small portfolio
        
        print(f"📊 테스트 기간: {config['start_date']} ~ {config['end_date']}")
        print(f"🎯 포트폴리오 크기: {config['portfolio_params']['portfolio_size']}")
        
        # Now import the main classes
        from factor_backtesting_v2 import DataHandler, FactorEngine
        
        # 1. Load data
        print("\n📈 데이터 로딩...")
        data_handler = DataHandler(config)
        data_handler.load_data()
        
        if data_handler.master_df is None or len(data_handler.master_df) == 0:
            print("❌ 마스터 데이터프레임 생성 실패")
            return False
        
        print(f"✅ 마스터 데이터: {len(data_handler.master_df):,}행")
        
        # Check date range in master data
        if 'date' in data_handler.master_df.columns:
            date_range = f"{data_handler.master_df['date'].min()} ~ {data_handler.master_df['date'].max()}"
            print(f"   날짜 범위: {date_range}")
        
        # 2. Quick factor calculation test
        print("\n🧮 팩터 계산 테스트...")
        factor_engine = FactorEngine(config)
        
        # Test with small subset
        test_df = data_handler.master_df.head(1000).copy()
        print(f"📊 테스트 데이터: {len(test_df)}행")
        
        # Calculate factors for test data
        test_df = factor_engine.compute_factors(test_df)
        
        if 'fscore' in test_df.columns:
            fscore_stats = test_df['fscore'].describe()
            print(f"✅ F-Score 계산 성공:")
            print(f"   평균: {fscore_stats['mean']:.2f}")
            print(f"   범위: {fscore_stats['min']:.0f} ~ {fscore_stats['max']:.0f}")
            
            # Check for reasonable values
            if fscore_stats['min'] >= 0 and fscore_stats['max'] <= 9:
                print("✅ F-Score 값 정상 범위")
            else:
                print("⚠️ F-Score 값 이상 - 추가 확인 필요")
        else:
            print("❌ F-Score 계산 실패")
            return False
        
        # 3. Test Magic Formula (already computed in compute_factors)
        try:
            if 'earnings_yield' in test_df.columns and 'roic' in test_df.columns:
                ey_stats = test_df['earnings_yield'].describe()
                roic_stats = test_df['roic'].describe()
                
                print(f"✅ Magic Formula 계산 성공:")
                print(f"   EY 범위: {ey_stats['min']:.4f} ~ {ey_stats['max']:.4f}")
                print(f"   ROIC 범위: {roic_stats['min']:.4f} ~ {roic_stats['max']:.4f}")
                
                # Check for extreme values
                if abs(ey_stats['min']) > 100 or abs(ey_stats['max']) > 100:
                    print("⚠️ Earnings Yield 극단값 발견")
                if abs(roic_stats['min']) > 100 or abs(roic_stats['max']) > 100:
                    print("⚠️ ROIC 극단값 발견")
            else:
                print("❌ Magic Formula 계산 실패")
        except Exception as e:
            print(f"❌ Magic Formula 오류: {e}")
        
        print("\n🎯 초기 진단 완료")
        print("주요 발견사항:")
        
        # Check for potential issues
        issues_found = []
        
        # Check data availability in test period
        if 'date' in data_handler.master_df.columns:
            test_period_data = data_handler.master_df[
                (data_handler.master_df['date'] >= config['start_date']) &
                (data_handler.master_df['date'] <= config['end_date'])
            ]
            
            if len(test_period_data) == 0:
                issues_found.append("테스트 기간에 데이터 없음")
            else:
                print(f"   - 테스트 기간 데이터: {len(test_period_data):,}행")
                
                # Check unique companies
                unique_companies = test_period_data['거래소코드'].nunique()
                print(f"   - 고유 기업 수: {unique_companies}")
                
                if unique_companies < 10:
                    issues_found.append(f"기업 수 부족 ({unique_companies}개)")
        
        # Check price data availability
        if hasattr(data_handler, 'daily_price_df') and data_handler.daily_price_df is not None:
            price_data_size = len(data_handler.daily_price_df)
            print(f"   - 일일 가격 데이터: {price_data_size:,}행")
            
            if price_data_size == 0:
                issues_found.append("가격 데이터 없음")
        else:
            issues_found.append("가격 데이터 로딩 실패")
        
        if issues_found:
            print("\n⚠️ 발견된 문제:")
            for issue in issues_found:
                print(f"   - {issue}")
            return False
        else:
            print("\n✅ 기본 데이터 및 팩터 계산 정상")
            print("실제 백테스팅 실행 준비 완료")
            return True
            
    except Exception as e:
        print(f"\n💥 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_backtest_test()
    if success:
        print("\n🎉 빠른 테스트 통과 - 실제 백테스팅 실행 권장")
    else:
        print("\n❌ 테스트 실패 - 추가 디버깅 필요")
    
    sys.exit(0 if success else 1)