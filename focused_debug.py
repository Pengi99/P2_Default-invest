"""
Focused debugging script for -97% returns issue
"""

import pandas as pd
import numpy as np
import os

def analyze_data_files():
    """Analyze the raw data files for potential issues"""
    print("🔍 원본 데이터 파일 분석...")
    
    # Check processed data
    processed_file = "data/processed/FS2_no_default.csv"
    if os.path.exists(processed_file):
        df = pd.read_csv(processed_file)
        print(f"\n📊 FS2_no_default.csv:")
        print(f"   - 행 수: {len(df):,}")
        print(f"   - 컬럼 수: {len(df.columns)}")
        print(f"   - 날짜 범위: {df['date'].min()} ~ {df['date'].max()}" if 'date' in df.columns else "   - 날짜 컬럼 없음")
        
        # Check for key financial metrics
        financial_cols = ['총자산', '당기순이익', '영업활동으로인한현금흐름', '시가총액']
        for col in financial_cols:
            if col in df.columns:
                col_stats = df[col].describe()
                print(f"   - {col}: 최소={col_stats['min']:,.0f}, 최대={col_stats['max']:,.0f}")
                
                # Check for negative or zero values in key metrics
                if col in ['총자산', '시가총액']:
                    invalid_count = (df[col] <= 0).sum()
                    if invalid_count > 0:
                        print(f"     ⚠️ {col}에서 0 이하 값 {invalid_count}개 발견")
        
        # Check recent data availability
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            recent_data = df[df['date'] >= '2023-01-01']
            print(f"   - 2023년 이후 데이터: {len(recent_data):,}행")
            
            if len(recent_data) == 0:
                print("     🚨 최근 데이터 부족 - 백테스팅 기간 문제 가능성")
    
    else:
        print(f"❌ {processed_file} 파일을 찾을 수 없습니다")

def analyze_price_files():
    """Analyze price data files"""
    print("\n🔍 가격 데이터 파일 분석...")
    
    price_files = [
        "data/final/merged_daily_data_2020_2023.csv",
        "data/final/merged_daily_data_2024.csv"
    ]
    
    total_records = 0
    date_ranges = []
    
    for file_path in price_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            total_records += len(df)
            
            print(f"\n📊 {os.path.basename(file_path)}:")
            print(f"   - 행 수: {len(df):,}")
            print(f"   - 고유 종목 수: {df['거래소코드'].nunique()}" if '거래소코드' in df.columns else "   - 거래소코드 컬럼 없음")
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                date_ranges.append((df['date'].min(), df['date'].max()))
                print(f"   - 날짜 범위: {df['date'].min().date()} ~ {df['date'].max().date()}")
            
            # Check price columns
            price_cols = ['종가', '시가총액', '일간_시가총액']
            for col in price_cols:
                if col in df.columns:
                    valid_prices = df[df[col] > 0][col]
                    if len(valid_prices) > 0:
                        print(f"   - {col}: 최소={valid_prices.min():,.0f}, 최대={valid_prices.max():,.0f}")
                        
                        # Check for extreme price changes
                        df_sorted = df.sort_values(['거래소코드', 'date'])
                        df_sorted['price_change'] = df_sorted.groupby('거래소코드')[col].pct_change()
                        extreme_changes = df_sorted[df_sorted['price_change'].abs() > 0.5]
                        
                        if len(extreme_changes) > 0:
                            print(f"     ⚠️ 일일 50% 이상 가격 변화: {len(extreme_changes)}건")
                    
                    zero_prices = (df[col] <= 0).sum()
                    if zero_prices > 0:
                        print(f"     ⚠️ {col}에서 0 이하 가격: {zero_prices}건")
        else:
            print(f"❌ {file_path} 파일을 찾을 수 없습니다")
    
    print(f"\n📈 가격 데이터 총계:")
    print(f"   - 총 레코드: {total_records:,}")
    if date_ranges:
        all_start = min(d[0] for d in date_ranges)
        all_end = max(d[1] for d in date_ranges)
        print(f"   - 전체 날짜 범위: {all_start.date()} ~ {all_end.date()}")

def check_optimization_issues():
    """Check for issues introduced by optimizations"""
    print("\n🔍 최적화 관련 이슈 체크...")
    
    # Check if optimized libraries are available
    try:
        import polars as pl
        print("✅ Polars 사용 가능")
    except ImportError:
        print("❌ Polars 불가 - Pandas 사용")
    
    try:
        from numba import jit
        print("✅ Numba 사용 가능")
    except ImportError:
        print("❌ Numba 불가 - 순수 Python 사용")
    
    try:
        import dask
        print("✅ Dask 사용 가능")
    except ImportError:
        print("❌ Dask 불가")

def identify_likely_issues():
    """최신 최적화 이후 가능한 문제점들"""
    print("\n🚨 가능한 문제점들:")
    print("1. Polars 변환 시 데이터 타입 변경 또는 손실")
    print("2. Numba JIT 모멘텀 계산에서 수치 오류")
    print("3. 벡터화된 가격 조회에서 인덱싱 오류")
    print("4. 멀티프로세싱 시 데이터 분할/병합 문제")
    print("5. 포트폴리오 리밸런싱 로직 오류")
    print("6. 거래비용 과다 적용")
    print("7. 포지션 크기 계산 오류")

if __name__ == "__main__":
    print("🔬 -97% 수익률 문제 집중 분석")
    print("=" * 50)
    
    analyze_data_files()
    analyze_price_files()
    check_optimization_issues()
    identify_likely_issues()
    
    print("\n🎯 다음 단계:")
    print("1. 최적화 이전 백업 버전과 결과 비교")
    print("2. 각 최적화 단계별 개별 테스트")
    print("3. 작은 데이터셋으로 디버깅")
    print("4. 포트폴리오 가치 변화 추적")