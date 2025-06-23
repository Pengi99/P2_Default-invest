#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5단계: 고급 재무변수 생성 (FS_flow 기반)
- 성장성, 안정성, 수익성의 질, 효율성 변화, 가치평가 및 현금흐름 변수 생성
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== 5단계: 고급 재무변수 생성 (FS_flow 기반) ===")

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("\n1. 데이터 로드 중...")
    
    # FS_flow_with_borrow.csv 로드
    fs_flow = pd.read_csv('data/processed/FS_flow_with_borrow.csv', dtype={'거래소코드': str})
    print(f"FS_flow_with_borrow.csv: {fs_flow.shape}")
    
    # 기존 FS_ratio_flow.csv 로드
    fs_ratio = pd.read_csv('data/final/FS_ratio_flow.csv', dtype={'거래소코드': str})
    print(f"FS_ratio_flow.csv: {fs_ratio.shape}")
    
    print(f"FS_flow 컬럼: {list(fs_flow.columns)[:10]}...")
    print(f"FS_ratio 컬럼: {list(fs_ratio.columns)[:10]}...")
    
    return fs_flow, fs_ratio

def calculate_growth_variables(df):
    """성장성 변수 계산"""
    print("\n2. 성장성 변수 계산 중...")
    
    # 회사별 연도순 정렬
    df_sorted = df.sort_values(['거래소코드', '회계년도'])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 전년 대비 성장률 계산을 위한 shift
        df_sorted['자산_평균_lag'] = df_sorted.groupby('거래소코드')['자산_평균'].shift(1)
        df_sorted['이익잉여금_평균_lag'] = df_sorted.groupby('거래소코드')['이익잉여금_평균'].shift(1)
        df_sorted['영업손익_lag'] = df_sorted.groupby('거래소코드')['영업손익'].shift(1)
        df_sorted['당기순이익_lag'] = df_sorted.groupby('거래소코드')['당기순이익'].shift(1)
        
        # 성장성 변수 계산
        # 자산 YoY 성장률
        df_sorted['자산_YoY_성장률'] = (df_sorted['자산_평균'] - df_sorted['자산_평균_lag']) / df_sorted['자산_평균_lag']
        
        # 이익잉여금 YoY 성장률
        df_sorted['이익잉여금_YoY_성장률'] = (df_sorted['이익잉여금_평균'] - df_sorted['이익잉여금_평균_lag']) / df_sorted['이익잉여금_평균_lag']
        
        # 영업이익 YoY 성장률
        df_sorted['영업이익_YoY_성장률'] = (df_sorted['영업손익'] - df_sorted['영업손익_lag']) / df_sorted['영업손익_lag']
        
        # 순이익 YoY 성장률
        df_sorted['순이익_YoY_성장률'] = (df_sorted['당기순이익'] - df_sorted['당기순이익_lag']) / df_sorted['당기순이익_lag']
    
    # lag 컬럼 제거
    lag_cols = [col for col in df_sorted.columns if col.endswith('_lag')]
    df_sorted = df_sorted.drop(columns=lag_cols)
    
    growth_vars = ['자산_YoY_성장률', '이익잉여금_YoY_성장률', '영업이익_YoY_성장률', '순이익_YoY_성장률']
    for var in growth_vars:
        valid_count = df_sorted[var].notna().sum()
        print(f"  {var}: {valid_count:,}개")
    
    return df_sorted

def calculate_stability_risk_variables(df):
    """안정성 및 리스크 변수 계산"""
    print("\n3. 안정성 및 리스크 변수 계산 중...")
    
    # 회사별 연도순 정렬
    df_sorted = df.sort_values(['거래소코드', '회계년도'])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 기본 비율 계산
        df_sorted['부채비율'] = df_sorted['부채_평균'] / df_sorted['자본_평균']
        df_sorted['유동비율'] = df_sorted['유동자산_평균'] / df_sorted['유동부채_평균']
        
        # 전년 비율 계산
        df_sorted['부채비율_lag'] = df_sorted.groupby('거래소코드')['부채비율'].shift(1)
        df_sorted['유동비율_lag'] = df_sorted.groupby('거래소코드')['유동비율'].shift(1)
        
        # 변화량 계산
        df_sorted['부채비율_변화량'] = df_sorted['부채비율'] - df_sorted['부채비율_lag']
        df_sorted['유동비율_변화량'] = df_sorted['유동비율'] - df_sorted['유동비율_lag']
        
        # 의존도 계산
        df_sorted['단기부채_의존도'] = df_sorted['유동부채_평균'] / df_sorted['부채_평균']
        
        # 차입의존도
        df_sorted['차입_의존도'] = df_sorted['차입금의존도']
    
    # lag 컬럼 제거
    lag_cols = [col for col in df_sorted.columns if col.endswith('_lag')]
    df_sorted = df_sorted.drop(columns=lag_cols)
    
    stability_vars = ['부채비율_변화량', '유동비율_변화량', '단기부채_의존도', '차입_의존도']
    for var in stability_vars:
        valid_count = df_sorted[var].notna().sum()
        print(f"  {var}: {valid_count:,}개")
    
    return df_sorted

def calculate_profitability_quality_variables(df):
    """수익성의 질 및 변화 변수 계산"""
    print("\n4. 수익성의 질 및 변화 변수 계산 중...")
    
    # 회사별 연도순 정렬
    df_sorted = df.sort_values(['거래소코드', '회계년도'])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 발생액 (Accruals) = 당기순이익 - 영업현금흐름
        df_sorted['발생액'] = df_sorted['당기순이익'] - df_sorted['영업현금흐름']
        
        # 총이익률 계산 (매출원가 컬럼이 없으므로 매출액총이익률 사용)
        df_sorted['총이익률'] = df_sorted['매출액총이익률'] / 100  # 퍼센트를 비율로 변환
        
        # ROE 계산
        df_sorted['ROE'] = df_sorted['당기순이익'] / df_sorted['자본_평균']
        
        # 전년 비율 계산
        df_sorted['총이익률_lag'] = df_sorted.groupby('거래소코드')['총이익률'].shift(1)
        df_sorted['ROE_lag'] = df_sorted.groupby('거래소코드')['ROE'].shift(1)
        
        # 변화량 계산
        df_sorted['총이익률_변화량'] = df_sorted['총이익률'] - df_sorted['총이익률_lag']
        df_sorted['ROE_변화량'] = df_sorted['ROE'] - df_sorted['ROE_lag']
    
    # lag 컬럼 제거
    lag_cols = [col for col in df_sorted.columns if col.endswith('_lag')]
    df_sorted = df_sorted.drop(columns=lag_cols)
    
    quality_vars = ['발생액', '총이익률_변화량', 'ROE_변화량']
    for var in quality_vars:
        valid_count = df_sorted[var].notna().sum()
        print(f"  {var}: {valid_count:,}개")
    
    return df_sorted

def calculate_efficiency_change_variables(df):
    """효율성 변화 변수 계산"""
    print("\n5. 효율성 변화 변수 계산 중...")
    
    # 회사별 연도순 정렬
    df_sorted = df.sort_values(['거래소코드', '회계년도'])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 총자본회전률 계산
        df_sorted['총자본회전률'] = df_sorted['매출액'] / df_sorted['자산_평균']
        
        # 운전자본회전률 계산
        working_capital = df_sorted['유동자산_평균'] - df_sorted['유동부채_평균']
        df_sorted['운전자본회전률'] = df_sorted['매출액'] / working_capital
        
        # 전년 비율 계산
        df_sorted['총자본회전률_lag'] = df_sorted.groupby('거래소코드')['총자본회전률'].shift(1)
        df_sorted['운전자본회전률_lag'] = df_sorted.groupby('거래소코드')['운전자본회전률'].shift(1)
        
        # 변화량 계산
        df_sorted['총자본회전률_변화량'] = df_sorted['총자본회전률'] - df_sorted['총자본회전률_lag']
        df_sorted['운전자본회전률_변화량'] = df_sorted['운전자본회전률'] - df_sorted['운전자본회전률_lag']
    
    # lag 컬럼 제거
    lag_cols = [col for col in df_sorted.columns if col.endswith('_lag')]
    df_sorted = df_sorted.drop(columns=lag_cols)
    
    efficiency_vars = ['총자본회전률_변화량', '운전자본회전률_변화량']
    for var in efficiency_vars:
        valid_count = df_sorted[var].notna().sum()
        print(f"  {var}: {valid_count:,}개")
    
    return df_sorted

def calculate_valuation_cashflow_variables(df):
    """가치평가 및 현금흐름 심화 변수 계산"""
    print("\n6. 가치평가 및 현금흐름 심화 변수 계산 중...")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 이익수익률 (Earnings Yield) = 당기순이익 / 시가총액
        # 시가총액이 없으므로 자본 대비로 대체
        df['이익수익률'] = df['당기순이익'] / df['자본_평균']
        
        # 매출액 대비 현금흐름
        df['매출액_대비_현금흐름'] = df['영업현금흐름'] / df['매출액']
    
    valuation_vars = ['이익수익률', '매출액_대비_현금흐름']
    for var in valuation_vars:
        valid_count = df[var].notna().sum()
        print(f"  {var}: {valid_count:,}개")
    
    return df

def merge_with_existing_ratios(fs_flow_enhanced, fs_ratio):
    """기존 FS_ratio_flow.csv와 병합"""
    print("\n7. 기존 FS_ratio_flow.csv와 병합 중...")
    
    # 새로 생성된 변수들만 선택
    new_variables = [
        '회사명', '거래소코드', '회계년도',
        # 성장성
        '자산_YoY_성장률', '이익잉여금_YoY_성장률', '영업이익_YoY_성장률', '순이익_YoY_성장률',
        # 안정성 및 리스크
        '부채비율_변화량', '유동비율_변화량', '단기부채_의존도', '차입_의존도',
        # 수익성의 질 및 변화
        '발생액', '총이익률_변화량', 'ROE_변화량',
        # 효율성 변화
        '총자본회전률_변화량', '운전자본회전률_변화량',
        # 가치평가 및 현금흐름 심화
        '이익수익률', '매출액_대비_현금흐름'
    ]
    
    # 새 변수들만 추출
    new_vars_df = fs_flow_enhanced[new_variables].copy()
    
    # 기존 FS_ratio_flow와 병합
    merged_df = fs_ratio.merge(
        new_vars_df,
        on=['회사명', '거래소코드', '회계년도'],
        how='left',
        suffixes=('', '_new')
    )
    
    print(f"병합 전 FS_ratio_flow: {fs_ratio.shape}")
    print(f"병합 후: {merged_df.shape}")
    
    return merged_df

def clean_and_finalize_data(df):
    """데이터 정리 및 최종화"""
    print("\n8. 데이터 정리 및 최종화 중...")
    
    # 무한대 및 NaN 처리
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 극값 처리 (Winsorization)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 100:  # 충분한 데이터가 있는 경우만
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q01, upper=q99)
    
    print(f"극값 처리 완료")
    
    return df

def main():
    """메인 실행 함수"""
    try:
        # 1. 데이터 로드
        fs_flow, fs_ratio = load_and_prepare_data()
        
        # 2. 성장성 변수 계산
        fs_flow = calculate_growth_variables(fs_flow)
        
        # 3. 안정성 및 리스크 변수 계산
        fs_flow = calculate_stability_risk_variables(fs_flow)
        
        # 4. 수익성의 질 및 변화 변수 계산
        fs_flow = calculate_profitability_quality_variables(fs_flow)
        
        # 5. 효율성 변화 변수 계산
        fs_flow = calculate_efficiency_change_variables(fs_flow)
        
        # 6. 가치평가 및 현금흐름 심화 변수 계산
        fs_flow = calculate_valuation_cashflow_variables(fs_flow)
        
        # 7. 기존 FS_ratio_flow와 병합
        final_df = merge_with_existing_ratios(fs_flow, fs_ratio)
        
        # 8. 데이터 정리 및 최종화
        final_df = clean_and_finalize_data(final_df)
        
        # 9. 결과 저장
        print("\n9. 결과 저장 중...")
        output_path = 'data/final/FS_ratio_flow_enhanced.csv'
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n=== 5단계 완료 ===")
        print(f"고급 재무변수 추가 완료: {final_df.shape}")
        print(f"저장 위치: {output_path}")
        
        # 10. 결과 요약
        print(f"\n=== 새로 추가된 변수 요약 ===")
        
        new_vars = [
            '자산_YoY_성장률', '이익잉여금_YoY_성장률', '영업이익_YoY_성장률', '순이익_YoY_성장률',
            '부채비율_변화량', '유동비율_변화량', '단기부채_의존도', '차입_의존도',
            '발생액', '총이익률_변화량', 'ROE_변화량',
            '총자본회전률_변화량', '운전자본회전률_변화량',
            '이익수익률', '매출액_대비_현금흐름'
        ]
        
        print("\n📊 변수별 유효 데이터 개수:")
        for var in new_vars:
            if var in final_df.columns:
                valid_count = final_df[var].notna().sum()
                valid_pct = valid_count / len(final_df) * 100
                print(f"  {var:25s}: {valid_count:6,}개 ({valid_pct:5.1f}%)")
        
        print(f"\n📋 변수 분류:")
        print("🌱 성장성 (4개): 자산_YoY_성장률, 이익잉여금_YoY_성장률, 영업이익_YoY_성장률, 순이익_YoY_성장률")
        print("🛡️ 안정성 (4개): 부채비율_변화량, 유동비율_변화량, 단기부채_의존도, 차입_의존도")
        print("💎 수익성질 (3개): 발생액, 총이익률_변화량, ROE_변화량")
        print("⚡ 효율성 (2개): 총자본회전률_변화량, 운전자본회전률_변화량")
        print("💰 가치평가 (2개): 이익수익률, 매출액_대비_현금흐름")
        
        # 샘플 데이터 출력
        print(f"\n📋 샘플 데이터 (상위 5행):")
        sample_cols = ['회사명', '회계년도', '자산_YoY_성장률', '부채비율_변화량', '발생액']
        available_cols = [col for col in sample_cols if col in final_df.columns]
        sample_data = final_df[available_cols].head()
        print(sample_data.to_string(index=False))
        
        return final_df
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    
    if result is not None:
        print(f"\n🎯 성공!")
        print(f"📁 새로운 파일: data/final/FS_ratio_flow_enhanced.csv")
        print(f"📊 총 변수 개수: {len(result.columns)}개")
        print(f"📈 새로 추가된 변수: 16개")
    else:
        print(f"\n❌ 실패!") 