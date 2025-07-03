"""
한국 상장기업 부도예측 데이터셋을 위한 피처 엔지니어링 파이프라인
==============================================================================

이 스크립트는 재무제표 데이터와 시장 데이터를 결합하여 부도예측 모델링에 
적합한 데이터셋을 만들기 위한 종합적인 피처 엔지니어링을 수행합니다.

입력 파일:
- FS2.csv: 마스터 재무제표 파일
- 2012.csv ~ 2023.csv: 연도별 주가/시장 데이터 파일
- 시가총액.csv: 시가총액 데이터

출력:
- FS2_features.csv: 엔지니어링된 피처들을 포함한 결합 데이터셋
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import warnings

def safe_divide(numerator, denominator, default=np.nan, outlier_threshold=1000):
    """
    안전한 나눗셈 함수 - 0으로 나누기 방지 및 이상치 처리
    
    Args:
        numerator: 분자
        denominator: 분모
        default: 분모가 0일 때 반환할 기본값
        outlier_threshold: 이상치로 간주할 절댓값 임계치
        
    Returns:
        안전하게 계산된 비율 (이상치는 NaN 처리)
    """
    # numpy 배열로 변환
    num = np.asarray(numerator)
    den = np.asarray(denominator)
    
    # 분모가 0이거나 매우 작은 경우 처리
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.where(
            (den == 0) | (np.abs(den) < 1e-10), 
            default, 
            num / den
        )
    
    # 이상치 처리 (절댓값이 임계치보다 큰 경우)
    result = np.where(np.abs(result) > outlier_threshold, np.nan, result)
    
    # inf, -inf 처리
    result = np.where(np.isinf(result), np.nan, result)
    
    return result

def validate_features(df):
    """
    피처 품질 검증 및 리포트 생성
    
    Args:
        df: 검증할 DataFrame
        
    Returns:
        validation_report: 검증 결과 딕셔너리
    """
    print("피처 품질 검증 중...")
    
    validation_report = {
        'negative_assets': [],
        'extreme_ratios': [],
        'high_missing_rate': [],
        'suspicious_values': []
    }
    
    # 1. 음수가 되면 안 되는 값들 체크
    asset_cols = ['총자산', '유동자산', '비유동자산', '유형자산', '무형자산', 
                  '재고자산', '현금및현금성자산', '시가총액']
    
    for col in asset_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                validation_report['negative_assets'].append({
                    'column': col,
                    'negative_count': negative_count,
                    'percentage': negative_count / len(df) * 100
                })
    
    # 2. 극단적인 비율 체크
    ratio_thresholds = {
        '총자산수익률(ROA)': (-1, 1),        # -100% ~ 100%
        '자기자본수익률(ROE)': (-3, 3),       # -300% ~ 300%
        '부채자산비율': (0, 2),               # 0% ~ 200%
        '유동비율': (0, 50),                  # 0 ~ 50배
        'PER': (-1000, 1000),                # -1000 ~ 1000배
        'PBR': (0, 100),                     # 0 ~ 100배
        '총자산회전율': (0, 10)               # 0 ~ 10회
    }
    
    for col, (min_val, max_val) in ratio_thresholds.items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                validation_report['extreme_ratios'].append({
                    'column': col,
                    'out_of_range_count': out_of_range,
                    'percentage': out_of_range / len(df) * 100,
                    'range': (min_val, max_val)
                })
    
    # 3. 높은 결측치 비율 체크 (80% 이상)
    missing_rates = df.isnull().sum() / len(df) * 100
    high_missing = missing_rates[missing_rates > 80]
    
    for col in high_missing.index:
        validation_report['high_missing_rate'].append({
            'column': col,
            'missing_percentage': high_missing[col]
        })
    
    # 4. 의심스러운 값들 체크
    suspicious_patterns = {
        '총자산이 0인 경우': (df.get('총자산', pd.Series()) == 0).sum(),
        '매출액이 음수인 경우': (df.get('매출액', pd.Series()) < 0).sum(),
        '시가총액이 0인 경우': (df.get('시가총액', pd.Series()) == 0).sum()
    }
    
    for pattern, count in suspicious_patterns.items():
        if count > 0:
            validation_report['suspicious_values'].append({
                'pattern': pattern,
                'count': count,
                'percentage': count / len(df) * 100
            })
    
    # 검증 결과 출력
    print("\n=== 피처 품질 검증 결과 ===")
    
    if validation_report['negative_assets']:
        print("\n⚠️ 음수 자산 발견:")
        for item in validation_report['negative_assets']:
            print(f"  - {item['column']}: {item['negative_count']}개 ({item['percentage']:.2f}%)")
    
    if validation_report['extreme_ratios']:
        print("\n⚠️ 극단적인 비율 발견:")
        for item in validation_report['extreme_ratios']:
            print(f"  - {item['column']}: {item['out_of_range_count']}개 ({item['percentage']:.2f}%) "
                  f"범위 외: {item['range']}")
    
    if validation_report['high_missing_rate']:
        print("\n⚠️ 높은 결측치 비율 (>80%):")
        for item in validation_report['high_missing_rate']:
            print(f"  - {item['column']}: {item['missing_percentage']:.2f}%")
    
    if validation_report['suspicious_values']:
        print("\n⚠️ 의심스러운 값들:")
        for item in validation_report['suspicious_values']:
            print(f"  - {item['pattern']}: {item['count']}개 ({item['percentage']:.2f}%)")
    
    if not any(validation_report.values()):
        print("✅ 모든 피처가 품질 기준을 통과했습니다!")
    
    return validation_report



def load_and_merge_data():
    """
    모든 입력 데이터 파일을 로드하고 병합
    """
    print("데이터 파일 로딩 중...")
    
    # 프로젝트 루트 디렉토리 경로 설정
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # src/data_processing -> src -> project_root
    
    # FS2.csv 로드
    fs2_path = project_root / "data" / "processed" / "FS2.csv"
    if fs2_path.exists():
        df = pd.read_csv(fs2_path, encoding='utf-8-sig')
    else:
        # 대안 경로 시도
        fs2_path = project_root / "FS2.csv"
        df = pd.read_csv(fs2_path, encoding='utf-8-sig')
    
    # 회계년도를 연도로 통일
    if '회계년도' in df.columns:
        df = df.rename(columns={'회계년도': '연도'})
    
    # 거래소코드를 문자형으로 통일 (병합 문제 해결)
    df['거래소코드'] = df['거래소코드'].astype(str)
    
    print(f"FS2.csv 로드 완료: {len(df)} 행")
    print(f"FS2 연도 범위: {df['연도'].min()} ~ {df['연도'].max()}")
    print(f"FS2 거래소코드 샘플: {df['거래소코드'].head().values}")
    
    # 연도별 주가 파일 로드 (2012.csv ~ 2023.csv)
    yearly_data = []
    data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        data_dir = project_root
    
    for year in range(2012, 2024):
        year_file = data_dir / f"{year}.csv"
        if year_file.exists():
            year_df = pd.read_csv(year_file, encoding='utf-8-sig')
            
            # 실제 컬럼명 확인 및 표준화 (첫 번째 파일만 출력)
            if year == 2012:
                print(f"연도별 파일 컬럼: {list(year_df.columns)}")
                print(f"주가 데이터 회계년도 샘플: {year_df['회계년도'].head().values}")
            
            # 거래소코드를 문자형으로 통일
            year_df['거래소코드'] = year_df['거래소코드'].astype(str)
            
            # 컬럼명 매핑 (주가 파일에서 회계년도는 실제로는 거래일자임)
            column_mapping = {
                '매매년월일': '일자',
                '종목코드(축약)': '종목코드',
                '시가총액(원)': '시가총액_temp',
                '상장주식수(주)': '상장주식수',
                '종가(원)': '종가',
                '거래량(주)': '거래량'
                # 회계년도는 거래일자이므로 제외
            }
            
            # 실제 존재하는 컬럼만 매핑
            available_mapping = {old: new for old, new in column_mapping.items() if old in year_df.columns}
            year_df = year_df.rename(columns=available_mapping)
            
            # 일자 컬럼 설정 (매매년월일 또는 회계년도 사용)
            if '일자' not in year_df.columns:
                if '매매년월일' in year_df.columns:
                    year_df['일자'] = year_df['매매년월일']
                elif '회계년도' in year_df.columns:
                    year_df['일자'] = year_df['회계년도']
                else:
                    year_df['일자'] = pd.to_datetime(f'{year}-12-31')
            
            # 파일명에서 연도 추출 (가장 중요!)
            year_df['연도'] = year
            
            # 연말 데이터 추출 (마지막 거래일)
            year_df['일자'] = pd.to_datetime(year_df['일자'])
            
            # 거래소코드별로 마지막 거래일 데이터 선택
            if '거래소코드' in year_df.columns:
                year_end_df = year_df.loc[year_df.groupby('거래소코드')['일자'].idxmax()].copy()
            else:
                # 거래소코드가 없으면 전체 데이터의 마지막 날짜만 선택
                max_date = year_df['일자'].max()
                year_end_df = year_df[year_df['일자'] == max_date].copy()
            
            # 필요한 컬럼들 선택 (실제 존재하는 컬럼만)
            price_cols = ['거래소코드', '연도', '종가', '거래량', '상장주식수']
            available_cols = [col for col in price_cols if col in year_end_df.columns]
            
            # 주당배당금이 있다면 추가
            if '주당배당금' in year_end_df.columns:
                available_cols.append('주당배당금')
            
            if available_cols and len(year_end_df) > 0:
                yearly_data.append(year_end_df[available_cols])
                print(f"{year}.csv 로드 완료: {len(year_end_df)} 기업")
                
                # 첫 번째 연도에서 데이터 확인
                if year == 2012:
                    print(f"  거래소코드 샘플: {year_end_df['거래소코드'].head().values}")
                    print(f"  연도 확인: {year_end_df['연도'].unique()}")
    
    # 모든 연도별 데이터 결합
    if yearly_data:
        price_df = pd.concat(yearly_data, ignore_index=True)
        
        # 데이터 타입 통일
        price_df['연도'] = pd.to_numeric(price_df['연도'], errors='coerce').astype('Int64')
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce').astype('Int64')
        
        print(f"결합된 주가 데이터: {len(price_df)} 행")
        print(f"주가 데이터 연도 범위: {price_df['연도'].min()} ~ {price_df['연도'].max()}")
        
        # 병합 전 공통 키 확인
        fs2_keys = set(df[['거래소코드', '연도']].apply(tuple, axis=1))
        price_keys = set(price_df[['거래소코드', '연도']].apply(tuple, axis=1))
        common_keys = fs2_keys.intersection(price_keys)
        
        print(f"FS2 키 개수: {len(fs2_keys)}")
        print(f"주가 데이터 키 개수: {len(price_keys)}")
        print(f"공통 키 개수: {len(common_keys)}")
        
        # FS2와 병합
        if '거래소코드' in price_df.columns and '거래소코드' in df.columns:
            df = df.merge(price_df, on=['거래소코드', '연도'], how='left')
            print(f"주가 데이터 병합 후: {len(df)} 행")
            
            # 병합 결과 확인
            merged_price_data = df[['종가', '거래량', '상장주식수']].notna().sum()
            print(f"병합된 주가 데이터 개수:")
            for col, count in merged_price_data.items():
                print(f"  {col}: {count}개 ({count/len(df)*100:.2f}%)")
        else:
            print("⚠️ 거래소코드 컬럼이 없어 주가 데이터 병합을 건너뜁니다.")
    
    # 시가총액 데이터 처리 (우선주 포함한 정확한 시가총액 사용)
    print("시가총액 데이터 처리 중...")
    
    # 기존 시가총액 컬럼이 있다면 제거 (시가총액.csv의 정확한 값 사용을 위해)
    if '시가총액' in df.columns:
        original_na_count = df['시가총액'].isna().sum()
        df = df.drop(columns=['시가총액'])
        print(f"⚠️ 기존 시가총액 컬럼 제거 (결측치 {original_na_count:,}개 포함)")
    
    # 시가총액.csv 파일 로드 (우선주 포함한 정확한 시가총액)
    market_cap_path = project_root / "data" / "processed" / "시가총액.csv"
    if not market_cap_path.exists():
        market_cap_path = project_root / "시가총액.csv"
    
    market_cap_loaded = False
    if market_cap_path.exists():
        try:
            market_cap_df = pd.read_csv(market_cap_path, encoding='utf-8-sig')
            if '회계년도' in market_cap_df.columns:
                market_cap_df = market_cap_df.rename(columns={'회계년도': '연도'})
            
            # 거래소코드 타입 통일
            market_cap_df['거래소코드'] = market_cap_df['거래소코드'].astype(str)
            
            # 데이터 타입 통일
            if '연도' in market_cap_df.columns:
                market_cap_df['연도'] = pd.to_numeric(market_cap_df['연도'], errors='coerce').astype('Int64')
            
            if '거래소코드' in market_cap_df.columns and '거래소코드' in df.columns:
                df = df.merge(market_cap_df[['거래소코드', '연도', '시가총액']], 
                             on=['거래소코드', '연도'], how='left')
                market_cap_loaded = True
                print(f"✅ 시가총액.csv 데이터 병합 완료: {len(df)} 행")
                
                # 병합 후 시가총액 통계
                total_count = len(df)
                valid_count = df['시가총액'].notna().sum()
                print(f"   시가총액 데이터: {valid_count:,}/{total_count:,} ({valid_count/total_count*100:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ 시가총액.csv 로딩 실패: {e}")
    
    if not market_cap_loaded:
        print("⚠️ 시가총액.csv를 찾을 수 없거나 로딩에 실패했습니다.")
        # 빈 시가총액 컬럼 생성
        df['시가총액'] = np.nan
    
    # 시가총액 결측치 보완 (종가 × 상장주식수 또는 발행주식총수)
    missing_cap = df['시가총액'].isna()
    missing_count = missing_cap.sum()
    
    if missing_count > 0:
        print(f"⚠️ 시가총액 결측치 {missing_count:,}개 발견 - 보간 시도")
        
        # 상장주식수를 우선 사용, 없으면 발행주식총수 사용
        share_col = None
        if '상장주식수' in df.columns:
            share_col = '상장주식수'
        elif '발행주식총수' in df.columns:
            share_col = '발행주식총수'
        
        if '종가' in df.columns and share_col:
            # 종가와 주식수가 모두 있는 경우만 계산
            calc_mask = missing_cap & df['종가'].notna() & df[share_col].notna()
            df.loc[calc_mask, '시가총액'] = df.loc[calc_mask, '종가'] * df.loc[calc_mask, share_col]
            filled_count = calc_mask.sum()
            print(f"✅ 종가 × {share_col}로 {filled_count:,}개 값 보간 완료")
            
            # 최종 결측치 확인
            final_missing = df['시가총액'].isna().sum()
            if final_missing > 0:
                print(f"⚠️ 최종 시가총액 결측치: {final_missing:,}개")
        else:
            print(f"❌ 시가총액 보간 불가: 종가 또는 주식수 데이터 부족")
    else:
        print("✅ 시가총액 결측치 없음")
    
    # 거래소코드, 연도 순으로 정렬
    df = df.sort_values(['거래소코드', '연도']).reset_index(drop=True)
    
    return df

def create_balance_sheet_averages(df):
    """
    적절한 플로우 컨벤션을 위한 대차대조표 항목들의 평균값 생성
    """
    print("대차대조표 플로우 평균값 생성 중...")
    
    bs_cols = [
        '총자산','유동자산','유형자산','무형자산','재고자산','현금및현금성자산',
        '비유동자산','총자본','발행주식총수','총부채','유동부채','비유동부채',
        '장기차입금','단기차입금','매출채권','매입채무','선수수익',
        '기타유동부채','기타유동자산','(금융)리스부채','단기금융상품(금융기관예치금)',
        '자본금','기타포괄손익누계액'
    ]
    
    # 대차대조표 항목들에 대한 avg_ 컬럼 생성
    for col in bs_cols:
        if col in df.columns:
            df[f'avg_{col}'] = df.groupby('거래소코드')[col].transform(
                lambda x: (x + x.shift(1)) / 2
            )
    
    return df

def create_growth_features(df):
    """
    전년 대비 성장률 피처 생성
    """
    print("성장률 피처 생성 중...")
    
    growth_cols = ['매출액', '영업이익', '당기순이익', '총자산', '총부채', '영업현금흐름']
    
    for col in growth_cols:
        if col in df.columns:
            df[f'{col}증가율'] = df.groupby('거래소코드')[col].pct_change()
    
    return df

def create_liquidity_leverage_features(df):
    """
    유동성 및 레버리지 비율 생성 (B/S vs B/S, 기말 잔액 사용)
    """
    print("유동성 및 레버리지 피처 생성 중...")
    
    # 유동성 비율
    if '유동자산' in df.columns and '유동부채' in df.columns:
        df['유동비율'] = safe_divide(df['유동자산'], df['유동부채'], outlier_threshold=100)
        df['당좌비율'] = safe_divide(
            df['유동자산'] - df.get('재고자산', 0), 
            df['유동부채'], 
            outlier_threshold=100
        )
        df['운전자본'] = df['유동자산'] - df['유동부채']
    
    # 레버리지 비율
    if '총부채' in df.columns and '총자산' in df.columns:
        df['부채자산비율'] = safe_divide(df['총부채'], df['총자산'], outlier_threshold=10)
    
    if '총부채' in df.columns and '총자본' in df.columns:
        df['부채자본비율'] = safe_divide(df['총부채'], df['총자본'], outlier_threshold=50)
        df['부채총자본비율'] = safe_divide(
            df['총부채'], 
            df['총부채'] + df['총자본'], 
            outlier_threshold=1
        )
    
    if '총자본' in df.columns and '총자산' in df.columns:
        df['자본비율'] = safe_divide(df['총자본'], df['총자산'], outlier_threshold=10)
    
    # 이자부담부채 비율
    debt_cols = ['단기차입금', '장기차입금']
    if all(col in df.columns for col in debt_cols) and '총자산' in df.columns:
        df['이자부담차입금비율'] = safe_divide(
            df['단기차입금'] + df['장기차입금'], 
            df['총자산'], 
            outlier_threshold=10
        )
    
    if '장기차입금' in df.columns and all(col in df.columns for col in ['총부채', '총자본']):
        df['장기차입금자본비율'] = safe_divide(
            df['장기차입금'], 
            df['총부채'] + df['총자본'], 
            outlier_threshold=5
        )
    
    # 영업운전자본 변형
    working_capital_cols = ['매출채권', '재고자산', '기타유동자산', '매입채무', '선수수익', '기타유동부채']
    if all(col in df.columns for col in working_capital_cols):
        df['영업운전자본'] = (df['매출채권'] + df['재고자산'] + df['기타유동자산']) - \
                          (df['매입채무'] + df['선수수익'] + df['기타유동부채'])
    
    # 투하자본
    cash_cols = ['현금및현금성자산', '단기금융상품(금융기관예치금)']
    if all(col in df.columns for col in ['총부채', '총자본']) and any(col in df.columns for col in cash_cols):
        cash_sum = sum(df.get(col, 0) for col in cash_cols if col in df.columns)
        df['투하자본'] = df['총부채'] + df['총자본'] - cash_sum
    
    # 경영자본
    if all(col in df.columns for col in ['유형자산', '무형자산']) and '영업운전자본' in df.columns:
        df['경영자본'] = df['유형자산'] + df['무형자산'] + df['영업운전자본']
    
    return df

def create_additional_balance_sheet_averages(df):
    """
    피처 생성 과정에서 새로 만들어진 대차대조표 항목들의 평균값 생성
    """
    print("추가 대차대조표 항목 평균값 생성 중...")
    
    # 새로 생성된 BS 항목들 (피처 엔지니어링 과정에서 생성)
    additional_bs_items = ['투하자본', '경영자본', '운전자본', '영업운전자본']
    
    # 추가 BS 항목들에 대한 avg_ 컬럼 생성
    created_count = 0
    for col in additional_bs_items:
        if col in df.columns:
            avg_col_name = f'avg_{col}'
            if avg_col_name not in df.columns:  # 중복 생성 방지
                df[avg_col_name] = df.groupby('거래소코드')[col].transform(
                    lambda x: (x + x.shift(1)) / 2
                )
                created_count += 1
                print(f"  생성됨: {avg_col_name}")
    
    if created_count == 0:
        print("  새로 생성할 평균 BS 항목이 없습니다.")
    else:
        print(f"  총 {created_count}개 추가 평균 BS 항목 생성 완료")
    
    return df

def create_profitability_features(df):
    """
    수익성 및 현금흐름 비율 생성 (I/S 또는 C/F vs avg_B/S)
    """
    print("수익성 피처 생성 중...")
    
    # ROA, ROE
    if '당기순이익' in df.columns:
        if 'avg_총자산' in df.columns:
            df['총자산수익률(ROA)'] = safe_divide(df['당기순이익'], df['avg_총자산'], outlier_threshold=5)
        if 'avg_총자본' in df.columns:
            df['자기자본수익률(ROE)'] = safe_divide(df['당기순이익'], df['avg_총자본'], outlier_threshold=10)
    
    # 자산회전율
    if '매출액' in df.columns and 'avg_총자산' in df.columns:
        df['총자산회전율'] = safe_divide(df['매출액'], df['avg_총자산'], outlier_threshold=20)
    
    # 부채커버리지 비율
    if '영업이익' in df.columns:
        debt_avg_cols = ['avg_단기차입금', 'avg_장기차입금']
        if all(col in df.columns for col in debt_avg_cols):
            total_debt = df['avg_단기차입금'] + df['avg_장기차입금']
            df['영업이익대차입금비율'] = safe_divide(df['영업이익'], total_debt, outlier_threshold=100)
    
    # EBITDA 커버리지
    ebitda_cols = ['영업이익', '감가상각비', '무형자산상각비']
    if all(col in df.columns for col in ebitda_cols):
        ebitda = df['영업이익'] + df['감가상각비'] + df['무형자산상각비']
        debt_avg_cols = ['avg_단기차입금', 'avg_장기차입금']
        if all(col in df.columns for col in debt_avg_cols):
            total_debt = df['avg_단기차입금'] + df['avg_장기차입금']
            df['EBITDA대차입금비율'] = safe_divide(ebitda, total_debt, outlier_threshold=100)
    
    # 현금흐름 비율
    if '영업현금흐름' in df.columns and 'avg_총부채' in df.columns:
        df['현금흐름대부채비율'] = safe_divide(df['영업현금흐름'], df['avg_총부채'], outlier_threshold=5)
    
    # ROIC
    if '영업이익' in df.columns and 'avg_투하자본' in df.columns:
        df['투하자본수익률(ROIC)'] = safe_divide(df['영업이익'], df['avg_투하자본'], outlier_threshold=5)
    
    return df

def create_leverage_metrics(df):
    """
    지정된 대로 DOL과 DFL 레버리지 지표만 생성
    """
    print("레버리지 지표 생성 중 (DOL과 DFL만)...")
    
    # DOL - 영업레버리지
    if all(col in df.columns for col in ['매출액', '영업이익']):
        # 성장률이 create_growth_features에서 이미 생성되어야 함
        if '매출액증가율' in df.columns and '영업이익증가율' in df.columns:
            df['DOL'] = safe_divide(df['영업이익증가율'], df['매출액증가율'], outlier_threshold=50)
        else:
            # 존재하지 않으면 생성
            df['매출액증가율'] = df.groupby('거래소코드')['매출액'].pct_change()
            df['영업이익증가율'] = df.groupby('거래소코드')['영업이익'].pct_change()
            df['DOL'] = safe_divide(df['영업이익증가율'], df['매출액증가율'], outlier_threshold=50)
    
    # DFL - 재무레버리지
    if all(col in df.columns for col in ['영업이익', '이자비용']):
        denominator = df['영업이익'] - df['이자비용']
        df['DFL'] = safe_divide(df['영업이익'], denominator, outlier_threshold=50)
    
    # 선택사항: 델타 피처
    for metric in ['DFL']:
        if metric in df.columns:
            df[f'Δ{metric}'] = df.groupby('거래소코드')[metric].diff()
    
    return df

def create_market_features(df):
    """
    시장 밸류에이션 피처 생성
    """
    print("시장 밸류에이션 피처 생성 중...")
    
    # 시가총액 피처
    if '시가총액' in df.columns:
        df['로그시가총액'] = np.log1p(df['시가총액'])
        # df['시가총액증가율'] = df.groupby('거래소코드')['시가총액'].pct_change()
    
    # 총자산 로그 변환
    if '총자산' in df.columns:
        df['로그총자산'] = np.log1p(df['총자산'])
    
    # 기업가치(Enterprise Value)
    cash_cols = ['현금및현금성자산', '단기금융상품(금융기관예치금)']
    if '시가총액' in df.columns and '총부채' in df.columns:
        cash_sum = sum(df.get(col, 0) for col in cash_cols if col in df.columns)
        df['EV'] = df['시가총액'] + df['총부채'] - cash_sum
        
        # EV/EBITDA 비율
        ebitda_cols = ['영업이익', '감가상각비', '무형자산상각비']
        if all(col in df.columns for col in ebitda_cols):
            ebitda = df['영업이익'] + df['감가상각비'] + df['무형자산상각비']
            df['EV/EBITDA'] = safe_divide(df['EV'], ebitda, outlier_threshold=500)
    
    # 밸류에이션 배수
    if '시가총액' in df.columns:
        if '총자본' in df.columns:
            df['PBR'] = safe_divide(df['시가총액'], df['총자본'], outlier_threshold=100)
        
        if '당기순이익' in df.columns:
            df['PER'] = safe_divide(df['시가총액'], df['당기순이익'], outlier_threshold=1000)
        
        if '매출액' in df.columns:
            df['PSR'] = safe_divide(df['시가총액'], df['매출액'], outlier_threshold=100)
        
        if '영업현금흐름' in df.columns:
            df['PCR'] = safe_divide(df['시가총액'], df['영업현금흐름'], outlier_threshold=1000)
    
    # 밸류에이션 배수의 델타 피처
    # for multiple in ['PBR', 'PER', 'PSR', 'PCR']:
    #     if multiple in df.columns:
    #         df[f'Δ{multiple}'] = df.groupby('거래소코드')[multiple].diff()
    
    return df

def create_dividend_features(df):
    """
    배당 및 성장 피처 생성
    """
    print("배당 피처 생성 중...")
    
    # 배당수익률
    if all(col in df.columns for col in ['주당배당금', '종가']):
        df['배당수익률'] = safe_divide(df['주당배당금'], df['종가'], outlier_threshold=1)
    
    # PEG 비율
    # if all(col in df.columns for col in ['PER', '당기순이익증가율']):
    #     # 성장률을 백분율로 변환 후 계산
    #     growth_rate_pct = df['당기순이익증가율'] * 100
    #     df['PEG'] = safe_divide(df['PER'], growth_rate_pct, outlier_threshold=10)
    
    return df

def create_momentum_features(df):
    """
    모멘텀, 변동성, 유동성 피처 생성
    """
    print("모멘텀 및 유동성 피처 생성 중...")
    
    # 주의: 연간 데이터의 경우 모멘텀 계산이 제한적임
    # 일일/월간 데이터에서 더 의미가 있음
    
    # if '종가' in df.columns:
    #     # 연간 수익률 (연간 데이터로는 제한적 활용)
    #     for period in [1, 2, 3]:  # 1, 2, 3년 수익률
    #         df[f'{period}년수익률'] = df.groupby('거래소코드')['종가'].pct_change(periods=period)
    
    # 주식 회전율
    if all(col in df.columns for col in ['거래량', '상장주식수']):
        df['주식회전율'] = safe_divide(df['거래량'], df['상장주식수'], outlier_threshold=10)
    
    return df

def create_delta_features(df):
    """
    주요 비율들의 델타(변화량) 피처 생성
    """
    print("델타 피처 생성 중...")
    
    # 주요 수익성 지표의 전년 대비 변화량 (명시적 생성)
    profitability_ratios = {
        '총자산수익률(ROA)': 'ROA_변화량',
        '자기자본수익률(ROE)': 'ROE_변화량', 
        '투하자본수익률(ROIC)': 'ROIC_변화량'
    }
    
    for original_col, new_col in profitability_ratios.items():
        if original_col in df.columns:
            df[new_col] = df.groupby('거래소코드')[original_col].diff()
            print(f"  생성됨: {new_col}")
    
    # 델타 피처를 생성할 주요 비율들
    delta_candidates = [
        '유동비율', '당좌비율', '부채자산비율', '부채자본비율', '자본비율',
        '총자산회전율', 'EBITDA대차입금비율', '현금흐름대부채비율',
        '배당수익률'
    ]
    
    for ratio in delta_candidates:
        if ratio in df.columns:
            df[f'Δ{ratio}'] = df.groupby('거래소코드')[ratio].diff()
    
    return df

def clean_numerical_data(df):
    """
    수치 데이터 정리 - 무한값을 NaN으로 치환
    """
    print("수치 데이터 정리 중...")
    
    # inf와 -inf를 NaN으로 치환
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print(f"최종 데이터셋 크기: {df.shape}")
    print(f"총 NaN 값 개수: {df.isna().sum().sum()}")
    
    return df

def remove_first_year_data(df):
    """
    각 기업별 첫해 데이터 삭제 (부도 기업 제외)
    - 부도 기업(default=1)의 데이터는 보존
    - 정상 기업의 첫해 데이터만 삭제하여 전년 대비 비교 가능한 데이터만 유지
    """
    print("각 기업별 첫해 데이터 삭제 중 (부도 기업 제외)...")
    
    original_count = len(df)
    print(f"전처리 전 데이터: {original_count:,}개")
    
    if 'default' in df.columns:
        # 전처리 전 부도 기업 현황
        default_before = df[df['default'] == 1].shape[0]
        print(f"부도 기업 데이터: {default_before:,}개 ({default_before/original_count*100:.2f}%)")
        
        # 부도 기업이 아닌 데이터에서만 첫해 찾기
        non_default_df = df[df['default'] != 1]
        first_year_by_code = non_default_df.groupby('거래소코드')['연도'].min()
        print(f"첫해 삭제 대상 기업 수 (부도 기업 제외): {len(first_year_by_code):,}개")
        
        # 첫해 데이터 식별 (부도 기업이 아닌 경우만)
        first_year_mask = df.apply(
            lambda row: (row['default'] != 1) and 
                       (row['거래소코드'] in first_year_by_code) and
                       (row['연도'] == first_year_by_code.get(row['거래소코드'], float('inf'))), 
            axis=1
        )
        
        first_year_count = first_year_mask.sum()
        print(f"각 기업별 첫해 데이터 (삭제 대상): {first_year_count:,}개")
        
        # 첫해 데이터 삭제
        df_final = df[~first_year_mask].copy()
        
        # 결과 요약
        final_count = len(df_final)
        default_after = df_final[df_final['default'] == 1].shape[0]
        default_loss = default_before - default_after
        
        print(f"첫해 데이터 삭제 후: {final_count:,}개")
        print(f"전체 삭제율: {(original_count - final_count)/original_count*100:.2f}%")
        print(f"부도 기업 보존: {default_after:,}개 (손실: {default_loss:,}개)")
        print(f"부도율 변화: {default_before/original_count*100:.3f}% → {default_after/final_count*100:.3f}%")
        
    else:
        # default 컬럼이 없는 경우 기존 로직 사용
        print("⚠️ default 컬럼이 없어 모든 기업의 첫해 데이터를 삭제합니다.")
        first_year_by_code = df.groupby('거래소코드')['연도'].min()
        print(f"분석 대상 기업 수: {len(first_year_by_code):,}개")
        
        first_year_mask = df.apply(
            lambda row: row['연도'] == first_year_by_code[row['거래소코드']], 
            axis=1
        )
        
        first_year_count = first_year_mask.sum()
        print(f"각 기업별 첫해 데이터 (삭제 대상): {first_year_count:,}개")
        
        # 첫해 데이터 삭제
        df_final = df[~first_year_mask].copy()
        final_count = len(df_final)
        
        print(f"첫해 데이터 삭제 후: {final_count:,}개")
        print(f"전체 삭제율: {(original_count - final_count)/original_count*100:.2f}%")
    
    # 기업별 데이터 개수 분포 확인
    company_data_counts = df_final['거래소코드'].value_counts()
    print(f"\n기업별 평균 연도 수: {company_data_counts.mean():.1f}년")
    print(f"기업별 연도 수 범위: {company_data_counts.min()}~{company_data_counts.max()}년")
    
    return df_final

def main():
    """
    메인 실행 함수
    """
    print("=" * 60)
    print("부도예측을 위한 피처 엔지니어링 및 전처리 파이프라인")
    print("=" * 60)
    
    # 1. 데이터 로드 및 병합
    df = load_and_merge_data()
    
    # 2. 대차대조표 평균값 생성
    df = create_balance_sheet_averages(df)
    
    # 3. 모든 피처 카테고리 생성
    df = create_growth_features(df)
    df = create_liquidity_leverage_features(df)
    
    # 3-1. 새로 생성된 BS 항목들의 평균값 생성
    df = create_additional_balance_sheet_averages(df)
    
    df = create_profitability_features(df)
    df = create_leverage_metrics(df)  # DOL과 DFL만
    df = create_market_features(df)
    df = create_momentum_features(df)
    df = create_dividend_features(df)
    df = create_delta_features(df)
    
    # 4. 수치 데이터 정리
    df = clean_numerical_data(df)
    
    # 5. 각 기업별 첫해 데이터 삭제 (부도 기업 제외)
    df = remove_first_year_data(df)
    
    # 6. 피처 품질 검증
    validation_report = validate_features(df)
    
    # 7. 출력 저장
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # src/data_processing -> src -> project_root
    output_path = project_root / "data" / "processed" / "FS2_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print("=" * 60)
    print("피처 엔지니어링 및 전처리 완료!")
    print(f"출력 저장 위치: {output_path}")
    print(f"최종 데이터셋: {df.shape[0]} 행 × {df.shape[1]} 열")
    print("💡 각 기업의 첫해 데이터가 삭제되어 전년 대비 비교 가능한 데이터만 포함")
    print("=" * 60)
    
    # 피처 요약 표시
    print("\n피처 요약:")
    print(f"원본 컬럼: {len([col for col in df.columns if not any(prefix in col for prefix in ['avg_', 'Δ', '증가율', '비율', 'ROA', 'ROE', 'PBR', 'PER', 'PSR', 'PCR', 'DOL', 'DFL', 'EV', '로그', '변화량'])])}")
    print(f"성장률 피처: {len([col for col in df.columns if '증가율' in col])}")
    print(f"비율 피처: {len([col for col in df.columns if '비율' in col])}")
    print(f"시장 피처: {len([col for col in df.columns if any(prefix in col for prefix in ['PBR', 'PER', 'PSR', 'PCR', 'EV', '시가총액'])])}")
    print(f"델타 피처: {len([col for col in df.columns if 'Δ' in col])}")
    print(f"변화량 피처: {len([col for col in df.columns if '변화량' in col])}")
    print(f"로그 변환 피처: {len([col for col in df.columns if '로그' in col])}")
    print(f"평균 B/S 피처: {len([col for col in df.columns if 'avg_' in col])}")
    print(f"  - 기본 BS 평균: {len([col for col in df.columns if 'avg_' in col and not any(item in col for item in ['투하자본', '경영자본', '운전자본'])])}")
    print(f"  - 추가 BS 평균: {len([col for col in df.columns if 'avg_' in col and any(item in col for item in ['투하자본', '경영자본', '운전자본'])])}")
    print(f"\n💡 전년 대비 피처들은 각 기업 첫해 삭제로 인해 정확하게 계산됨")
    print(f"💡 새로 생성된 BS 항목들(투하자본, 경영자본 등)의 평균값도 생성됨")
    
    # 검증 리포트 요약
    print("\n데이터 품질 요약:")
    total_issues = (
        len(validation_report['negative_assets']) +
        len(validation_report['extreme_ratios']) +
        len(validation_report['high_missing_rate']) +
        len(validation_report['suspicious_values'])
    )
    print(f"총 데이터 품질 이슈: {total_issues}개")
    print(f"무한값/NaN 처리 완료: {df.isna().sum().sum()}개 NaN 값 존재")

if __name__ == "__main__":
    main()