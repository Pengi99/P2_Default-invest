"""
연도별 시가총액 합산 스크립트
==========================

2012.csv부터 2023.csv까지의 파일을 읽어서
각 연도 말일 기준으로 모기업과 우선주의 시가총액을 합산하여
최종 결과 파일을 생성합니다.

출력: 시가총액.csv
컬럼: 회계년도, 회사명, 거래소코드, 시가총액
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

def extract_parent_company_name(stock_name):
    """
    종목명에서 모기업명을 추출하는 함수
    
    Args:
        stock_name: 종목명 (예: 삼성전자보통주, 삼성전자1우선주, BYC보통주, BYC1우선주)
        
    Returns:
        모기업명 (예: 삼성전자, BYC)
    """
    # 우선주 및 보통주 패턴들 (더 포괄적으로 수정)
    patterns = [
        r'\d*우선주\(신형\)$',         # 2우선주(신형), 우선주(신형) 등
        r'전환상환\d*우선주\(신형\)$',   # 전환상환2우선주(신형) 등
        r'\d*우선주$',                 # 1우선주, 2우선주, 우선주 등
        r'\(\d*우선주\(신형\)\)$',     # (1우선주(신형)) 등
        r'\(\d*우선주\)$',             # (1우선주) 등
        r'\(우선주\)$',                # (우선주)
        r'\d*우\([AB]*\)$',            # 우(A), 우(B), 2우(A), 2우(B) 등
        r'\d*우[AB]*$',                # 우, 우A, 우B, 2우, 2우A, 2우B 등
        r'보통주$',                    # 보통주
        r'스팩\d*$',                   # 스팩 관련
    ]
    
    parent_name = stock_name.strip()
    
    # 각 패턴에 대해 검사하여 접미사 제거
    for pattern in patterns:
        parent_name = re.sub(pattern, '', parent_name)
    
    return parent_name.strip()

def find_parent_stock_code(group_df):
    """
    그룹 내에서 모기업(보통주)의 종목코드를 찾는 함수
    보통주가 있으면 보통주를 선택하고, 없으면 첫 번째 우선주를 선택
    
    Args:
        group_df: 같은 모기업명으로 그룹화된 DataFrame
        
    Returns:
        모기업의 종목코드
    """
    # 1단계: 우선주 패턴이 전혀 없는 보통주 찾기
    common_stocks = []
    preferred_stocks = []
    
    for _, row in group_df.iterrows():
        stock_name = row['종목명']
        # 우선주 패턴 체크 (업데이트된 패턴 적용)
        preferred_pattern = (
            r'\d*우선주\(신형\)$|'          # 2우선주(신형), 우선주(신형) 등
            r'전환상환\d*우선주\(신형\)$|'    # 전환상환2우선주(신형) 등
            r'\d*우선주$|'                   # 1우선주, 2우선주, 우선주 등
            r'\(\d*우선주\(신형\)\)$|'       # (1우선주(신형)) 등
            r'\(\d*우선주\)$|'               # (1우선주) 등
            r'\(우선주\)$|'                  # (우선주)
            r'\d*우\([AB]*\)$|'              # 우(A), 우(B), 2우(A), 2우(B) 등
            r'\d*우[AB]*$|'                  # 우, 우A, 우B, 2우, 2우A, 2우B 등
            r'스팩\d*$'                      # 스팩 관련
        )
        
        if re.search(preferred_pattern, stock_name):
            preferred_stocks.append(row)
        else:
            common_stocks.append(row)
    
    # 보통주가 있으면 보통주의 종목코드 반환
    if common_stocks:
        return common_stocks[0]['종목코드']
    
    # 보통주가 없고 우선주만 있으면 첫 번째 우선주 반환
    if preferred_stocks:
        return preferred_stocks[0]['종목코드']
    
    # 예외 상황: 첫 번째 행 반환
    return group_df.iloc[0]['종목코드']

def process_yearly_data(year, data_dir):
    """
    특정 연도의 데이터를 처리하는 함수
    
    Args:
        year: 처리할 연도
        data_dir: 데이터 파일이 있는 디렉토리
        
    Returns:
        처리된 DataFrame (회계년도, 회사명, 거래소코드, 시가총액)
    """
    file_path = data_dir / f"{year}.csv"
    
    if not file_path.exists():
        print(f"경고: {file_path} 파일이 존재하지 않습니다.")
        return pd.DataFrame()
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"{year}년 데이터 로드 완료: {len(df)} 행")
        
        # 필수 컬럼 확인
        required_columns = ['매매년월일', '종목코드(축약)', '회사명', '시가총액(원)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"오류: {year}년 파일에 필요한 컬럼이 없습니다: {missing_columns}")
            return pd.DataFrame()
        
        # 컬럼명 통일을 위한 매핑
        df = df.rename(columns={
            '매매년월일': '일자',
            '종목코드(축약)': '종목코드', 
            '회사명': '종목명',
            '시가총액(원)': '시가총액'
        })
        
        # 일자 컬럼을 datetime으로 변환
        df['일자'] = pd.to_datetime(df['일자'])
        
        # 해당 연도의 마지막 거래일 찾기
        last_trading_date = df['일자'].max()
        print(f"{year}년 마지막 거래일: {last_trading_date.strftime('%Y-%m-%d')}")
        
        # 마지막 거래일 데이터만 필터링
        last_day_df = df[df['일자'] == last_trading_date].copy()
        print(f"마지막 거래일 데이터: {len(last_day_df)} 종목")
        
        # 시가총액이 숫자가 아니거나 결측치인 데이터 제거
        last_day_df = last_day_df.dropna(subset=['시가총액'])
        last_day_df = last_day_df[pd.to_numeric(last_day_df['시가총액'], errors='coerce').notna()]
        last_day_df['시가총액'] = pd.to_numeric(last_day_df['시가총액'])
        
        print(f"유효한 시가총액 데이터: {len(last_day_df)} 종목")
        
        # 모기업명 추출
        last_day_df['모기업명'] = last_day_df['종목명'].apply(extract_parent_company_name)
        
        # 모기업명별로 그룹화하여 시가총액 합산
        result_list = []
        
        for parent_name, group in last_day_df.groupby('모기업명'):
            total_market_cap = group['시가총액'].sum()
            parent_stock_code = find_parent_stock_code(group)
            
            result_list.append({
                '회계년도': year,
                '회사명': parent_name,
                '거래소코드': parent_stock_code,
                '시가총액': total_market_cap
            })
        
        result_df = pd.DataFrame(result_list)
        print(f"{year}년 처리 완료: {len(result_df)} 개 기업")
        
        return result_df
        
    except Exception as e:
        print(f"오류: {year}년 데이터 처리 중 문제 발생: {str(e)}")
        return pd.DataFrame()

def main():
    """
    메인 실행 함수
    """
    print("=" * 50)
    print("연도별 시가총액 합산 스크립트 시작")
    print("=" * 50)
    
    # 현재 스크립트가 있는 디렉토리를 기준으로 데이터 디렉토리 설정
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data" / "raw"  # data 폴더에 CSV 파일들이 있다고 가정
    
    # 데이터 디렉토리가 없으면 현재 디렉토리에서 찾기
    if not data_dir.exists():
        data_dir = current_dir
    
    print(f"데이터 디렉토리: {data_dir}")
    
    # 2012년부터 2023년까지 처리
    years = range(2012, 2024)
    all_results = []
    
    for year in years:
        print(f"\n{year}년 데이터 처리 중...")
        yearly_result = process_yearly_data(year, data_dir)
        
        if not yearly_result.empty:
            all_results.append(yearly_result)
    
    # 모든 연도 결과 합치기
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # 회계년도, 회사명 순으로 정렬
        final_df = final_df.sort_values(['회계년도', '회사명']).reset_index(drop=True)
        
        # 결과 파일 저장
        output_file = current_dir / "data" / "processed" / "시가총액.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 50)
        print("처리 완료!")
        print(f"총 {len(final_df)} 행의 데이터가 생성되었습니다.")
        print(f"출력 파일: {output_file}")
        print("=" * 50)
        
        # 결과 요약 출력
        print("\n[결과 요약]")
        print(f"처리된 연도: {final_df['회계년도'].min()}년 ~ {final_df['회계년도'].max()}년")
        print(f"총 기업 수: {final_df['회사명'].nunique()}개")
        print(f"연도별 기업 수:")
        year_counts = final_df['회계년도'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}년: {count}개 기업")
        
        # 샘플 데이터 출력
        print(f"\n[샘플 데이터] (첫 5행)")
        print(final_df.head().to_string(index=False))
        
    else:
        print("오류: 처리된 데이터가 없습니다. 입력 파일들을 확인해주세요.")

if __name__ == "__main__":
    main()