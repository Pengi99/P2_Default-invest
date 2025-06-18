import pandas as pd
import os

def apply_default_labeling(final_path, fail_path):
    """
    final.csv에 default 컬럼을 추가하고 부도 기업 정보를 기반으로 레이블링을 수행합니다.
    - 부도 기업은 부도 전년도에 'default' = 1로 레이블링됩니다.
    - 'default' = 1로 레이블링된 기업의 다른 모든 연도 데이터는 삭제됩니다.
    - 나머지 모든 데이터는 'default' = 0으로 설정됩니다.

    Args:
        final_path (str): final.csv 파일 경로
        fail_path (str): value_fail.csv 파일 경로
    """
    try:
        final_df = pd.read_csv(final_path)
        fail_df = pd.read_csv(fail_path)
        print("CSV 파일 로드 완료.")
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요. {e}")
        return

    # 데이터 타입 변환 및 연도 추출
    fail_df['폐지일자'] = pd.to_datetime(fail_df['폐지일자'], errors='coerce')
    fail_df.dropna(subset=['폐지일자'], inplace=True) # 날짜 변환 실패한 행 제거
    fail_df['폐지년도'] = fail_df['폐지일자'].dt.year
    
    # '회계년도'에서 연도만 추출하여 임시 컬럼 생성
    final_df['회계년도_year'] = pd.to_datetime(final_df['회계년도'], format='%Y/%m', errors='coerce').dt.year

    # 'default' 컬럼 초기화
    final_df['default'] = 0

    # 부도 기업 레이블링
    for _, row in fail_df.iterrows():
        company_code = row['종목코드']
        target_year = row['폐지년도'] - 1
        
        condition = (final_df['거래소코드'] == company_code) & (final_df['회계년도_year'] == target_year)
        final_df.loc[condition, 'default'] = 1

    print(f"'default=1'로 레이블링된 기업 수: {final_df[final_df['default'] == 1].shape[0]} 건")

    # 부도 처리된 기업들의 종목코드 추출
    default_company_codes = final_df[final_df['default'] == 1]['거래소코드'].unique()
    print(f"고유한 부도 기업 수: {len(default_company_codes)} 개")

    # 부도 기업은 default=1인 행만 남기고, 정상 기업은 모든 행을 유지하는 조건
    condition_to_keep = ~final_df['거래소코드'].isin(default_company_codes) | (final_df['default'] == 1)
    final_df_processed = final_df[condition_to_keep].copy()

    # 불필요한 임시 컬럼 제거
    final_df_processed.drop(columns=['회계년도_year'], inplace=True)

    # 결과 저장
    final_df_processed.to_csv(final_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료. 최종 데이터 {len(final_df_processed)} 행이 '{final_path}'에 저장되었습니다.")


if __name__ == '__main__':
    # 이 스크립트는 'src' 폴더에 위치한다고 가정합니다.
    # 프로젝트 루트 디렉토리를 기준으로 파일 경로를 설정합니다.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    final_csv_path = os.path.join(base_dir, 'data', 'processed', 'final.csv')
    value_fail_csv_path = os.path.join(base_dir, 'data', 'processed', 'value_fail.csv')

    # 함수 실행
    apply_default_labeling(final_csv_path, value_fail_csv_path)
