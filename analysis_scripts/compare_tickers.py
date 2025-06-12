import pandas as pd
import re
import os

def analyze_ticker_discrepancy():
    """
    Analyzes the discrepancy between delisted tickers and tickers in financial statements.
    """
    print("--- Ticker 불일치 분석 시작 ---")

    # --- 파일 경로 설정 ---
    base_path = '/Users/jojongho/KDT/P2_Default-invest/data/raw'
    kospi_delisted_path = os.path.join(base_path, '코스피_상장폐지.csv')
    kosdaq_delisted_path = os.path.join(base_path, '코스닥_상장폐지.csv')
    consolidated_financial_path = os.path.join(base_path, '연결 재무제표(IFRS).csv')
    standard_financial_path = os.path.join(base_path, '재무제표(IFRS).csv')

    # --- CSV 읽기 도우미 함수 ---
    def read_csv_with_fallback(path):
        try:
            return pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"{os.path.basename(path)} 파일에 UTF-8 인코딩 적용 실패, cp949로 재시도합니다...")
            return pd.read_csv(path, encoding='cp949')

    # --- 1. 상장폐지 종목코드 추출 ---
    print("\n1단계: 상장폐지 목록에서 종목코드 추출 중...")
    try:
        kospi_delisted = read_csv_with_fallback(kospi_delisted_path)
        kosdaq_delisted = read_csv_with_fallback(kosdaq_delisted_path)
        all_delisted = pd.concat([kospi_delisted, kosdaq_delisted], ignore_index=True)
        delisted_tickers = set(all_delisted['종목코드'].astype(str).str.zfill(6))
        print(f"-> 총 {len(delisted_tickers)}개의 고유 상장폐지 종목코드를 찾았습니다.")
    except Exception as e:
        print(f"오류: 상장폐지 목록 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    # --- 2. 재무제표 종목코드 추출 (연결 + 일반) ---
    print("\n2단계: 재무제표 파일들에서 종목코드 추출 중...")
    
    def extract_tickers_from_financial_file(file_path):
        tickers = set()
        if not os.path.exists(file_path):
            print(f"경고: {os.path.basename(file_path)} 파일이 존재하지 않습니다. 건너뜁니다.")
            return tickers
        
        try:
            with open(file_path, 'r', encoding='cp949', errors='ignore') as f:
                for line in f:
                    match = re.search(r',(\d{6}),\d{4}/\d{2},', line)
                    if match:
                        tickers.add(match.group(1))
            print(f"-> {os.path.basename(file_path)}: {len(tickers)}개 코드 추출 완료.")
        except Exception as e:
            print(f"오류: {os.path.basename(file_path)} 처리 중 오류 발생: {e}")
        return tickers

    consolidated_tickers = extract_tickers_from_financial_file(consolidated_financial_path)
    standard_tickers = extract_tickers_from_financial_file(standard_financial_path)
    
    financial_tickers = consolidated_tickers.union(standard_tickers)
    print(f"-> 총 {len(financial_tickers)}개의 고유 종목코드를 모든 재무제표에서 찾았습니다.")

    # --- 3. 데이터 비교 및 분석 ---
    print("\n3단계: 두 종목코드 목록 비교 중...")
    delisted_not_in_financials = delisted_tickers - financial_tickers
    financials_not_in_delisted = financial_tickers - delisted_tickers
    common_tickers = delisted_tickers.intersection(financial_tickers)

    print("\n--- 최종 분석 결과 ---")
    print(f"고유 상장폐지 종목 수: {len(delisted_tickers)}")
    print(f"고유 재무제표 종목 수: {len(financial_tickers)}")
    print("-" * 30)
    print(f"공통 종목 수: {len(common_tickers)}")
    print(f"[불일치] 상장폐지 목록에만 존재: {len(delisted_not_in_financials)}")
    print(f"[정상] 재무제표 목록에만 존재: {len(financials_not_in_delisted)}")
    print("-" * 30)

    # --- 4. 불일치 예시 출력 ---
    if delisted_not_in_financials:
        print("\n[정보] 상장폐지 목록에는 있지만, 재무제표 데이터에는 없는 종목 예시:")
        example_tickers = list(delisted_not_in_financials)[:10]
        examples_df = all_delisted[all_delisted['종목코드'].astype(str).str.zfill(6).isin(example_tickers)]
        print(examples_df[['회사명', '종목코드', '폐지일자', '폐지사유']].to_string())
        print("\n>> 불일치 발생 추정 원인:")
        print("   1. 스팩(SPAC)과 같은 특수 목적 회사는 재무제표 집계에서 제외되었을 수 있습니다.")
        print("   2. 해당 기업이 '연결' 재무제표가 아닌 '개별' 재무제표만 공시했을 수 있습니다.")
        print("   3. 데이터 수집 과정에서 해당 기업의 재무 데이터가 누락되었을 수 있습니다.")

if __name__ == "__main__":
    analyze_ticker_discrepancy()
