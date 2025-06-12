import pandas as pd
import re
import os
import sys
import io
import csv

# 터미널 인코딩을 UTF-8로 설정하여 출력 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_reasons():
    """
    상장폐지 목록과 재무제표 데이터를 비교하여, 
    재무제표에 누락된 상장폐지 기업의 폐지 사유를 분석하고 그룹화하여 출력합니다.
    """
    print("--- 상장폐지 사유 심층 분석 시작 ---")

    base_path = '/Users/jojongho/KDT/P2_Default-invest/data/raw'
    
    # 1. 상장폐지 목록 로드
    try:
        column_names = ['번호', '회사명', '종목코드', '폐지일자', '폐지사유', '비고']
        
        kospi_delisted = pd.read_csv(
            os.path.join(base_path, '코스피_상장폐지.csv'), 
            encoding='utf-8',
            header=None,
            skiprows=1,
            names=column_names
        )
        kosdaq_delisted = pd.read_csv(
            os.path.join(base_path, '코스닥_상장폐지.csv'), 
            encoding='utf-8',
            header=None,
            skiprows=1,
            names=column_names
        )
        
        all_delisted_df = pd.concat([kospi_delisted, kosdaq_delisted], ignore_index=True)
        # '종목코드'가 숫자인 경우만 필터링하고 zfill 적용
        all_delisted_df = all_delisted_df[all_delisted_df['종목코드'].astype(str).str.isdigit()].copy()
        all_delisted_df['종목코드'] = all_delisted_df['종목코드'].astype(str).str.zfill(6)
        delisted_tickers = set(all_delisted_df['종목코드'])
        
    except Exception as e:
        print(f"오류: 상장폐지 목록 파일 로드 실패: {e}")
        return

    # 2. 재무제표 데이터에서 종목코드 추출
    try:
        financial_files = ['연결 재무제표(IFRS).csv', '재무제표(IFRS).csv']
        financial_tickers = set()
        chunk_size = 10000  # 한 번에 10,000줄씩 처리

        for file in financial_files:
            path = os.path.join(base_path, file)
            if not os.path.exists(path):
                print(f"경고: 재무제표 파일 '{file}'을(를) 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 파일을 청크 단위로 읽어 메모리 효율적으로 처리
            for chunk in pd.read_csv(path, encoding='utf-8', dtype={'거래소코드': str}, chunksize=chunk_size, low_memory=False):
                if '거래소코드' in chunk.columns:
                    # NaN 값을 제거하고, 6자리로 포맷팅
                    valid_tickers = chunk['거래소코드'].dropna().str.zfill(6)
                    financial_tickers.update(valid_tickers)

    except Exception as e:
        print(f"오류: 재무제표 파일 로드 또는 파싱 실패: {e}")
        return

    # 3. 재무제표에 없는 상장폐지 기업 필터링
    missing_tickers = delisted_tickers - financial_tickers
    print(f"\n총 {len(missing_tickers)}개의 기업이 상장폐지 목록에만 존재합니다. 이들의 폐지 사유를 분석합니다.")
    
    missing_df = all_delisted_df[all_delisted_df['종목코드'].isin(missing_tickers)].copy()
    
    # 4. 폐지 사유 그룹화
    def categorize_reason(reason):
        reason_str = str(reason).replace(' ', '').lower()
        if '스팩' in reason_str or 'spac' in reason_str:
            return '스팩(SPAC) 관련'
        if '자회사' in reason_str or '합병' in reason_str or '인수' in reason_str or '완전모회사' in reason_str:
            return '합병/자회사 편입'
        if '의견거절' in reason_str or '부적정' in reason_str or '거절' in reason_str:
            return '감사의견 거절'
        if '해산' in reason_str or '파산' in reason_str:
            return '파산/해산'
        if '신청에의한' in reason_str or '자진' in reason_str or '신청' in reason_str:
            return '자진 상장폐지'
        if '자본잠식' in reason_str or '자본전액잠식' in reason_str:
            return '자본잠식'
        if '존속기간만료' in reason_str:
            return '존속기간 만료'
        if '시가총액미달' in reason_str or '매출액미달' in reason_str or '주식분산미달' in reason_str:
            return '상장유지기준 미달'
        if '기업의계속성' in reason_str or '상장적격성실질심사' in reason_str:
            return '상장적격성 실질심사'
        return '기타'

    missing_df['사유_그룹'] = missing_df['폐지사유'].apply(categorize_reason)
    
    reason_groups = missing_df['사유_그룹'].value_counts()
    
    # 5. 그룹별 분석 결과 출력
    print("\n--- 폐지 사유 그룹별 분석 결과 ---\n")
    
    analysis_text = {
        '합병/자회사 편입': "  [분석] 다른 회사에 흡수합병되거나 완전 자회사로 편입된 경우, 개별 재무 정보는 모회사의 연결 재무제표에 통합됩니다. 따라서 별도의 재무 데이터가 존재하지 않게 됩니다.",
        '감사의견 거절': "  [분석] 감사의견 거절은 회계 정보의 신뢰성에 심각한 문제가 있음을 의미합니다. 데이터 제공 업체는 이러한 신뢰할 수 없는 재무 정보를 데이터베이스에서 제외하는 경우가 많습니다.",
        '파산/해산': "  [분석] 파산이나 해산 절차에 들어간 기업은 정상적인 영업 활동을 중단하므로, 재무제표를 공시할 의무가 사라지거나 데이터의 연속성이 깨져 집계에서 누락될 수 있습니다.",
        '스팩(SPAC) 관련': "  [분석] 스팩은 서류상 회사로, 합병 전까지는 영업활동이 없어 일반적인 재무제표를 생성하지 않습니다. 따라서 데이터 집계에서 제외되는 것이 일반적입니다.",
        '자진 상장폐지': "  [분석] 기업이 자발적으로 상장을 폐지하면, 더 이상 재무 정보를 대중에게 공시할 의무가 없습니다. 이로 인해 상장폐지 이후의 데이터는 찾을 수 없게 됩니다.",
        '자본잠식': "  [분석] 완전 자본잠식 등 심각한 재무 악화는 상장폐지 사유가 되며, 이는 감사의견 거절과 유사하게 데이터의 신뢰성 문제로 이어져 집계에서 제외될 수 있습니다.",
        '존속기간 만료': "  [분석] 선박투자회사, 부동산투자회사(REITs) 등 한시적으로 운영되는 특수 목적 회사가 여기에 해당합니다. 이들은 예정된 기간이 끝나면 자동 해산되므로 재무 데이터가 더 이상 생성되지 않습니다.",
        '상장유지기준 미달': "  [분석] 매출액, 시가총액, 주주 수 등 거래소가 정한 최소한의 상장 유지 요건을 충족하지 못한 경우입니다. 기업의 규모나 시장성이 너무 작아져 투자자 보호를 위해 퇴출된 사례입니다.",
        '상장적격성 실질심사': "  [분석] 기업의 계속성, 경영 투명성 등 계량적 기준 외의 질적 요건에 심각한 문제가 발생하여 거래소의 종합적인 판단에 따라 퇴출된 경우입니다.",
        '기타': "  [분석] 보고서 미제출, 불성실 공시 등 위에 분류되지 않은 다양한 사유로 상장폐지된 경우입니다. 이 역시 기업의 신뢰도 문제와 직결되어 데이터 집계에서 제외될 수 있습니다."
    }

    for group, count in reason_groups.items():
        print(f"--- 그룹: {group} ({count}개) ---")
        print(analysis_text.get(group, "  [분석] 분류되지 않은 사유입니다."))

if __name__ == "__main__":
    analyze_reasons()
