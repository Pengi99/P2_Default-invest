"""
백테스팅 -97% 수익률 문제 근본 원인 분석
"""

import pandas as pd
import os

def analyze_root_causes():
    """근본 원인 분석 및 해결책 제시"""
    print("🔍 -97% 수익률 문제 근본 원인 분석")
    print("=" * 60)
    
    # 1. 설정 파일 문제
    print("\n1️⃣ 설정 파일 문제:")
    
    config_issues = []
    
    # 재무 데이터 파일 확인
    expected_fs = 'data/processed/FS2_default.csv'
    actual_fs = 'data/processed/FS2_no_default.csv'
    
    if not os.path.exists(expected_fs) and os.path.exists(actual_fs):
        config_issues.append(f"❌ 설정에서 {expected_fs}를 요구하지만 {actual_fs}만 존재")
    
    # 가격 데이터 구조 확인
    price_dir = 'data/raw'
    if os.path.exists(price_dir):
        files = os.listdir(price_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"   📁 {price_dir}에 {len(csv_files)}개 CSV 파일 발견:")
        for f in csv_files[:5]:  # 처음 5개만 표시
            print(f"      - {f}")
        if len(csv_files) > 5:
            print(f"      ... 및 {len(csv_files)-5}개 더")
        
        # 파일 구조 분석
        sample_file = f"{price_dir}/{csv_files[0]}" if csv_files else None
        if sample_file and os.path.exists(sample_file):
            df_sample = pd.read_csv(sample_file, encoding='utf-8-sig', nrows=5)
            print(f"   📊 샘플 파일 구조 ({csv_files[0]}):")
            print(f"      컬럼: {list(df_sample.columns)}")
            
            # 필수 컬럼 확인
            required_cols = ['매매년월일', '시가총액(원)', '거래소코드']
            missing_cols = [col for col in required_cols if col not in df_sample.columns]
            if missing_cols:
                config_issues.append(f"❌ 필수 컬럼 누락: {missing_cols}")
            else:
                print(f"      ✅ 필수 컬럼 모두 존재")
    
    # 2. 데이터 타입 문제
    print("\n2️⃣ 데이터 타입 및 품질 문제:")
    
    data_issues = []
    
    # 재무 데이터 분석
    fs_file = actual_fs if os.path.exists(actual_fs) else expected_fs
    if os.path.exists(fs_file):
        fs_df = pd.read_csv(fs_file, encoding='utf-8-sig')
        print(f"   📊 재무 데이터 ({os.path.basename(fs_file)}):")
        print(f"      - 행 수: {len(fs_df):,}")
        print(f"      - 컬럼 수: {len(fs_df.columns)}")
        
        # 필수 재무 지표 확인
        key_metrics = ['총자산', '당기순이익', '영업활동으로인한현금흐름', '시가총액']
        for metric in key_metrics:
            if metric in fs_df.columns:
                valid_count = fs_df[metric].notna().sum()
                zero_neg_count = (fs_df[metric] <= 0).sum()
                print(f"      - {metric}: 유효값 {valid_count:,}개, 0이하값 {zero_neg_count:,}개")
                
                if zero_neg_count > valid_count * 0.1:  # 10% 이상이 문제값
                    data_issues.append(f"❌ {metric}에 과도한 0/음수값 ({zero_neg_count:,}개)")
            else:
                data_issues.append(f"❌ {metric} 컬럼 누락")
        
        # 날짜 정보 확인
        date_cols = [col for col in fs_df.columns if any(keyword in col.lower() for keyword in ['date', '날짜', '연도', '년도'])]
        if date_cols:
            print(f"      - 날짜 관련 컬럼: {date_cols}")
        else:
            data_issues.append("❌ 날짜 관련 컬럼 누락")
    
    # 3. 라이브러리 의존성 문제
    print("\n3️⃣ 라이브러리 의존성 문제:")
    
    lib_issues = []
    
    try:
        import polars as pl
        print("   ✅ Polars 사용 가능")
    except ImportError:
        lib_issues.append("❌ Polars 미설치 - 최적화 코드가 fallback으로 실행")
    
    try:
        from numba import jit
        print("   ✅ Numba 사용 가능")
    except ImportError:
        lib_issues.append("❌ Numba 미설치 - 성능 최적화 무효화")
    
    try:
        import dask
        print("   ✅ Dask 사용 가능")
    except ImportError:
        lib_issues.append("❌ Dask 미설치")
    
    try:
        import yfinance as yf
        print("   ✅ yfinance 사용 가능")
    except ImportError:
        lib_issues.append("❌ yfinance 미설치 - 벤치마크 데이터 로딩 실패")
    
    # 4. 요약 및 해결책
    print("\n🎯 문제 요약:")
    all_issues = config_issues + data_issues + lib_issues
    
    for i, issue in enumerate(all_issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n💡 해결책:")
    solutions = [
        "1. config.yaml에서 fundamental 경로를 'data/processed/FS2_no_default.csv'로 수정",
        "2. 누락된 Python 패키지 설치: pip install polars numba dask yfinance",
        "3. 가격 데이터 파일 구조 및 필수 컬럼 검증",
        "4. 재무 데이터의 0/음수값 처리 로직 개선",
        "5. 최적화 라이브러리 부재 시 안전한 fallback 로직 확인"
    ]
    
    for solution in solutions:
        print(f"   {solution}")
    
    return {
        'config_issues': config_issues,
        'data_issues': data_issues,
        'lib_issues': lib_issues,
        'total_issues': len(all_issues)
    }

if __name__ == "__main__":
    results = analyze_root_causes()
    
    if results['total_issues'] == 0:
        print("\n🎉 문제가 발견되지 않았습니다!")
    else:
        print(f"\n⚠️ 총 {results['total_issues']}개 문제 발견됨")
        print("위 해결책을 순서대로 적용하여 문제를 해결하세요.")