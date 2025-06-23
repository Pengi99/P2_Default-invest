"""
2011년 데이터를 활용한 2012년 성장률 및 변화량 재계산 스크립트

목적:
- 2011년 재무제표 데이터를 활용하여 2012년 성장률과 변화량의 결측치 감소
- YoY 성장률: 자산, 이익잉여금, 영업이익, 순이익
- 변화량: 부채비율, 유동비율, 총이익률, ROE, 총자본회전률, 운전자본회전률
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터 로드"""
    print("📁 데이터 로드 중...")
    
    # 2011년 데이터 로드
    bs2011 = pd.read_csv('data/processed/bs2011.csv', encoding='utf-8-sig')
    print(f"2011년 데이터: {len(bs2011)}개 기업")
    
    # 현재 최종 데이터셋 로드
    current_data = pd.read_csv('data/final/FS_ratio_flow.csv', encoding='utf-8-sig')
    print(f"현재 데이터셋: {len(current_data)}개 레코드")
    
    # 원본 재무 데이터 로드 (절대값 계산용)
    fs_flow = pd.read_csv('data/processed/FS_flow.csv', encoding='utf-8-sig')
    print(f"원본 재무 데이터: {len(fs_flow)}개 레코드")
    
    # 전체 재무제표 데이터 로드 (영업이익, 순이익 계산용)
    fs_raw = pd.read_csv('data/raw/FS.csv', encoding='utf-8-sig')
    print(f"전체 재무제표 데이터: {len(fs_raw)}개 레코드")
    
    return bs2011, current_data, fs_flow, fs_raw

def prepare_2011_data(bs2011):
    """2011년 데이터 전처리"""
    print("\n🔧 2011년 데이터 전처리 중...")
    
    # 필요한 컬럼만 선택하고 정리
    key_columns = {
        '거래소코드': '거래소코드',
        '자산(*)(연결)(천원)': '총자산',
        '이익잉여금(*)(연결)(천원)': '이익잉여금',
        '영업이익(손실)(연결)': '영업이익',
        '당기순이익(순손실)(연결)': '순이익',
        '부채비율': '부채비율',
        '유동비율': '유동비율',
        '매출액총이익률': '총이익률',
        '자기자본순이익률': 'ROE',
        '총자본회전률': '총자본회전률'
    }
    
    # 컬럼 매핑
    available_cols = {}
    for orig_col, new_col in key_columns.items():
        if orig_col in bs2011.columns:
            available_cols[orig_col] = new_col
        else:
            # 유사한 컬럼명 찾기
            similar_cols = [col for col in bs2011.columns if any(keyword in col for keyword in orig_col.split('(')[0].split())]
            if similar_cols:
                available_cols[similar_cols[0]] = new_col
                print(f"  {orig_col} -> {similar_cols[0]}로 대체")
    
    # 데이터 선택 및 정리
    select_cols = ['거래소코드'] + [k for k in available_cols.keys() if k != '거래소코드']
    bs2011_clean = bs2011[select_cols].copy()
    bs2011_clean.rename(columns=available_cols, inplace=True)
    
    # 거래소코드를 문자열로 변환 (앞의 0 제거)
    bs2011_clean['거래소코드'] = bs2011_clean['거래소코드'].astype(str).str.zfill(6)
    
    # 중복 제거 (같은 기업의 여러 기록이 있을 경우)
    bs2011_clean = bs2011_clean.drop_duplicates(subset=['거래소코드']).reset_index(drop=True)
    
    print(f"  정리된 2011년 데이터: {len(bs2011_clean)}개 기업")
    print(f"  사용 가능한 컬럼: {list(bs2011_clean.columns)}")
    
    return bs2011_clean

def calculate_growth_rates(current_data, bs2011_clean, fs_flow, fs_raw):
    """성장률 계산"""
    print("\n📈 성장률 재계산 중...")
    
    # 2012년 데이터만 필터링
    data_2012_current = current_data[current_data['회계년도'] == '2012/12'].copy()
    data_2012_fs = fs_flow[fs_flow['회계년도'] == '2012/12'].copy()
    data_2012_raw = fs_raw[fs_raw['회계년도'] == 2012].copy()
    print(f"2012년 데이터: {len(data_2012_current)}개")
    
    # 기존 성장률 결측치 개수 확인
    growth_vars = ['자산_YoY_성장률', '이익잉여금_YoY_성장률', '영업이익_YoY_성장률', '순이익_YoY_성장률']
    print("\n기존 결측치 개수:")
    for var in growth_vars:
        if var in data_2012_current.columns:
            missing_count = data_2012_current[var].isna().sum()
            print(f"  {var}: {missing_count}개")
    
    # 영업이익, 순이익 컬럼명 찾기
    profit_cols = [col for col in fs_raw.columns if '영업이익' in col and '천원' in col]
    net_income_cols = [col for col in fs_raw.columns if '당기순이익' in col and '천원' in col]
    
    print(f"영업이익 컬럼: {profit_cols[:3]}")
    print(f"순이익 컬럼: {net_income_cols[:3]}")
    
    # 성장률 재계산
    improved_count = 0
    
    for idx, row in data_2012_current.iterrows():
        code = str(row['거래소코드']).zfill(6)
        
        # 2012년 FS_flow 데이터에서 해당 기업 찾기
        fs_2012_data = data_2012_fs[data_2012_fs['거래소코드'].astype(str).str.zfill(6) == code]
        
        # 2012년 FS_raw 데이터에서 해당 기업 찾기
        raw_2012_data = data_2012_raw[data_2012_raw['거래소코드'].astype(str).str.zfill(6) == code]
        
        # 2011년 데이터에서 해당 기업 찾기
        base_data = bs2011_clean[bs2011_clean['거래소코드'] == code]
        
        if len(base_data) > 0:
            base_row = base_data.iloc[0]
            
            # 자산 성장률
            if pd.isna(row['자산_YoY_성장률']) and len(fs_2012_data) > 0:
                fs_row = fs_2012_data.iloc[0]
                if '총자산' in base_row and pd.notna(base_row['총자산']) and pd.notna(fs_row['자산_당기말']):
                    if base_row['총자산'] != 0:
                        growth_rate = (fs_row['자산_당기말'] - base_row['총자산']) / base_row['총자산']
                        current_data.loc[idx, '자산_YoY_성장률'] = growth_rate
                        improved_count += 1
            
            # 이익잉여금 성장률  
            if pd.isna(row['이익잉여금_YoY_성장률']) and len(fs_2012_data) > 0:
                fs_row = fs_2012_data.iloc[0]
                if '이익잉여금' in base_row and pd.notna(base_row['이익잉여금']) and pd.notna(fs_row['이익잉여금_당기말']):
                    if base_row['이익잉여금'] != 0:
                        growth_rate = (fs_row['이익잉여금_당기말'] - base_row['이익잉여금']) / abs(base_row['이익잉여금'])
                        current_data.loc[idx, '이익잉여금_YoY_성장률'] = growth_rate
                        improved_count += 1
                    elif base_row['이익잉여금'] == 0 and fs_row['이익잉여금_당기말'] != 0:
                        # 0에서 양수/음수로 변한 경우
                        current_data.loc[idx, '이익잉여금_YoY_성장률'] = 1.0 if fs_row['이익잉여금_당기말'] > 0 else -1.0
                        improved_count += 1
            
            # 영업이익 성장률
            if pd.isna(row['영업이익_YoY_성장률']) and len(raw_2012_data) > 0 and profit_cols:
                raw_row = raw_2012_data.iloc[0]
                profit_col = profit_cols[0]  # 첫 번째 영업이익 컬럼 사용
                if '영업이익' in base_row and pd.notna(base_row['영업이익']) and pd.notna(raw_row[profit_col]):
                    if base_row['영업이익'] != 0:
                        growth_rate = (raw_row[profit_col] - base_row['영업이익']) / abs(base_row['영업이익'])
                        current_data.loc[idx, '영업이익_YoY_성장률'] = growth_rate
                        improved_count += 1
                    elif base_row['영업이익'] == 0 and raw_row[profit_col] != 0:
                        current_data.loc[idx, '영업이익_YoY_성장률'] = 1.0 if raw_row[profit_col] > 0 else -1.0
                        improved_count += 1
            
            # 순이익 성장률
            if pd.isna(row['순이익_YoY_성장률']) and len(raw_2012_data) > 0 and net_income_cols:
                raw_row = raw_2012_data.iloc[0]
                net_income_col = net_income_cols[0]  # 첫 번째 순이익 컬럼 사용
                if '순이익' in base_row and pd.notna(base_row['순이익']) and pd.notna(raw_row[net_income_col]):
                    if base_row['순이익'] != 0:
                        growth_rate = (raw_row[net_income_col] - base_row['순이익']) / abs(base_row['순이익'])
                        current_data.loc[idx, '순이익_YoY_성장률'] = growth_rate
                        improved_count += 1
                    elif base_row['순이익'] == 0 and raw_row[net_income_col] != 0:
                        current_data.loc[idx, '순이익_YoY_성장률'] = 1.0 if raw_row[net_income_col] > 0 else -1.0
                        improved_count += 1
    
    print(f"✅ 성장률 개선 완료: {improved_count}개 값 보완")
    return current_data

def calculate_changes(current_data, bs2011_clean, fs_raw):
    """변화량 계산"""
    print("\n📊 변화량 재계산 중...")
    
    # 2012년 데이터만 필터링
    data_2012_current = current_data[current_data['회계년도'] == '2012/12'].copy()
    data_2012_raw = fs_raw[fs_raw['회계년도'] == 2012].copy()
    
    # 변화량 변수들
    change_vars = ['부채비율_변화량', '유동비율_변화량', '총이익률_변화량', 
                  'ROE_변화량', '총자본회전률_변화량', '운전자본회전률_변화량']
    
    print("\n기존 결측치 개수:")
    for var in change_vars:
        if var in data_2012_current.columns:
            missing_count = data_2012_current[var].isna().sum()
            print(f"  {var}: {missing_count}개")
    
    # 변화량 재계산
    improved_count = 0
    
    for idx, row in data_2012_current.iterrows():
        code = str(row['거래소코드']).zfill(6)
        
        # 2012년 FS_raw 데이터에서 해당 기업 찾기
        raw_2012_data = data_2012_raw[data_2012_raw['거래소코드'].astype(str).str.zfill(6) == code]
        
        # 2011년 데이터에서 해당 기업 찾기
        base_data = bs2011_clean[bs2011_clean['거래소코드'] == code]
        
        if len(base_data) > 0 and len(raw_2012_data) > 0:
            base_row = base_data.iloc[0]
            raw_row = raw_2012_data.iloc[0]
            
            # 부채비율 변화량 - FS_raw에서 부채비율 찾기
            debt_ratio_cols = [col for col in fs_raw.columns if '부채비율' in col]
            if pd.isna(row.get('부채비율_변화량')) and debt_ratio_cols and '부채비율' in base_row:
                debt_ratio_col = debt_ratio_cols[0]
                if pd.notna(base_row['부채비율']) and pd.notna(raw_row[debt_ratio_col]):
                    change = raw_row[debt_ratio_col] - base_row['부채비율']
                    current_data.loc[idx, '부채비율_변화량'] = change
                    improved_count += 1
            
            # 유동비율 변화량
            current_ratio_cols = [col for col in fs_raw.columns if '유동비율' in col]
            if pd.isna(row.get('유동비율_변화량')) and current_ratio_cols and '유동비율' in base_row:
                current_ratio_col = current_ratio_cols[0]
                if pd.notna(base_row['유동비율']) and pd.notna(raw_row[current_ratio_col]):
                    change = raw_row[current_ratio_col] - base_row['유동비율']
                    current_data.loc[idx, '유동비율_변화량'] = change
                    improved_count += 1
            
            # 총이익률 변화량
            gross_margin_cols = [col for col in fs_raw.columns if '총이익률' in col or '매출액총이익률' in col]
            if pd.isna(row.get('총이익률_변화량')) and gross_margin_cols and '총이익률' in base_row:
                gross_margin_col = gross_margin_cols[0]
                if pd.notna(base_row['총이익률']) and pd.notna(raw_row[gross_margin_col]):
                    change = raw_row[gross_margin_col] - base_row['총이익률']
                    current_data.loc[idx, '총이익률_변화량'] = change
                    improved_count += 1
            
            # ROE 변화량
            roe_cols = [col for col in fs_raw.columns if 'ROE' in col or '자기자본순이익률' in col]
            if pd.isna(row.get('ROE_변화량')) and roe_cols and 'ROE' in base_row:
                roe_col = roe_cols[0]
                if pd.notna(base_row['ROE']) and pd.notna(raw_row[roe_col]):
                    change = raw_row[roe_col] - base_row['ROE']
                    current_data.loc[idx, 'ROE_변화량'] = change
                    improved_count += 1
            
            # 총자본회전률 변화량
            turnover_cols = [col for col in fs_raw.columns if '총자본회전률' in col]
            if pd.isna(row.get('총자본회전률_변화량')) and turnover_cols and '총자본회전률' in base_row:
                turnover_col = turnover_cols[0]
                if pd.notna(base_row['총자본회전률']) and pd.notna(raw_row[turnover_col]):
                    change = raw_row[turnover_col] - base_row['총자본회전률']
                    current_data.loc[idx, '총자본회전률_변화량'] = change
                    improved_count += 1
    
    print(f"✅ 변화량 개선 완료: {improved_count}개 값 보완")
    return current_data

def analyze_improvement(original_data, improved_data):
    """개선 효과 분석"""
    print("\n📊 개선 효과 분석...")
    
    # 성장률과 변화량 변수들
    all_vars = ['자산_YoY_성장률', '이익잉여금_YoY_성장률', '영업이익_YoY_성장률', '순이익_YoY_성장률',
               '부채비율_변화량', '유동비율_변화량', '총이익률_변화량', 'ROE_변화량', 
               '총자본회전률_변화량', '운전자본회전률_변화량']
    
    print("\n변수별 결측치 개선 현황:")
    print("=" * 80)
    print(f"{'변수명':<25} {'기존 결측':<12} {'개선 후':<12} {'개선량':<12} {'개선율':<12}")
    print("=" * 80)
    
    total_original_missing = 0
    total_improved_missing = 0
    
    for var in all_vars:
        if var in original_data.columns and var in improved_data.columns:
            original_missing = original_data[var].isna().sum()
            improved_missing = improved_data[var].isna().sum()
            improvement = original_missing - improved_missing
            improvement_rate = (improvement / original_missing * 100) if original_missing > 0 else 0
            
            total_original_missing += original_missing
            total_improved_missing += improved_missing
            
            print(f"{var:<25} {original_missing:<12} {improved_missing:<12} {improvement:<12} {improvement_rate:<11.1f}%")
    
    print("=" * 80)
    total_improvement = total_original_missing - total_improved_missing
    total_improvement_rate = (total_improvement / total_original_missing * 100) if total_original_missing > 0 else 0
    print(f"{'전체 합계':<25} {total_original_missing:<12} {total_improved_missing:<12} {total_improvement:<12} {total_improvement_rate:<11.1f}%")
    
    # 2012년 데이터만 분석
    data_2012_orig = original_data[original_data['회계년도'] == '2012/12']
    data_2012_impr = improved_data[improved_data['회계년도'] == '2012/12']
    
    print(f"\n2012년 데이터 개선 현황 (총 {len(data_2012_orig)}개 레코드):")
    print("=" * 80)
    print(f"{'변수명':<25} {'기존 결측':<12} {'개선 후':<12} {'개선량':<12} {'개선율':<12}")
    print("=" * 80)
    
    for var in all_vars:
        if var in data_2012_orig.columns and var in data_2012_impr.columns:
            original_missing = data_2012_orig[var].isna().sum()
            improved_missing = data_2012_impr[var].isna().sum()
            improvement = original_missing - improved_missing
            improvement_rate = (improvement / original_missing * 100) if original_missing > 0 else 0
            
            print(f"{var:<25} {original_missing:<12} {improved_missing:<12} {improvement:<12} {improvement_rate:<11.1f}%")

def main():
    """메인 실행 함수"""
    print("🚀 2011년 데이터를 활용한 성장률 및 변화량 재계산 시작")
    print("=" * 80)
    
    # 1. 데이터 로드
    bs2011, current_data, fs_flow, fs_raw = load_data()
    original_data = current_data.copy()  # 원본 데이터 백업
    
    # 2. 2011년 데이터 전처리
    bs2011_clean = prepare_2011_data(bs2011)
    
    # 3. 성장률 재계산
    current_data = calculate_growth_rates(current_data, bs2011_clean, fs_flow, fs_raw)
    
    # 4. 변화량 재계산
    current_data = calculate_changes(current_data, bs2011_clean, fs_raw)
    
    # 5. 개선 효과 분석
    analyze_improvement(original_data, current_data)
    
    # 6. 개선된 데이터 저장
    output_path = 'data/final/FS_ratio_flow_improved.csv'
    current_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 개선된 데이터 저장: {output_path}")
    
    # 7. 기존 파일 백업
    backup_path = 'data/final/FS_ratio_flow_enhanced_backup.csv'
    original_data.to_csv(backup_path, index=False, encoding='utf-8-sig')
    print(f"📁 원본 데이터 백업: {backup_path}")
    
    print("\n🎉 성장률 및 변화량 재계산 완료!")

if __name__ == "__main__":
    main() 