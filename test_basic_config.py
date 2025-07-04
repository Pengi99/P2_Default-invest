"""
기본 설정 수정 후 간단 테스트
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """데이터 로딩 테스트"""
    print("🔍 데이터 로딩 테스트 시작...")
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Test fundamental data loading
    fs_path = config['data_paths']['fundamental']
    print(f"📊 재무 데이터 로딩: {fs_path}")
    
    if os.path.exists(fs_path):
        fs_df = pd.read_csv(fs_path, encoding='utf-8-sig')
        print(f"   ✅ 성공: {len(fs_df):,}행, {len(fs_df.columns)}컬럼")
        
        # Check key columns for F-score
        key_cols = ['거래소코드', '연도', '당기순이익', '총자산', '영업현금흐름']
        missing_cols = [col for col in key_cols if col not in fs_df.columns]
        if missing_cols:
            print(f"   ⚠️ 누락 컬럼: {missing_cols}")
            # Show similar columns
            all_cols = list(fs_df.columns)
            for missing in missing_cols:
                similar = [col for col in all_cols if missing.replace('현금흐름', '') in col or col in missing]
                if similar:
                    print(f"      '{missing}' 대신 가능한 컬럼: {similar[:3]}")
        else:
            print(f"   ✅ 필수 컬럼 모두 존재")
    else:
        print(f"   ❌ 파일 없음: {fs_path}")
        return False
    
    # Test market cap data
    cap_path = config['data_paths']['market_cap']
    print(f"💰 시가총액 데이터 로딩: {cap_path}")
    
    if os.path.exists(cap_path):
        cap_df = pd.read_csv(cap_path, encoding='utf-8-sig')
        print(f"   ✅ 성공: {len(cap_df):,}행")
    else:
        print(f"   ❌ 파일 없음: {cap_path}")
        return False
    
    # Test price data directory
    price_dir = config['data_paths']['price_data_dir']
    print(f"📈 가격 데이터 디렉토리: {price_dir}")
    
    if os.path.exists(price_dir):
        csv_files = [f for f in os.listdir(price_dir) if f.endswith('.csv')]
        print(f"   ✅ {len(csv_files)}개 CSV 파일 발견")
        
        # Test sample file
        if csv_files:
            sample_file = f"{price_dir}/{csv_files[0]}"
            sample_df = pd.read_csv(sample_file, encoding='utf-8-sig', nrows=100)
            
            required_price_cols = ['거래소코드', '매매년월일', '시가총액(원)']
            missing_price_cols = [col for col in required_price_cols if col not in sample_df.columns]
            
            if missing_price_cols:
                print(f"   ⚠️ 가격 데이터 누락 컬럼: {missing_price_cols}")
                print(f"   실제 컬럼: {list(sample_df.columns)}")
            else:
                print(f"   ✅ 가격 데이터 필수 컬럼 존재")
    else:
        print(f"   ❌ 디렉토리 없음: {price_dir}")
        return False
    
    return True

def test_factor_calculation():
    """간단한 팩터 계산 테스트"""
    print("\n🧮 팩터 계산 테스트...")
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load fundamental data
    fs_path = config['data_paths']['fundamental']
    fs_df = pd.read_csv(fs_path, encoding='utf-8-sig')
    
    print(f"📊 재무 데이터: {len(fs_df):,}행")
    
    # Test F-score components
    print("F-Score 컴포넌트 테스트:")
    
    # ROA test
    if 'ROA' in fs_df.columns:
        roa_positive = (fs_df['ROA'] > 0).sum()
        print(f"   - ROA > 0: {roa_positive:,}개 ({roa_positive/len(fs_df)*100:.1f}%)")
    elif '당기순이익' in fs_df.columns and '총자산' in fs_df.columns:
        net_income = pd.to_numeric(fs_df['당기순이익'], errors='coerce')
        total_assets = pd.to_numeric(fs_df['총자산'], errors='coerce').replace(0, np.nan)
        roa = net_income / total_assets
        roa_positive = (roa > 0).sum()
        print(f"   - 계산된 ROA > 0: {roa_positive:,}개 ({roa_positive/len(fs_df)*100:.1f}%)")
    
    # CFO test  
    cfo_col = '영업현금흐름' if '영업현금흐름' in fs_df.columns else '영업CF'
    if cfo_col in fs_df.columns:
        cfo_positive = (pd.to_numeric(fs_df[cfo_col], errors='coerce') > 0).sum()
        print(f"   - {cfo_col} > 0: {cfo_positive:,}개 ({cfo_positive/len(fs_df)*100:.1f}%)")
    else:
        print(f"   ⚠️ CFO 컬럼 없음")
    
    # Check data years
    if '연도' in fs_df.columns:
        years = fs_df['연도'].value_counts().sort_index()
        print(f"   - 연도별 데이터: {dict(years)}")
        
        if len(years) >= 2:
            # Test year-over-year calculation
            companies = fs_df['거래소코드'].unique()[:10]  # Test with 10 companies
            test_df = fs_df[fs_df['거래소코드'].isin(companies)].copy()
            test_df = test_df.sort_values(['거래소코드', '연도'])
            
            if 'ROA' in test_df.columns:
                test_df['roa_change'] = test_df.groupby('거래소코드')['ROA'].diff()
                positive_changes = (test_df['roa_change'] > 0).sum()
                total_changes = test_df['roa_change'].notna().sum()
                
                if total_changes > 0:
                    print(f"   - ROA 개선: {positive_changes}/{total_changes} ({positive_changes/total_changes*100:.1f}%)")
        else:
            print(f"   ⚠️ 연도별 비교 불가 (1년 데이터만 존재)")
    
    return True

def main():
    """메인 테스트 실행"""
    print("🔬 백테스팅 설정 수정 후 기본 테스트")
    print("=" * 50)
    
    try:
        # Test 1: Data loading
        if not test_data_loading():
            print("\n❌ 데이터 로딩 실패")
            return False
        
        # Test 2: Factor calculation
        if not test_factor_calculation():
            print("\n❌ 팩터 계산 실패")
            return False
        
        print("\n🎉 기본 테스트 통과!")
        print("다음 단계: 실제 백테스팅 실행 가능")
        
        return True
        
    except Exception as e:
        print(f"\n💥 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)