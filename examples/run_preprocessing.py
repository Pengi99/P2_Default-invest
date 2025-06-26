"""
데이터 전처리 파이프라인 실행 예제
================================

간단한 실행 스크립트로 전체 전처리 과정을 수행합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing.data_pipeline import DataPreprocessingPipeline

def main():
    """전처리 파이프라인 실행"""
    
    print("🚀 데이터 전처리 파이프라인을 시작합니다...")
    
    # Config 파일 경로
    config_path = project_root / "config" / "preprocessing_config.yaml"
    
    # 파이프라인 초기화 및 실행
    pipeline = DataPreprocessingPipeline(str(config_path))
    experiment_dir = pipeline.run_pipeline()
    
    # 결과 요약 출력
    print("\n" + "="*60)
    print("✅ 전처리 파이프라인이 성공적으로 완료되었습니다!")
    print("="*60)
    
    print(f"\n📁 결과 저장 위치:")
    print(f"   {experiment_dir}")
    
    print(f"\n📊 데이터 정보:")
    print(f"   원본 데이터: {pipeline.results['data_info']['original_shape']}")
    print(f"   Train: {pipeline.results['preprocessing_steps']['data_split']['train_shape']}")
    print(f"   Validation: {pipeline.results['preprocessing_steps']['data_split']['val_shape']}")
    print(f"   Test: {pipeline.results['preprocessing_steps']['data_split']['test_shape']}")
    
    print(f"\n🔧 전처리 단계:")
    print(f"   결측치 처리: {pipeline.results['preprocessing_steps']['missing_data']['method']}")
    print(f"   윈저라이징: {'활성화' if pipeline.results['preprocessing_steps']['winsorization']['enabled'] else '비활성화'}")
    print(f"   피처 선택: 비활성화됨 (모든 피처 유지)")
    print(f"   스케일링: 비활성화됨 (원본 값 유지)")
    
    print(f"\n📝 생성된 파일들:")
    files = [
        "X_train.csv",
        "y_train.csv",
        "X_val.csv",
        "y_val.csv",
        "X_test.csv", 
        "y_test.csv",
        "preprocessing_report.txt"
    ]
    
    for file in files:
        file_path = Path(experiment_dir) / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    
    # 피처 정보 출력 (데이터에서 직접 확인)
    try:
        import pandas as pd
        
        # 실제 저장된 파일에서 피처 정보 가져오기
        X_train_path = Path(experiment_dir) / "X_train.csv"
        if X_train_path.exists():
            X_train = pd.read_csv(X_train_path)
            feature_cols = list(X_train.columns)
            
            print(f"\n🔍 피처 정보:")
            print(f"   전체 피처 수: {len(feature_cols)}개")
            print(f"   주요 피처들 (처음 10개):")
            for i, feature in enumerate(feature_cols[:10], 1):
                print(f"   {i:2d}. {feature}")
            
            if len(feature_cols) > 10:
                print(f"   ... 외 {len(feature_cols) - 10}개")
        else:
            print(f"\n🔍 피처 정보:")
            print(f"   피처 파일을 찾을 수 없어 정보를 표시할 수 없습니다.")
            
    except Exception as e:
        print(f"\n🔍 피처 정보:")
        print(f"   피처 정보 로드 중 오류: {e}")
    
    return experiment_dir

if __name__ == "__main__":
    experiment_dir = main()
    print(f"\n🎉 완료! 결과를 확인하세요: {experiment_dir}")