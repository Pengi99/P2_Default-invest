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
    
    # 피처 선택 결과 (활성화된 경우에만)
    if 'feature_selection' in pipeline.results['preprocessing_steps']:
        print(f"\n🎯 피처 선택 결과:")
        print(f"   원본 피처 수: {pipeline.results['preprocessing_steps']['feature_selection']['original_features']}")
        print(f"   선택된 피처 수: {pipeline.results['preprocessing_steps']['feature_selection']['selected_features']}")
        print(f"   선택률: {pipeline.results['preprocessing_steps']['feature_selection']['selected_features'] / pipeline.results['preprocessing_steps']['feature_selection']['original_features'] * 100:.1f}%")
        
        print(f"\n📈 모델 성능:")
        print(f"   검증 R²: {pipeline.results['model_performance']['val_r2']:.4f}")
        print(f"   테스트 R²: {pipeline.results['model_performance']['test_r2']:.4f}")
    else:
        print(f"\n🎯 피처 선택:")
        print(f"   상태: 비활성화됨")
        print(f"   모든 피처가 유지됨")
    
    print(f"\n📝 생성된 파일들:")
    files = [
        "X_train.csv",
        "y_train.csv",
        "X_val.csv",
        "y_val.csv",
        "X_test.csv", 
        "y_test.csv",
        "scaler_standard.pkl",
        "scaler_robust.pkl",
        "preprocessing_report.txt"
    ]
    
    # 피처 선택이 활성화된 경우에만 feature_selector.pkl 체크
    if 'feature_selection' in pipeline.results['preprocessing_steps']:
        files.append("feature_selector.pkl")
    
    for file in files:
        file_path = Path(experiment_dir) / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    
    # 피처 선택이 활성화된 경우에만 선택된 피처 목록 출력
    if 'feature_selection' in pipeline.results['preprocessing_steps'] and pipeline.results['selected_features']:
        print(f"\n🔍 선택된 주요 피처들 (상위 10개):")
        selected_features = pipeline.results['selected_features'][:10]
        for i, feature in enumerate(selected_features, 1):
            print(f"   {i:2d}. {feature}")
        
        if len(pipeline.results['selected_features']) > 10:
            print(f"   ... 외 {len(pipeline.results['selected_features']) - 10}개")
    else:
        print(f"\n🔍 피처 정보:")
        # 스케일링 단계에서 피처 컬럼 정보 가져오기
        feature_cols = pipeline.results['preprocessing_steps']['scaling']['feature_columns']
        print(f"   전체 피처 수: {len(feature_cols)}개")
        print(f"   주요 피처들 (처음 10개):")
        for i, feature in enumerate(feature_cols[:10], 1):
            print(f"   {i:2d}. {feature}")
        
        if len(feature_cols) > 10:
            print(f"   ... 외 {len(feature_cols) - 10}개")
    
    return experiment_dir

if __name__ == "__main__":
    experiment_dir = main()
    print(f"\n🎉 완료! 결과를 확인하세요: {experiment_dir}")