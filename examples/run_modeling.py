#!/usr/bin/env python3
"""
모델링 파이프라인 실행 스크립트
===============================

전처리된 데이터를 기반으로 모델링 파이프라인을 실행합니다.

사용법:
    python run_modeling.py --config config/modeling_config.yaml
    python run_modeling.py --config config/modeling_config.yaml --quick-test
    python run_modeling.py --ensemble-only
"""

import argparse
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.modeling.modeling_pipeline import ModelingPipeline
from src.modeling.ensemble_pipeline import EnsemblePipeline


def create_quick_test_config(base_config_path: str) -> str:
    """빠른 테스트용 설정 생성"""
    import yaml
    import json
    from pathlib import Path
    
    # 기본 설정 로드
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 빠른 테스트용 수정
    config['experiment']['name'] = "quick_test_modeling"
    
    # 샘플링 전략 간소화 (normal과 combined만)
    config['sampling']['data_types']['undersampling']['enabled'] = False
    config['sampling']['data_types']['smote']['enabled'] = False
    
    # 특성 선택 비활성화 (빠른 실행을 위해)
    config['feature_selection']['enabled'] = False
    
    # 모델별 trial 수 감소
    for model_name in config['models']:
        if 'n_trials' in config['models'][model_name]:
            config['models'][model_name]['n_trials'] = 10
    
    # 앙상블 활성화 (빠른 테스트에서도 앙상블 테스트)
    config['ensemble']['enabled'] = True
    
    # 임시 설정 파일 저장
    temp_config_path = project_root / "examples" / "temp_quick_test_config.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    return str(temp_config_path)


def run_ensemble_only(data_path: str, models_path: str, config_path: str):
    """기존 모델들을 로드하여 앙상블만 실행"""
    print("🎭 앙상블 전용 실행 모드")
    print("="*60)
    
    import joblib
    import pandas as pd
    import yaml
    import json
    
    # 설정 로드
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 데이터 로드
    data_dir = Path(data_path)
    X_val = pd.read_csv(data_dir / config['data']['files']['X_val'])
    y_val = pd.read_csv(data_dir / config['data']['files']['y_val']).iloc[:, 0]
    X_test = pd.read_csv(data_dir / config['data']['files']['X_test'])
    y_test = pd.read_csv(data_dir / config['data']['files']['y_test']).iloc[:, 0]
    
    # 모델 로드
    models_dir = Path(models_path)
    models = {}
    
    for model_file in models_dir.glob("*.joblib"):
        model_key = model_file.stem.replace('_model', '')
        try:
            model = joblib.load(model_file)
            models[model_key] = model
            print(f"✅ 모델 로드: {model_key}")
        except Exception as e:
            print(f"⚠️ 모델 로드 실패 ({model_key}): {e}")
    
    if not models:
        print("❌ 로드할 모델이 없습니다.")
        return
    
    # 앙상블 실행
    ensemble = EnsemblePipeline(config, models)
    
    # 최적 threshold 찾기
    optimal_threshold, threshold_analysis = ensemble.find_optimal_threshold(
        X_val, y_val, metric=config.get('ensemble', {}).get('threshold_optimization', {}).get('metric_priority', 'f1')
    )
    
    # 최종 평가
    test_metrics = ensemble.evaluate_ensemble(X_test, y_test, optimal_threshold)
    
    # 결과 출력
    print(f"\n🏆 앙상블 최종 결과:")
    print(f"   최적 Threshold: {optimal_threshold:.3f}")
    print(f"   Test AUC: {test_metrics['auc']:.4f}")
    print(f"   Test F1: {test_metrics['f1']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall: {test_metrics['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='모델링 파이프라인 실행')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/modeling_config.yaml',
        help='설정 파일 경로 (기본값: config/modeling_config.yaml)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='빠른 테스트 모드 (trial 수 감소, 기본 설정 간소화)'
    )
    parser.add_argument(
        '--ensemble-only',
        action='store_true',
        help='앙상블만 실행 (기존 모델들 필요)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/final',
        help='데이터 경로 (앙상블 전용 모드에서 사용)'
    )
    parser.add_argument(
        '--models-path',
        type=str,
        default='outputs/modeling_runs/latest/models',
        help='모델 경로 (앙상블 전용 모드에서 사용)'
    )
    
    args = parser.parse_args()
    
    # 설정 파일 존재 확인
    if not os.path.exists(args.config):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    try:
        # 앙상블 전용 모드
        if args.ensemble_only:
            run_ensemble_only(args.data_path, args.models_path, args.config)
            return
        
        # 빠른 테스트 모드
        if args.quick_test:
            print("⚡ 빠른 테스트 모드 활성화")
            config_path = create_quick_test_config(args.config)
            print(f"📝 임시 설정 파일 생성: {config_path}")
        else:
            config_path = args.config
        
        print("🏢 한국 기업 부실예측 - 모델링 파이프라인")
        print("="*80)
        print(f"📋 설정 파일: {config_path}")
        
        # 확인 메시지 (빠른 테스트가 아닌 경우)
        if not args.quick_test:
            response = input("\n실행하시겠습니까? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                print("실행이 취소되었습니다.")
                sys.exit(0)
        
        # 파이프라인 실행
        pipeline = ModelingPipeline(config_path)
        experiment_dir = pipeline.run_pipeline()
        
        print(f"\n🎉 모델링 파이프라인 완료!")
        print(f"📁 결과 저장 위치: {experiment_dir}")
        print("="*80)
        
        # 임시 설정 파일 삭제
        if args.quick_test:
            temp_config = Path(config_path)
            if temp_config.exists():
                temp_config.unlink()
                print(f"🗑️ 임시 설정 파일 삭제: {config_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()