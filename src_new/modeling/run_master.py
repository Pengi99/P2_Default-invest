#!/usr/bin/env python3
"""
마스터 모델 러너 실행 스크립트
"""

import argparse
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from master_model_runner import MasterModelRunner, load_config

def main():
    parser = argparse.ArgumentParser(description='마스터 모델 러너 실행')
    parser.add_argument(
        '--config', 
        type=str, 
        default='src_new/modeling/master_config.json',
        help='설정 파일 경로 (기본값: master_config.json)'
    )
    parser.add_argument(
        '--template',
        type=str,
        choices=['quick', 'production', 'lasso'],
        help='사전 정의된 템플릿 사용 (quick/production/lasso)'
    )
    
    args = parser.parse_args()
    
    # 템플릿 선택 시 해당 설정 파일 사용
    if args.template:
        template_map = {
            'quick': 'src_new/modeling/config_templates/quick_test_config.json',
            'production': 'src_new/modeling/config_templates/production_config.json',
            'lasso': 'src_new/modeling/config_templates/lasso_focus_config.json'
        }
        config_path = template_map[args.template]
        print(f"🔧 템플릿 사용: {args.template} ({config_path})")
    else:
        config_path = args.config
        print(f"🔧 설정 파일: {config_path}")
    
    # 설정 파일 존재 확인
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)
    
    try:
        # 설정 로드
        config = load_config(config_path)
        
        print("🏢 한국 기업 부실예측 - 마스터 모델 러너")
        print("="*80)
        print(f"📋 실행 이름: {config['run_name']}")
        print(f"🎲 랜덤 시드: {config['random_state']}")
        print(f"🎯 Threshold 최적화: {config.get('threshold_optimization', {}).get('enabled', False)}")
        if config.get('threshold_optimization', {}).get('enabled', False):
            metric = config['threshold_optimization'].get('metric_priority', 'f1')
            print(f"📊 우선순위 메트릭: {metric.upper()}")
        print(f"🔍 Lasso 활성화: {config['lasso']['enabled']}")
        
        # 활성화된 모델 확인
        enabled_models = [name for name, settings in config['models'].items() if settings['enabled']]
        print(f"🤖 활성화된 모델: {', '.join(enabled_models)}")
        
        # 확인 메시지
        response = input("\n실행하시겠습니까? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("실행이 취소되었습니다.")
            sys.exit(0)
        
        # 러너 생성 및 실행
        runner = MasterModelRunner(config)
        runner.load_data()
        runner.run_all_models()
        runner.save_all_results()
        
        print(f"\n🎉 모든 모델 실행 완료!")
        print(f"📁 결과 저장 위치: {runner.output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 