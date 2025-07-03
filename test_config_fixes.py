#!/usr/bin/env python3
"""
Config 오류 수정 테스트
====================

수정된 config 접근이 안전하게 작동하는지 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_config_initialization():
    """Config 초기화 테스트"""
    print("🧪 Config 초기화 테스트 시작")
    print("="*50)
    
    try:
        from src.modeling.modeling_pipeline import ModelingPipeline
        
        # 빈 config로 테스트
        empty_config = {}
        
        # 임시 파일 생성
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(empty_config, f)
            temp_config_path = f.name
        
        try:
            # 파이프라인 초기화 시도
            pipeline = ModelingPipeline(temp_config_path)
            print("✅ 빈 config로도 초기화 성공!")
            print(f"   - run_name: {pipeline.run_name}")
            print(f"   - output_dir: {pipeline.output_dir}")
            print(f"   - logger: {pipeline.logger.name}")
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            return False
            
        finally:
            # 임시 파일 삭제
            os.unlink(temp_config_path)
            
        return True
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        return False

def test_data_loading():
    """데이터 로딩 테스트"""
    print("\n🧪 데이터 로딩 안전성 테스트")
    print("="*50)
    
    try:
        from src.modeling.modeling_pipeline import ModelingPipeline
        
        # 최소 config
        config = {
            'experiment': {'name': 'test'},
            'output': {'base_dir': 'test_output'},
            'logging': {'level': 'INFO', 'save_to_file': False}
        }
        
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            pipeline = ModelingPipeline(temp_config_path)
            print("✅ 최소 config로 초기화 성공!")
            
            # 존재하지 않는 데이터 경로로 테스트
            try:
                pipeline.load_data()
                print("❌ 존재하지 않는 데이터로 로딩이 성공해서는 안됨")
                return False
            except Exception as e:
                print(f"✅ 예상된 데이터 로딩 실패: {type(e).__name__}")
                return True
                
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🔧 Config 오류 수정 검증 테스트")
    print("="*70)
    
    success_count = 0
    total_tests = 2
    
    # 테스트 실행
    if test_config_initialization():
        success_count += 1
    
    if test_data_loading():
        success_count += 1
    
    # 결과 요약
    print(f"\n📊 테스트 결과: {success_count}/{total_tests} 성공")
    
    if success_count == total_tests:
        print("🎉 모든 config 오류가 수정되었습니다!")
        return True
    else:
        print("⚠️ 일부 config 오류가 남아있을 수 있습니다.")
        return False

if __name__ == "__main__":
    main()