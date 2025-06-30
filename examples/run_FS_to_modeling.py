#!/usr/bin/env python3
"""
FS.csv부터 모델링까지 전체 파이프라인 실행 스크립트
=================================================

다음 단계를 순차적으로 실행합니다:
1. FS.csv 컬럼 필터링 (column_manager.py)
2. 데이터 전처리 (run_preprocessing.py)  
3. 모델링 파이프라인 (run_modeling.py)

사용법:
    python run_FS_to_modeling.py
    python run_FS_to_modeling.py --quick-test
    python run_FS_to_modeling.py --skip-filtering
    python run_FS_to_modeling.py --preprocessing-only
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 각 단계별 모듈 import
from src.data_processing.column_manager import process_fs_data
from src.preprocessing.data_pipeline import DataPreprocessingPipeline
from src.modeling.modeling_pipeline import ModelingPipeline


class FullPipelineRunner:
    """전체 파이프라인 실행 클래스"""
    
    def __init__(self, quick_test=False, skip_filtering=False):
        self.quick_test = quick_test
        self.skip_filtering = skip_filtering
        self.start_time = time.time()
        self.step_times = {}
        
        # 경로 설정
        self.project_root = project_root
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        
        print("🚀 FS.csv → 모델링 전체 파이프라인")
        print("="*80)
        print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚡ 빠른 테스트 모드: {'활성화' if quick_test else '비활성화'}")
        print(f"🔄 컬럼 필터링: {'건너뛰기' if skip_filtering else '실행'}")
        print("="*80)
    
    def log_step_start(self, step_name: str, description: str):
        """단계 시작 로그"""
        print(f"\n{'='*60}")
        print(f"🔄 단계 {step_name}: {description}")
        print(f"{'='*60}")
        self.step_times[step_name] = time.time()
    
    def log_step_end(self, step_name: str, success: bool = True):
        """단계 완료 로그"""
        elapsed = time.time() - self.step_times[step_name]
        status = "✅ 완료" if success else "❌ 실패"
        print(f"\n{status} - 단계 {step_name} ({elapsed:.1f}초)")
        
        if not success:
            raise Exception(f"단계 {step_name} 실패")
    
    def step1_column_filtering(self):
        """1단계: FS.csv 컬럼 필터링"""
        if self.skip_filtering:
            print("\n⏭️ 컬럼 필터링 단계를 건너뜁니다.")
            return True
        
        self.log_step_start("1", "FS.csv 컬럼 필터링")
        
        try:
            # FS.csv 파일 존재 확인
            fs_path = self.data_dir / "processed" / "FS.csv"
            if not fs_path.exists():
                print(f"❌ FS.csv 파일을 찾을 수 없습니다: {fs_path}")
                return False
            
            # 컬럼 필터링 실행
            success = process_fs_data()
            
            if success:
                # 결과 파일 확인
                filtered_path = self.data_dir / "processed" / "FS_filtered.csv"
                if filtered_path.exists():
                    print(f"✅ 필터링된 파일 생성: {filtered_path}")
                    
                    # 간단한 통계 출력
                    import pandas as pd
                    df = pd.read_csv(filtered_path)
                    print(f"📊 필터링 결과: {df.shape[0]:,}행 × {df.shape[1]}열")
                else:
                    print("❌ 필터링된 파일이 생성되지 않았습니다.")
                    return False
            
            self.log_step_end("1", success)
            return success
            
        except Exception as e:
            print(f"❌ 컬럼 필터링 중 오류: {e}")
            self.log_step_end("1", False)
            return False
    
    def step2_preprocessing(self):
        """2단계: 데이터 전처리"""
        self.log_step_start("2", "데이터 전처리")
        
        try:
            # 전처리 설정 파일 확인
            preprocessing_config = self.config_dir / "preprocessing_config.yaml"
            if not preprocessing_config.exists():
                print(f"❌ 전처리 설정 파일을 찾을 수 없습니다: {preprocessing_config}")
                return False
            
            # 전처리 파이프라인 실행
            print(f"📋 설정 파일: {preprocessing_config}")
            pipeline = DataPreprocessingPipeline(str(preprocessing_config))
            experiment_dir = pipeline.run_pipeline()
            
            print(f"📁 전처리 결과 저장: {experiment_dir}")
            
            # 결과 검증
            result_files = [
                "X_train.csv", "y_train.csv",
                "X_val.csv", "y_val.csv", 
                "X_test.csv", "y_test.csv"
            ]
            
            missing_files = []
            for file in result_files:
                file_path = Path(experiment_dir) / file
                if not file_path.exists():
                    missing_files.append(file)
            
            if missing_files:
                print(f"❌ 누락된 파일들: {missing_files}")
                return False
            
            # 전처리 결과 정보 출력
            print(f"\n📊 전처리 결과:")
            print(f"   Train: {pipeline.results['preprocessing_steps']['data_split']['train_shape']}")
            print(f"   Validation: {pipeline.results['preprocessing_steps']['data_split']['val_shape']}")
            print(f"   Test: {pipeline.results['preprocessing_steps']['data_split']['test_shape']}")
            
            self.preprocessing_dir = experiment_dir
            self.log_step_end("2", True)
            return True
            
        except Exception as e:
            print(f"❌ 전처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            self.log_step_end("2", False)
            return False
    
    def step3_modeling(self):
        """3단계: 모델링"""
        self.log_step_start("3", "모델링 파이프라인")
        
        try:
            # 모델링 설정 파일 준비
            modeling_config = self.config_dir / "modeling_config.yaml"
            if not modeling_config.exists():
                print(f"❌ 모델링 설정 파일을 찾을 수 없습니다: {modeling_config}")
                return False
            
            # 빠른 테스트 모드인 경우 설정 수정
            if self.quick_test:
                config_path = self.create_quick_test_config(str(modeling_config))
                print(f"⚡ 빠른 테스트 설정 생성: {config_path}")
            else:
                config_path = str(modeling_config)
            
            print(f"📋 설정 파일: {config_path}")
            
            # 전처리 결과 경로 계산 (임시 파일 없이 동적 전달)
            data_path_override = None
            if hasattr(self, 'preprocessing_dir'):
                relative_path = Path(self.preprocessing_dir).relative_to(self.project_root)
                data_path_override = str(relative_path)
                print(f"📝 데이터 경로 오버라이드: {data_path_override}")
            
            # 모델링 파이프라인 실행 (동적 경로 전달)
            pipeline = ModelingPipeline(config_path, data_path_override=data_path_override)
            experiment_dir = pipeline.run_pipeline()
            
            print(f"📁 모델링 결과 저장: {experiment_dir}")
            
            # 결과 요약 출력
            self.print_modeling_summary(pipeline)
            
            self.modeling_dir = experiment_dir
            self.log_step_end("3", True)
            
            return True
            
        except Exception as e:
            print(f"❌ 모델링 중 오류: {e}")
            import traceback
            traceback.print_exc()
            self.log_step_end("3", False)
            return False
    
    def create_quick_test_config(self, base_config_path: str) -> str:
        """빠른 테스트용 설정 생성"""
        import yaml
        
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 빠른 테스트용 수정
        config['experiment']['name'] = "quick_test_full_pipeline"
        
        # 샘플링 전략 간소화
        config['sampling']['data_types']['undersampling']['enabled'] = False
        config['sampling']['data_types']['combined']['enabled'] = False
        
        # 특성 선택 비활성화
        config['feature_selection']['enabled'] = False
        
        # 모델별 trial 수 감소
        if 'models' in config:
            for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                if model_name in config['models'] and 'n_trials' in config['models'][model_name]:
                    config['models'][model_name]['n_trials'] = 20
        
        # 임시 설정 파일 저장
        temp_config_path = self.project_root / "examples" / "temp_full_pipeline_config.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return str(temp_config_path)

    
    def print_modeling_summary(self, pipeline):
        """모델링 결과 요약 출력"""
        try:
            if hasattr(pipeline, 'results') and pipeline.results:
                print(f"\n🏆 모델링 결과 요약:")
                
                # 최고 성능 모델 찾기
                best_model = None
                best_score = 0
                
                for model_name, result in pipeline.results.items():
                    if isinstance(result, dict) and 'test_metrics' in result:
                        test_auc = result['test_metrics'].get('auc', 0)
                        if test_auc > best_score:
                            best_score = test_auc
                            best_model = model_name
                
                if best_model:
                    best_result = pipeline.results[best_model]
                    print(f"   🥇 최고 성능: {best_model}")
                    print(f"   📊 Test AUC: {best_result['test_metrics']['auc']:.4f}")
                    print(f"   📊 Test F1: {best_result['test_metrics']['f1']:.4f}")
                    print(f"   📊 Test Precision: {best_result['test_metrics']['precision']:.4f}")
                    print(f"   📊 Test Recall: {best_result['test_metrics']['recall']:.4f}")
                
                # 앙상블 결과 (있는 경우)
                if 'ensemble' in pipeline.results:
                    ensemble_result = pipeline.results['ensemble']
                    print(f"\n🎭 앙상블 결과:")
                    print(f"   📊 Test AUC: {ensemble_result['test_metrics']['auc']:.4f}")
                    print(f"   📊 Test F1: {ensemble_result['test_metrics']['f1']:.4f}")
                    print(f"   📊 최적 Threshold: {ensemble_result.get('optimal_threshold', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ 결과 요약 출력 중 오류: {e}")
    
    def run_full_pipeline(self, preprocessing_only=False):
        """전체 파이프라인 실행"""
        try:
            # 1단계: 컬럼 필터링
            if not self.step1_column_filtering():
                return False
            
            # 2단계: 전처리
            if not self.step2_preprocessing():
                return False
            
            # 전처리만 실행하는 경우 여기서 종료
            if preprocessing_only:
                print(f"\n✅ 전처리까지 완료! (--preprocessing-only 옵션)")
                self.print_final_summary(preprocessing_only=True)
                return True
            
            # 3단계: 모델링
            if not self.step3_modeling():
                return False
            
            # 최종 성공
            self.print_final_summary()
            return True
            
        except Exception as e:
            print(f"\n❌ 파이프라인 실행 중 오류: {e}")
            return False
    
    def print_final_summary(self, preprocessing_only=False):
        """최종 요약 출력"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"🎉 {'전처리' if preprocessing_only else '전체'} 파이프라인 완료!")
        print(f"{'='*80}")
        print(f"⏱️ 총 실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        
        # 단계별 시간
        print(f"\n📊 단계별 실행 시간:")
        for step, start_time in self.step_times.items():
            if step in ["1", "2", "3"]:
                step_name = {
                    "1": "컬럼 필터링",
                    "2": "데이터 전처리", 
                    "3": "모델링"
                }[step]
                
                # 해당 단계가 완료되었는지 확인
                step_time = "진행 중..." 
                for i, (s, t) in enumerate(self.step_times.items()):
                    if s == step and i < len(self.step_times) - 1:
                        next_time = list(self.step_times.values())[i + 1]
                        step_time = f"{next_time - t:.1f}초"
                        break
                    elif s == step and i == len(self.step_times) - 1:
                        step_time = f"{time.time() - t:.1f}초"
                        break
                
                print(f"   {step}. {step_name}: {step_time}")
        
        # 결과 파일 위치
        print(f"\n📁 결과 파일 위치:")
        if hasattr(self, 'preprocessing_dir'):
            print(f"   전처리 결과: {self.preprocessing_dir}")
        if hasattr(self, 'modeling_dir') and not preprocessing_only:
            print(f"   모델링 결과: {self.modeling_dir}")
        
        print(f"\n🎯 다음 단계:")
        if preprocessing_only:
            print(f"   - 전처리된 데이터로 모델링 실행")
            print(f"   - python examples/run_modeling.py")
        else:
            print(f"   - 모델링 결과 분석 및 해석")
            print(f"   - 시각화 결과 확인")
            print(f"   - 모델 성능 개선 실험")


def main():
    parser = argparse.ArgumentParser(description='FS.csv부터 모델링까지 전체 파이프라인 실행')
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='빠른 테스트 모드 (trial 수 감소, 설정 간소화)'
    )
    parser.add_argument(
        '--skip-filtering',
        action='store_true',
        help='컬럼 필터링 단계 건너뛰기 (FS_filtered.csv가 이미 있는 경우)'
    )
    parser.add_argument(
        '--preprocessing-only',
        action='store_true',
        help='전처리까지만 실행 (모델링 제외)'
    )
    
    args = parser.parse_args()
    
    try:
        # 파이프라인 실행기 생성
        runner = FullPipelineRunner(
            quick_test=args.quick_test,
            skip_filtering=args.skip_filtering
        )
        
        # 실행 확인 (빠른 테스트가 아닌 경우)
        if not args.quick_test:
            mode_desc = "전처리까지만" if args.preprocessing_only else "전체 파이프라인을"
            response = input(f"\n{mode_desc} 실행하시겠습니까? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                print("실행이 취소되었습니다.")
                sys.exit(0)
        
        # 전체 파이프라인 실행
        success = runner.run_full_pipeline(preprocessing_only=args.preprocessing_only)
        
        if success:
            print(f"\n✨ 모든 작업이 성공적으로 완료되었습니다!")
            sys.exit(0)
        else:
            print(f"\n💥 파이프라인 실행 중 오류가 발생했습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
