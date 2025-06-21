import subprocess
import sys
import time
from datetime import datetime
import os

def run_step(step_file, step_name):
    """단계별 스크립트 실행"""
    print(f"\n{'='*60}")
    print(f"🚀 {step_name} 실행 시작")
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # archive_old_structure/src 폴더에서 실행 (기존 스크립트들 위치)
        result = subprocess.run([sys.executable, f'archive_old_structure/src/{step_file}'], 
                              capture_output=True, 
                              text=True, 
                              encoding='utf-8')
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ 경고/오류:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {step_name} 완료 (소요시간: {duration:.1f}초)")
            return True
        else:
            print(f"❌ {step_name} 실패 (반환코드: {result.returncode})")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {step_name} 실행 중 오류 발생: {e}")
        print(f"소요시간: {duration:.1f}초")
        return False

def main():
    """메인 실행 함수"""
    print("🎯 재무비율 계산 프로세스 시작 (FS_flow 기반)")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start_time = time.time()
    
    # 실행할 단계들
    steps = [
        ("step1_basic_financial_ratios.py", "1단계: 기본 재무비율 계산 (FS_flow 활용)"),
        ("step2_market_based_ratios.py", "2단계: 시장기반 재무비율 계산"),
        ("step3_volatility_returns.py", "3단계: 변동성(SIGMA)과 수익률 계산"),
        ("step4_finalize_ratios.py", "4단계: 최종 재무비율 정리 및 저장")
    ]
    
    success_count = 0
    
    for step_file, step_name in steps:
        success = run_step(step_file, step_name)
        if success:
            success_count += 1
        else:
            print(f"\n❌ {step_name} 실패로 인해 프로세스를 중단합니다.")
            break
        
        # 단계 간 잠시 대기
        time.sleep(1)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("🏁 재무비율 계산 프로세스 완료")
    print(f"{'='*60}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요시간: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
    print(f"성공한 단계: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("✅ 모든 단계가 성공적으로 완료되었습니다!")
        print("📁 최종 결과: data_new/processed/FS_ratio_flow.csv")
        print("\n💡 주요 개선사항:")
        print("- Stock 지표는 평균값 사용으로 더 정확한 비율 계산")
        print("- Flow 지표와의 매칭 개선")
        print("- 시계열적 일관성 향상")
        print("- 재무비율의 경제적 의미 정확성 향상")
        
        # 파일 크기 확인
        try:
            file_size = os.path.getsize('data_new/processed/FS_ratio_flow.csv') / 1024 / 1024
            print(f"📊 파일 크기: {file_size:.2f} MB")
        except:
            pass
            
    else:
        print("⚠️ 일부 단계가 실패했습니다.")
    
    return success_count == len(steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 