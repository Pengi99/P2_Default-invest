import config
from src import data_collector, preprocess, model_trainer, analysis

def main():
    """프로젝트 전체 파이프라인 실행"""
    print("1. 데이터 수집 시작...")
    # data_collector.fetch_and_save_all_data(config.START_YEAR, config.END_YEAR)

    print("2. 데이터 전처리 시작...")
    # preprocess.run_preprocessing()

    print("3. 부도 예측 모델 훈련 시작...")
    # best_model = model_trainer.run_training_pipeline()

    print("4. 백테스팅 및 성과 분석 시작...")
    # analysis.run_backtest_and_analyze(best_model)

    print("프로젝트 실행 완료!")

if __name__ == "__main__":
    main()
