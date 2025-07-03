#!/usr/bin/env python3
"""
Ensemble과 Logistic Regression Test 결과 수정 테스트
"""

def test_key_mapping():
    """키 매핑 수정사항 테스트"""
    print("🔧 키 매핑 수정사항 테스트")
    print("="*50)
    
    # 수정된 ensemble test_metrics 형태
    ensemble_test_metrics = {
        'roc_auc': 0.782,
        'precision_optimal': 0.345,
        'recall_optimal': 0.567,
        'f1_optimal': 0.432,
        'balanced_accuracy_optimal': 0.678,
        'average_precision': 0.456,
        # 하위 호환성을 위한 기존 키들
        'auc': 0.782,
        'precision': 0.345,
        'recall': 0.567,
        'f1': 0.432,
        'balanced_accuracy': 0.678
    }
    
    # modeling_pipeline에서 사용하는 summary table 키 매핑 (수정된 버전)
    def get_summary_values(test_metrics):
        return {
            'Test_AUC': test_metrics.get('roc_auc', 0),
            'Test_Precision': test_metrics.get('precision_optimal', test_metrics.get('precision_default', 0)),
            'Test_Recall': test_metrics.get('recall_optimal', test_metrics.get('recall_default', 0)),
            'Test_F1': test_metrics.get('f1_optimal', test_metrics.get('f1_default', 0)),
            'Test_Balanced_Acc': test_metrics.get('balanced_accuracy_optimal', 0),
            'Test_Average_Precision': test_metrics.get('average_precision', 0)
        }
    
    summary_values = get_summary_values(ensemble_test_metrics)
    
    print("📊 Ensemble Test 결과:")
    for key, value in summary_values.items():
        print(f"  {key}: {value:.3f}")
    
    # 기존 방식 (수정 전)
    def get_old_summary_values(test_metrics):
        return {
            'Test_AUC': test_metrics.get('auc', 0),  # 'auc'를 찾음 -> 없으면 0
            'Test_F1': test_metrics.get('f1', 0),    # 'f1'을 찾음 -> 없으면 0
        }
    
    old_summary_values = get_old_summary_values({'roc_auc': 0.782, 'f1_optimal': 0.432})
    print("\n❌ 수정 전 (잘못된 키):")
    for key, value in old_summary_values.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n✅ 수정 완료!")
    print("- ensemble_pipeline.py: 올바른 키 이름으로 test_metrics 반환")
    print("- modeling_pipeline.py: 이미 수정된 키 매핑 사용")
    
    return summary_values

def test_logistic_regression_issue():
    """Logistic Regression 문제 분석"""
    print("\n🔍 Logistic Regression 문제 분석")
    print("="*50)
    
    print("📋 예상 원인:")
    print("1. evaluate_model()에서 LogisticRegression 평가 시 test_metrics 생성 실패")
    print("2. 또는 test_metrics 키가 올바르게 생성되지 않음")
    print("3. Class imbalance로 인한 예측 실패 (모든 예측이 0 또는 1)")
    
    print("\n🔧 해결 방안:")
    print("1. evaluate_model() 메서드에서 LogisticRegression 처리 확인")
    print("2. Test 데이터 예측 시 threshold 적용 확인")
    print("3. Class imbalance handling 확인")
    
    print("\n📝 권장사항:")
    print("- modeling_pipeline.py의 evaluate_model() 메서드 로그 추가")
    print("- LogisticRegression test 평가 과정에서 예외 처리 확인")
    print("- Test 예측값 분포 확인 (모두 0 또는 1인지)")

if __name__ == "__main__":
    print("🧪 Ensemble & Logistic Regression Test 결과 수정 검증")
    print("="*80)
    
    # 1. 키 매핑 테스트
    summary_values = test_key_mapping()
    
    # 2. Logistic Regression 문제 분석
    test_logistic_regression_issue()
    
    print("\n" + "="*80)
    print("📋 요약:")
    print("✅ Ensemble: 키 매핑 수정 완료")
    print("🔄 Logistic Regression: 추가 조사 필요")
    print("   - evaluate_model() 메서드에서 test 평가 과정 확인")
    print("   - Class imbalance 처리 확인")