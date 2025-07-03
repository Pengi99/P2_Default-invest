#!/usr/bin/env python3
"""
빠른 모델링 테스트 스크립트
========================

기존 modeling_pipeline 대신 사용할 수 있는 간단한 테스트용 스크립트
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import logging

# 모델링 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, balanced_accuracy_score
)
import xgboost as xgb

warnings.filterwarnings('ignore')

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data(data_path="data/final"):
    """데이터 로드"""
    logger = logging.getLogger(__name__)
    data_path = Path(data_path)
    
    try:
        X_train = pd.read_csv(data_path / "X_train.csv", index_col=0)
        X_val = pd.read_csv(data_path / "X_val.csv", index_col=0)
        X_test = pd.read_csv(data_path / "X_test.csv", index_col=0)
        y_train = pd.read_csv(data_path / "y_train.csv", index_col=0).squeeze()
        y_val = pd.read_csv(data_path / "y_val.csv", index_col=0).squeeze()
        y_test = pd.read_csv(data_path / "y_test.csv", index_col=0).squeeze()
        
        logger.info(f"데이터 로드 완료:")
        logger.info(f"  Train: {X_train.shape}, 부실률: {y_train.mean():.3f}")
        logger.info(f"  Val: {X_val.shape}, 부실률: {y_val.mean():.3f}")
        logger.info(f"  Test: {X_test.shape}, 부실률: {y_test.mean():.3f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise

def find_best_threshold(y_true, y_prob):
    """최적 임계값 찾기"""
    thresholds = np.linspace(0.1, 0.9, 41)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def repeated_holdout_validation(model, X_train, y_train, n_iterations=10, test_size=0.2):
    """반복 홀드아웃 검증"""
    scores = []
    
    for i in range(n_iterations):
        # 매번 다른 시드로 train/validation 분리
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, 
            test_size=test_size, 
            stratify=y_train, 
            random_state=42 + i
        )
        
        # 모델 복사 및 훈련
        from sklearn.base import clone
        model_copy = clone(model)
        model_copy.fit(X_train_split, y_train_split)
        
        # 검증 세트에서 예측 및 F1 점수 계산
        y_val_prob = model_copy.predict_proba(X_val_split)[:, 1]
        best_threshold, f1_score_val = find_best_threshold(y_val_split, y_val_prob)
        scores.append(f1_score_val)
    
    return np.array(scores)

def evaluate_model(model, X_val, y_val, X_test, y_test, model_name):
    """모델 평가"""
    logger = logging.getLogger(__name__)
    
    # 최적 임계값 찾기
    y_val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, val_f1 = find_best_threshold(y_val, y_val_prob)
    
    # 테스트 세트 예측
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)
    
    # 성능 계산
    metrics = {
        'model': model_name,
        'best_threshold': best_threshold,
        'val_f1': val_f1,
        'test_roc_auc': roc_auc_score(y_test, y_test_prob),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_balanced_acc': balanced_accuracy_score(y_test, y_test_pred)
    }
    
    logger.info(f"{model_name} 성능:")
    logger.info(f"  Best Threshold: {best_threshold:.3f}")
    logger.info(f"  Val F1: {val_f1:.4f}")
    logger.info(f"  Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    logger.info(f"  Test F1: {metrics['test_f1']:.4f}")
    logger.info(f"  Test Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Test Recall: {metrics['test_recall']:.4f}")
    
    return metrics

def train_logistic_regression(X_train, y_train):
    """로지스틱 회귀 훈련"""
    logger = logging.getLogger(__name__)
    logger.info("로지스틱 회귀 훈련 시작")
    
    # 간단한 파라미터로 빠른 훈련
    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        random_state=42,
        max_iter=1000
    )
    
    # 반복 홀드아웃 검증
    validation_scores = repeated_holdout_validation(model, X_train, y_train, n_iterations=5)
    
    # 전체 데이터로 훈련
    model.fit(X_train, y_train)
    
    logger.info(f"로지스틱 회귀 완료 - 홀드아웃 F1: {validation_scores.mean():.4f} (±{validation_scores.std():.4f})")
    return model

def train_random_forest(X_train, y_train):
    """랜덤 포레스트 훈련"""
    logger = logging.getLogger(__name__)
    logger.info("랜덤 포레스트 훈련 시작")
    
    # 간단한 파라미터로 빠른 훈련
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 반복 홀드아웃 검증
    validation_scores = repeated_holdout_validation(model, X_train, y_train, n_iterations=5)
    
    # 전체 데이터로 훈련
    model.fit(X_train, y_train)
    
    logger.info(f"랜덤 포레스트 완료 - 홀드아웃 F1: {validation_scores.mean():.4f} (±{validation_scores.std():.4f})")
    return model

def train_xgboost(X_train, y_train):
    """XGBoost 훈련"""
    logger = logging.getLogger(__name__)
    logger.info("XGBoost 훈련 시작")
    
    # scale_pos_weight 계산
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # 간단한 파라미터로 빠른 훈련
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # 반복 홀드아웃 검증
    validation_scores = repeated_holdout_validation(model, X_train, y_train, n_iterations=5)
    
    # 전체 데이터로 훈련
    model.fit(X_train, y_train)
    
    logger.info(f"XGBoost 완료 - 홀드아웃 F1: {validation_scores.mean():.4f} (±{validation_scores.std():.4f})")
    return model

def main():
    """메인 실행 함수"""
    start_time = datetime.now()
    logger = setup_logging()
    logger.info("빠른 모델링 테스트 시작")
    
    try:
        # 1. 데이터 로드
        X_train, X_val, X_test, y_train, y_val, y_test = load_data()
        
        # 2. 모델 훈련 및 평가
        results = []
        
        # 로지스틱 회귀
        try:
            lr_model = train_logistic_regression(X_train, y_train)
            lr_metrics = evaluate_model(lr_model, X_val, y_val, X_test, y_test, "LogisticRegression")
            results.append(lr_metrics)
        except Exception as e:
            logger.error(f"로지스틱 회귀 실패: {e}")
        
        # 랜덤 포레스트
        try:
            rf_model = train_random_forest(X_train, y_train)
            rf_metrics = evaluate_model(rf_model, X_val, y_val, X_test, y_test, "RandomForest")
            results.append(rf_metrics)
        except Exception as e:
            logger.error(f"랜덤 포레스트 실패: {e}")
        
        # XGBoost
        try:
            xgb_model = train_xgboost(X_train, y_train)
            xgb_metrics = evaluate_model(xgb_model, X_val, y_val, X_test, y_test, "XGBoost")
            results.append(xgb_metrics)
        except Exception as e:
            logger.error(f"XGBoost 실패: {e}")
        
        # 3. 결과 요약
        if results:
            logger.info("\n" + "="*60)
            logger.info("최종 결과 요약")
            logger.info("="*60)
            
            df_results = pd.DataFrame(results)
            df_results = df_results.round(4)
            
            # 결과 출력
            for idx, row in df_results.iterrows():
                logger.info(f"{row['model']:<20} F1: {row['test_f1']:.4f}  ROC-AUC: {row['test_roc_auc']:.4f}")
            
            # 최고 성능 모델
            best_f1_idx = df_results['test_f1'].idxmax()
            best_model = df_results.loc[best_f1_idx, 'model']
            best_f1 = df_results.loc[best_f1_idx, 'test_f1']
            
            logger.info(f"\n최고 F1 점수: {best_model} ({best_f1:.4f})")
            
            # 결과 저장
            output_dir = Path("outputs/quick_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            logger.info(f"결과 저장: {output_dir}")
        
        # 실행 시간
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n실행 완료 - 소요 시간: {duration:.1f}초")
        
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    main()