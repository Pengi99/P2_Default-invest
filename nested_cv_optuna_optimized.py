import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')


def create_model_config():
    """모델별 설정 및 하이퍼파라미터 탐색 공간 정의 (최적화됨)"""
    model_configs = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'param_space': {
                'model__penalty': ['l2'],  # l1 제거로 속도 향상
                'model__solver': ['lbfgs', 'saga'],  # 안정적인 solver만
                'model__C': (0.01, 10.0),  # 범위 축소
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            'param_space': {
                'model__n_estimators': (50, 200),  # 범위 축소
                'model__max_depth': (3, 15),  # 범위 축소
                'model__min_samples_split': (5, 15),  # 범위 축소
                'model__min_samples_leaf': (2, 8),  # 범위 축소
                'model__max_features': ['sqrt', 'log2']  # None 제거
            }
        },
        'XGBoost': {
            'model': None,
            'param_space': {
                'model__n_estimators': (50, 200),  # 범위 축소
                'model__max_depth': (3, 8),  # 범위 축소
                'model__learning_rate': (0.05, 0.3),  # 최소값 상향
                'model__subsample': (0.7, 1.0),  # 범위 축소
                'model__colsample_bytree': (0.7, 1.0),  # 범위 축소
                'model__reg_alpha': (0.0, 0.5),  # 범위 축소
                'model__reg_lambda': (0.5, 2.0)  # 범위 축소
            }
        }
    }
    return model_configs


def create_pipeline(model, sampler_type='BorderlineSMOTE'):
    """전처리와 샘플링을 포함한 파이프라인 생성"""
    if sampler_type == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE(random_state=42, n_jobs=1)  # n_jobs=1로 메모리 절약
    else:
        sampler = SMOTE(random_state=42, n_jobs=1)
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('sampler', sampler),
        ('model', model)
    ])
    return pipeline


def suggest_params(trial, param_space, model_name):
    """Optuna trial을 이용한 하이퍼파라미터 제안 (최적화됨)"""
    params = {}
    
    for param_name, param_range in param_space.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            if isinstance(param_range[0], float):
                # log 스케일 사용으로 탐색 효율성 향상
                if 'learning_rate' in param_name or 'C' in param_name:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
        elif isinstance(param_range, list):
            params[param_name] = trial.suggest_categorical(param_name, param_range)
    
    return params


def find_best_threshold(y_true, y_prob, metric='f1'):
    """최적 임계값 탐색 (최적화됨)"""
    # 더 적은 threshold로 속도 향상
    thresholds = np.linspace(0.1, 0.9, 41)  # 81 -> 41
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        else:
            score = roc_auc_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def evaluate_fold(trial_params, X_train, y_train, model_config, model_name, inner_cv):
    """단일 fold 평가 함수 (병렬 처리용)"""
    trial, params = trial_params
    
    # 모델 생성
    if model_name == 'XGBoost':
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,  # 개별 모델은 단일 스레드
            eval_metric='logloss'
        )
    else:
        model = model_config['model']
    
    pipeline = create_pipeline(model)
    pipeline.set_params(**params)
    
    # Inner CV로 성능 평가
    scores = []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
        X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
        y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
        
        pipeline.fit(X_inner_train, y_inner_train)
        y_pred_proba = pipeline.predict_proba(X_inner_val)[:, 1]
        
        _, f1_score_val = find_best_threshold(y_inner_val, y_pred_proba, metric='f1')
        scores.append(f1_score_val)
    
    return np.mean(scores)


def objective_function(trial, X_train, y_train, model_config, model_name, inner_cv):
    """Optuna 목적 함수 (최적화됨)"""
    params = suggest_params(trial, model_config['param_space'], model_name)
    
    # 조기 종료 조건 추가
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return evaluate_fold((trial, params), X_train, y_train, model_config, model_name, inner_cv)


def repeated_holdout_nested_cv(X, y, n_trials=15, n_outer_iterations=3, n_inner_iterations=3, 
                               outer_test_size=0.2, inner_test_size=0.25, sampler_type='BorderlineSMOTE'):
    """
    반복 홀드아웃 기반 Nested CV with Optuna 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : pandas.DataFrame
        특징 행렬
    y : pandas.Series  
        레이블
    n_trials : int
        Optuna 탐색 횟수
    n_outer_iterations : int
        외부 홀드아웃 반복 횟수
    n_inner_iterations : int  
        내부 홀드아웃 반복 횟수
    outer_test_size : float
        외부 테스트 세트 비율
    inner_test_size : float
        내부 검증 세트 비율
    sampler_type : str
        샘플링 방법
    
    Returns:
    --------
    dict : 모델별 결과
    """
    
    from sklearn.model_selection import train_test_split
    
    model_configs = create_model_config()
    results = {}
    
    for model_name, model_config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model_results = {
            'iteration_results': [],
            'mean_scores': {}
        }
        
        # 외부 홀드아웃 반복
        for outer_iter in range(n_outer_iterations):
            print(f"\nOuter Iteration {outer_iter + 1}/{n_outer_iterations}")
            
            # 외부 train/test 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=outer_test_size, stratify=y, random_state=42 + outer_iter
            )
            
            def inner_objective(trial):
                params = suggest_params(trial, model_config['param_space'], model_name)
                
                # 내부 홀드아웃 반복
                inner_scores = []
                for inner_iter in range(n_inner_iterations):
                    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                        X_train, y_train, test_size=inner_test_size, stratify=y_train, 
                        random_state=42 + outer_iter * 10 + inner_iter
                    )
                    
                    # 모델 생성 및 훈련
                    if model_name == 'XGBoost':
                        neg_count = (y_inner_train == 0).sum()
                        pos_count = (y_inner_train == 1).sum()
                        scale_pos_weight = neg_count / pos_count
                        model = xgb.XGBClassifier(
                            scale_pos_weight=scale_pos_weight,
                            random_state=42,
                            n_jobs=1,
                            eval_metric='logloss'
                        )
                    else:
                        model = model_config['model']
                    
                    pipeline = create_pipeline(model, sampler_type)
                    pipeline.set_params(**params)
                    pipeline.fit(X_inner_train, y_inner_train)
                    
                    # 검증 세트에서 F1 계산
                    y_pred_proba = pipeline.predict_proba(X_inner_val)[:, 1]
                    _, f1_score_val = find_best_threshold(y_inner_val, y_pred_proba, metric='f1')
                    inner_scores.append(f1_score_val)
                
                return np.mean(inner_scores)
            
            # Optuna 최적화
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42 + outer_iter),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
            )
            
            study.optimize(inner_objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            best_inner_score = study.best_value
            
            print(f"  Best inner score: {best_inner_score:.4f}")
            
            # 최적 파라미터로 모델 재훈련
            if model_name == 'XGBoost':
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                scale_pos_weight = neg_count / pos_count
                best_model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            else:
                best_model = model_config['model']
            
            best_pipeline = create_pipeline(best_model, sampler_type)
            best_pipeline.set_params(**best_params)
            
            # 임계값 튜닝을 위한 train/val 분리
            X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42 + outer_iter
            )
            
            best_pipeline.fit(X_train_temp, y_train_temp)
            y_val_proba = best_pipeline.predict_proba(X_val_temp)[:, 1]
            best_threshold, _ = find_best_threshold(y_val_temp, y_val_proba, metric='f1')
            
            # 전체 훈련 데이터로 최종 모델 훈련
            best_pipeline.fit(X_train, y_train)
            
            # 테스트 세트 예측
            y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= best_threshold).astype(int)
            
            # 성능 계산
            test_f1 = f1_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            
            iteration_result = {
                'iteration': outer_iter + 1,
                'best_params': best_params,
                'best_inner_score': best_inner_score,
                'best_threshold': best_threshold,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc
            }
            
            model_results['iteration_results'].append(iteration_result)
            
            print(f"  Best threshold: {best_threshold:.3f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
        
        # 평균 성능 계산
        mean_inner_score = np.mean([r['best_inner_score'] for r in model_results['iteration_results']])
        mean_test_f1 = np.mean([r['test_f1'] for r in model_results['iteration_results']])
        mean_test_roc_auc = np.mean([r['test_roc_auc'] for r in model_results['iteration_results']])
        std_test_f1 = np.std([r['test_f1'] for r in model_results['iteration_results']])
        std_test_roc_auc = np.std([r['test_roc_auc'] for r in model_results['iteration_results']])
        
        model_results['mean_scores'] = {
            'mean_inner_score': mean_inner_score,
            'mean_test_f1': mean_test_f1,
            'mean_test_roc_auc': mean_test_roc_auc,
            'std_test_f1': std_test_f1,
            'std_test_roc_auc': std_test_roc_auc
        }
        
        results[model_name] = model_results
        
        print(f"\n{model_name} 평균 성능:")
        print(f"  Mean Inner Score: {mean_inner_score:.4f}")
        print(f"  Mean Test F1: {mean_test_f1:.4f} (±{std_test_f1:.4f})")
        print(f"  Mean Test ROC-AUC: {mean_test_roc_auc:.4f} (±{std_test_roc_auc:.4f})")
    
    return results


def print_detailed_results(results):
    """상세 결과 출력"""
    print("\n" + "="*80)
    print("DETAILED RESULTS (REPEATED HOLDOUT)")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        print("-" * 60)
        
        for iter_result in model_results['iteration_results']:
            print(f"Iteration {iter_result['iteration']}:")
            print(f"  Best Params: {iter_result['best_params']}")
            print(f"  Inner Score: {iter_result['best_inner_score']:.4f}")
            print(f"  Best Threshold: {iter_result['best_threshold']:.3f}")
            print(f"  Test F1: {iter_result['test_f1']:.4f}")
            print(f"  Test ROC-AUC: {iter_result['test_roc_auc']:.4f}")
            print()
        
        mean_scores = model_results['mean_scores']
        print(f"Average Performance:")
        print(f"  Mean Inner Score: {mean_scores['mean_inner_score']:.4f}")
        print(f"  Mean Test F1: {mean_scores['mean_test_f1']:.4f} (±{mean_scores['std_test_f1']:.4f})")
        print(f"  Mean Test ROC-AUC: {mean_scores['mean_test_roc_auc']:.4f} (±{mean_scores['std_test_roc_auc']:.4f})")
        print("\n" + "="*60)


# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성
    from sklearn.datasets import make_classification
    X_temp, y_temp = make_classification(
        n_samples=1000, n_features=20, n_redundant=5, 
        n_informative=15, random_state=42, weights=[0.8, 0.2]
    )
    X = pd.DataFrame(X_temp, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y_temp, name='target')
    
    print("Starting Optimized Nested CV with Optuna...")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # 반복 홀드아웃 기반 Nested CV 실행
    results = repeated_holdout_nested_cv(
        X, y, 
        n_trials=10,  # 빠른 실행을 위해 감소
        n_outer_iterations=3, 
        n_inner_iterations=3,
        sampler_type='BorderlineSMOTE'
    )
    
    # 상세 결과 출력
    print_detailed_results(results)