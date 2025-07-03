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
warnings.filterwarnings('ignore')


def create_model_config():
    """모델별 설정 및 하이퍼파라미터 탐색 공간 정의"""
    model_configs = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'param_space': {
                'model__penalty': ['l1', 'l2', 'elasticnet'],
                'model__solver': ['liblinear', 'saga'],
                'model__C': (0.001, 100.0),
                'model__l1_ratio': (0.0, 1.0)
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            'param_space': {
                'model__n_estimators': (50, 300),
                'model__max_depth': (3, 20),
                'model__min_samples_split': (2, 20),
                'model__min_samples_leaf': (1, 10),
                'model__max_features': ['sqrt', 'log2', None]
            }
        },
        'XGBoost': {
            'model': None,  # scale_pos_weight는 실행시 동적으로 설정
            'param_space': {
                'model__n_estimators': (50, 300),
                'model__max_depth': (3, 10),
                'model__learning_rate': (0.01, 0.3),
                'model__subsample': (0.6, 1.0),
                'model__colsample_bytree': (0.6, 1.0),
                'model__reg_alpha': (0.0, 1.0),
                'model__reg_lambda': (0.0, 1.0)
            }
        }
    }
    return model_configs


def create_pipeline(model, sampler_type='BorderlineSMOTE'):
    """전처리와 샘플링을 포함한 파이프라인 생성"""
    if sampler_type == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE(random_state=42, n_jobs=-1)
    else:
        sampler = SMOTE(random_state=42, n_jobs=-1)
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('sampler', sampler),
        ('model', model)
    ])
    return pipeline


def suggest_params(trial, param_space, model_name):
    """Optuna trial을 이용한 하이퍼파라미터 제안"""
    params = {}
    
    for param_name, param_range in param_space.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            if isinstance(param_range[0], float):
                params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
        elif isinstance(param_range, list):
            params[param_name] = trial.suggest_categorical(param_name, param_range)
    
    # LogisticRegression의 특별한 경우 처리
    if model_name == 'LogisticRegression':
        if params.get('model__penalty') != 'elasticnet':
            params.pop('model__l1_ratio', None)
        if params.get('model__penalty') == 'l1' and params.get('model__solver') not in ['liblinear', 'saga']:
            params['model__solver'] = 'liblinear'
    
    return params


def find_best_threshold(y_true, y_prob, metric='f1'):
    """최적 임계값 탐색 (F1 점수 기준)"""
    thresholds = np.linspace(0.1, 0.9, 81)
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


def objective_function(trial, X_train, y_train, model_config, model_name, inner_cv):
    """Optuna 목적 함수"""
    # 모델 생성
    if model_name == 'XGBoost':
        # scale_pos_weight 계산
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    else:
        model = model_config['model']
    
    # 파이프라인 생성
    pipeline = create_pipeline(model)
    
    # 하이퍼파라미터 제안
    params = suggest_params(trial, model_config['param_space'], model_name)
    pipeline.set_params(**params)
    
    # Inner CV로 성능 평가
    scores = []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
        X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
        y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
        
        pipeline.fit(X_inner_train, y_inner_train)
        y_pred_proba = pipeline.predict_proba(X_inner_val)[:, 1]
        
        # F1 점수로 평가
        _, f1_score_val = find_best_threshold(y_inner_val, y_pred_proba, metric='f1')
        scores.append(f1_score_val)
    
    return np.mean(scores)


def nested_cv_with_optuna(X, y, n_trials=50, sampler_type='BorderlineSMOTE'):
    """
    Nested CV with Optuna 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : pandas.DataFrame
        특징 행렬
    y : pandas.Series  
        레이블
    n_trials : int
        Optuna 탐색 횟수
    sampler_type : str
        샘플링 방법 ('BorderlineSMOTE' 또는 'SMOTE')
    
    Returns:
    --------
    dict : 모델별 결과
    """
    
    # CV 설정
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 모델 설정
    model_configs = create_model_config()
    
    # 결과 저장
    results = {}
    
    for model_name, model_config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model_results = {
            'fold_results': [],
            'mean_scores': {}
        }
        
        # Outer CV 루프
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\nOuter Fold {fold_idx + 1}/5")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Optuna 스터디 생성
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # 하이퍼파라미터 최적화
            study.optimize(
                lambda trial: objective_function(
                    trial, X_train, y_train, model_config, model_name, inner_cv
                ),
                n_trials=n_trials,
                show_progress_bar=False
            )
            
            best_params = study.best_params
            best_inner_score = study.best_value
            
            print(f"  Best inner CV score: {best_inner_score:.4f}")
            
            # 최적 하이퍼파라미터로 모델 재훈련
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
            
            # Train/Validation 분리 (임계값 튜닝용)
            temp_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_idx_temp, val_idx_temp = next(temp_cv.split(X_train, y_train))
            X_train_temp, X_val_temp = X_train.iloc[train_idx_temp], X_train.iloc[val_idx_temp]
            y_train_temp, y_val_temp = y_train.iloc[train_idx_temp], y_train.iloc[val_idx_temp]
            
            # 모델 훈련 (임계값 튜닝용)
            best_pipeline.fit(X_train_temp, y_train_temp)
            y_val_proba = best_pipeline.predict_proba(X_val_temp)[:, 1]
            
            # 최적 임계값 찾기
            best_threshold, _ = find_best_threshold(y_val_temp, y_val_proba, metric='f1')
            
            # 전체 훈련 데이터로 최종 모델 훈련
            best_pipeline.fit(X_train, y_train)
            
            # 테스트 세트 예측
            y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= best_threshold).astype(int)
            
            # 성능 계산
            test_f1 = f1_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            
            fold_result = {
                'fold': fold_idx + 1,
                'best_params': best_params,
                'best_inner_score': best_inner_score,
                'best_threshold': best_threshold,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc
            }
            
            model_results['fold_results'].append(fold_result)
            
            print(f"  Best threshold: {best_threshold:.3f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
        
        # 평균 성능 계산
        mean_inner_score = np.mean([r['best_inner_score'] for r in model_results['fold_results']])
        mean_test_f1 = np.mean([r['test_f1'] for r in model_results['fold_results']])
        mean_test_roc_auc = np.mean([r['test_roc_auc'] for r in model_results['fold_results']])
        std_test_f1 = np.std([r['test_f1'] for r in model_results['fold_results']])
        std_test_roc_auc = np.std([r['test_roc_auc'] for r in model_results['fold_results']])
        
        model_results['mean_scores'] = {
            'mean_inner_score': mean_inner_score,
            'mean_test_f1': mean_test_f1,
            'mean_test_roc_auc': mean_test_roc_auc,
            'std_test_f1': std_test_f1,
            'std_test_roc_auc': std_test_roc_auc
        }
        
        results[model_name] = model_results
        
        print(f"\n{model_name} 평균 성능:")
        print(f"  Mean Inner CV Score: {mean_inner_score:.4f}")
        print(f"  Mean Test F1: {mean_test_f1:.4f} (±{std_test_f1:.4f})")
        print(f"  Mean Test ROC-AUC: {mean_test_roc_auc:.4f} (±{std_test_roc_auc:.4f})")
    
    return results


def print_detailed_results(results):
    """상세 결과 출력"""
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        print("-" * 60)
        
        # Fold별 결과
        for fold_result in model_results['fold_results']:
            print(f"Fold {fold_result['fold']}:")
            print(f"  Best Params: {fold_result['best_params']}")
            print(f"  Inner CV Score: {fold_result['best_inner_score']:.4f}")
            print(f"  Best Threshold: {fold_result['best_threshold']:.3f}")
            print(f"  Test F1: {fold_result['test_f1']:.4f}")
            print(f"  Test ROC-AUC: {fold_result['test_roc_auc']:.4f}")
            print()
        
        # 평균 성능
        mean_scores = model_results['mean_scores']
        print(f"Average Performance:")
        print(f"  Mean Inner CV Score: {mean_scores['mean_inner_score']:.4f}")
        print(f"  Mean Test F1: {mean_scores['mean_test_f1']:.4f} (±{mean_scores['std_test_f1']:.4f})")
        print(f"  Mean Test ROC-AUC: {mean_scores['mean_test_roc_auc']:.4f} (±{mean_scores['std_test_roc_auc']:.4f})")
        print("\n" + "="*60)


# 사용 예시
if __name__ == "__main__":
    # 데이터 로드 예시 (실제 데이터로 교체 필요)
    # X = pd.read_csv('your_features.csv')
    # y = pd.read_csv('your_labels.csv').squeeze()
    
    # 예시 데이터 생성 (실제 사용시 삭제)
    from sklearn.datasets import make_classification
    X_temp, y_temp = make_classification(
        n_samples=1000, n_features=20, n_redundant=5, 
        n_informative=15, random_state=42, weights=[0.8, 0.2]
    )
    X = pd.DataFrame(X_temp, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y_temp, name='target')
    
    print("Starting Nested CV with Optuna...")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Nested CV 실행
    results = nested_cv_with_optuna(X, y, n_trials=30, sampler_type='BorderlineSMOTE')
    
    # 상세 결과 출력
    print_detailed_results(results)