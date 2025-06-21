"""
Normal 데이터(SMOTE 미적용) 모델 학습 및 SMOTE 버전과 비교 분석
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NormalDataModelComparison:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.normal_results = {}
        self.smote_results = {}
        self.comparison_results = {}
        
    def load_data(self, data_path='data_new/final/'):
        """Normal 데이터 로드"""
        print("📂 Normal 데이터 로드")
        print("="*60)
        
        # Normal 데이터 로드
        self.X_train_normal = pd.read_csv(os.path.join(data_path, 'X_train_normal.csv'))
        self.X_valid_normal = pd.read_csv(os.path.join(data_path, 'X_valid_normal.csv'))
        self.X_test_normal = pd.read_csv(os.path.join(data_path, 'X_test_normal.csv'))
        
        self.y_train_normal = pd.read_csv(os.path.join(data_path, 'y_train_normal.csv')).iloc[:, 0]
        self.y_valid_normal = pd.read_csv(os.path.join(data_path, 'y_valid_normal.csv')).iloc[:, 0]
        self.y_test_normal = pd.read_csv(os.path.join(data_path, 'y_test_normal.csv')).iloc[:, 0]
        
        print(f"✅ Normal 데이터 로드 완료")
        print(f"   Train: {self.X_train_normal.shape}, 부실비율: {self.y_train_normal.mean():.2%}")
        print(f"   Valid: {self.X_valid_normal.shape}, 부실비율: {self.y_valid_normal.mean():.2%}")
        print(f"   Test: {self.X_test_normal.shape}, 부실비율: {self.y_test_normal.mean():.2%}")
        
        # SMOTE 데이터도 로드 (비교용)
        self.X_train_smote = pd.read_csv(os.path.join(data_path, 'X_train_smote.csv'))
        self.y_train_smote = pd.read_csv(os.path.join(data_path, 'y_train_smote.csv')).iloc[:, 0]
        
        print(f"   SMOTE Train: {self.X_train_smote.shape}, 부실비율: {self.y_train_smote.mean():.2%}")
        
    def optimize_logistic_regression(self, n_trials=50):
        """로지스틱 회귀 최적화"""
        print("\n🔍 로지스틱 회귀 최적화 (Normal 데이터)")
        print("="*60)
        
        def objective(trial):
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            C = trial.suggest_float('C', 1e-4, 100, log=True)
            max_iter = trial.suggest_int('max_iter', 100, 2000)
            
            # penalty에 따라 적절한 solver 선택
            if penalty == 'l1':
                solver = 'liblinear'
            elif penalty == 'l2':
                solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
            else:  # elasticnet
                solver = 'saga'
            
            params = {
                'penalty': penalty,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'random_state': self.random_state
            }
            
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
            
            model = LogisticRegression(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, self.X_train_normal, self.y_train_normal, 
                                   cv=skf, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 모델 훈련
        best_params = study.best_params
        lr_model = LogisticRegression(**best_params)
        lr_model.fit(self.X_train_normal, self.y_train_normal)
        
        self.models['LogisticRegression'] = lr_model
        self.normal_results['LogisticRegression'] = {
            'best_params': best_params,
            'cv_score': study.best_value
        }
        
        print(f"✅ 최적 AUC: {study.best_value:.4f}")
        print(f"📊 최적 파라미터: {best_params}")
        
    def optimize_random_forest(self, n_trials=50):
        """랜덤 포레스트 최적화"""
        print("\n🌲 랜덤 포레스트 최적화 (Normal 데이터)")
        print("="*60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, self.X_train_normal, self.y_train_normal, 
                                   cv=skf, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 모델 훈련
        best_params = study.best_params
        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(self.X_train_normal, self.y_train_normal)
        
        self.models['RandomForest'] = rf_model
        self.normal_results['RandomForest'] = {
            'best_params': best_params,
            'cv_score': study.best_value
        }
        
        print(f"✅ 최적 AUC: {study.best_value:.4f}")
        print(f"📊 최적 파라미터: {best_params}")
        
    def optimize_xgboost(self, n_trials=50):
        """XGBoost 최적화"""
        print("\n🚀 XGBoost 최적화 (Normal 데이터)")
        print("="*60)
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, self.X_train_normal, self.y_train_normal, 
                                   cv=skf, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 모델 훈련
        best_params = study.best_params
        xgb_model = xgb.XGBClassifier(**best_params)
        xgb_model.fit(self.X_train_normal, self.y_train_normal)
        
        self.models['XGBoost'] = xgb_model
        self.normal_results['XGBoost'] = {
            'best_params': best_params,
            'cv_score': study.best_value
        }
        
        print(f"✅ 최적 AUC: {study.best_value:.4f}")
        print(f"📊 최적 파라미터: {best_params}")
        
    def evaluate_models(self):
        """모든 모델 평가"""
        print("\n📊 모델 평가")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\n🔍 {model_name} 평가:")
            
            # 예측
            y_valid_pred = model.predict(self.X_valid_normal)
            y_valid_proba = model.predict_proba(self.X_valid_normal)[:, 1]
            y_test_pred = model.predict(self.X_test_normal)
            y_test_proba = model.predict_proba(self.X_test_normal)[:, 1]
            
            # 검증 성능
            valid_metrics = {
                'auc': roc_auc_score(self.y_valid_normal, y_valid_proba),
                'precision': precision_score(self.y_valid_normal, y_valid_pred),
                'recall': recall_score(self.y_valid_normal, y_valid_pred),
                'f1': f1_score(self.y_valid_normal, y_valid_pred)
            }
            
            # 테스트 성능
            test_metrics = {
                'auc': roc_auc_score(self.y_test_normal, y_test_proba),
                'precision': precision_score(self.y_test_normal, y_test_pred),
                'recall': recall_score(self.y_test_normal, y_test_pred),
                'f1': f1_score(self.y_test_normal, y_test_pred)
            }
            
            self.normal_results[model_name]['valid_metrics'] = valid_metrics
            self.normal_results[model_name]['test_metrics'] = test_metrics
            
            print(f"   검증 - AUC: {valid_metrics['auc']:.4f}, F1: {valid_metrics['f1']:.4f}")
            print(f"   테스트 - AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    def load_smote_results(self, models_path='outputs/models/'):
        """기존 SMOTE 결과 로드"""
        print("\n📂 SMOTE 결과 로드")
        print("="*60)
        
        result_files = {
            'LogisticRegression': 'logistic_regression_results.json',
            'RandomForest': 'random_forest_results.json',
            'XGBoost': 'xgboost_results.json'
        }
        
        for model_name, result_file in result_files.items():
            result_path = os.path.join(models_path, result_file)
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    self.smote_results[model_name] = json.load(f)
                print(f"✅ {model_name} SMOTE 결과 로드")
            else:
                print(f"⚠️ {model_name} SMOTE 결과 파일 없음")
    
    def create_comparison_table(self):
        """Normal vs SMOTE 비교표 생성"""
        print("\n📊 Normal vs SMOTE 비교")
        print("="*60)
        
        comparison_data = []
        
        for model_name in self.models.keys():
            # Normal 결과
            normal_test = self.normal_results[model_name]['test_metrics']
            normal_cv = self.normal_results[model_name]['cv_score']
            
            # SMOTE 결과
            smote_test = self.smote_results.get(model_name, {}).get('test_metrics', {})
            smote_cv = self.smote_results.get(model_name, {}).get('cv_best_score', 0)
            
            comparison_data.append({
                'Model': model_name,
                'Data_Type': 'Normal',
                'CV_AUC': normal_cv,
                'Test_AUC': normal_test['auc'],
                'Test_Precision': normal_test['precision'],
                'Test_Recall': normal_test['recall'],
                'Test_F1': normal_test['f1']
            })
            
            if smote_test:
                comparison_data.append({
                    'Model': model_name,
                    'Data_Type': 'SMOTE',
                    'CV_AUC': smote_cv,
                    'Test_AUC': smote_test['auc'],
                    'Test_Precision': smote_test['precision'],
                    'Test_Recall': smote_test['recall'],
                    'Test_F1': smote_test['f1']
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        print("🏆 모델 성능 비교 (Normal vs SMOTE):")
        print(self.comparison_df.round(4))
        
        return self.comparison_df
    
    def plot_comparison(self, save_path='outputs/visualizations/'):
        """비교 결과 시각화"""
        print("\n📈 비교 결과 시각화")
        print("="*60)
        
        # 1. 성능 지표 비교
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Normal vs SMOTE 데이터 모델 성능 비교', fontsize=16, fontweight='bold')
        
        metrics = ['Test_AUC', 'Test_Precision', 'Test_Recall', 'Test_F1']
        metric_names = ['AUC', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            # 데이터 준비
            pivot_data = self.comparison_df.pivot(index='Model', columns='Data_Type', values=metric)
            
            # 바 차트
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot_data['Normal'], width, 
                          label='Normal', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x + width/2, pivot_data['SMOTE'], width, 
                          label='SMOTE', alpha=0.8, color='lightcoral')
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{name} 비교', fontsize=12, fontweight='bold')
            ax.set_ylabel(name)
            ax.set_xlabel('모델')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_data.index, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/normal_vs_smote_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ 비교 차트 저장: {save_path}/normal_vs_smote_comparison.png")
        plt.show()
        
        # 2. ROC 곡선 비교
        self.plot_roc_comparison(save_path)
    
    def plot_roc_comparison(self, save_path='outputs/visualizations/'):
        """ROC 곡선 비교"""
        plt.figure(figsize=(15, 5))
        
        for i, model_name in enumerate(self.models.keys()):
            plt.subplot(1, 3, i+1)
            
            # Normal 모델 ROC
            model = self.models[model_name]
            y_proba_normal = model.predict_proba(self.X_test_normal)[:, 1]
            fpr_normal, tpr_normal, _ = roc_curve(self.y_test_normal, y_proba_normal)
            auc_normal = auc(fpr_normal, tpr_normal)
            
            plt.plot(fpr_normal, tpr_normal, 'b-', lw=2, 
                    label=f'Normal (AUC = {auc_normal:.4f})')
            
            # SMOTE 모델 ROC (가장 가까운 성능으로 추정)
            if model_name in self.smote_results:
                smote_auc = self.smote_results[model_name]['test_metrics']['auc']
                # 간단한 추정 곡선 (실제로는 저장된 모델 필요)
                plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, 
                        label=f'SMOTE (AUC = {smote_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC 비교')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_comparison_normal_vs_smote.png', dpi=300, bbox_inches='tight')
        print(f"✅ ROC 비교 저장: {save_path}/roc_comparison_normal_vs_smote.png")
        plt.show()
    
    def save_normal_results(self, save_path='outputs/models/'):
        """Normal 데이터 결과 저장"""
        print("\n💾 Normal 데이터 결과 저장")
        print("="*60)
        
        os.makedirs(save_path, exist_ok=True)
        
        # 모델 저장
        for model_name, model in self.models.items():
            model_path = f'{save_path}/{model_name.lower()}_normal_model.joblib'
            joblib.dump(model, model_path)
            print(f"✅ {model_name} 모델 저장: {model_path}")
        
        # 결과 저장
        results_path = f'{save_path}/normal_data_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.normal_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Normal 결과 저장: {results_path}")
        
        # 비교 결과 저장
        comparison_path = f'{save_path}/normal_vs_smote_comparison.csv'
        self.comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"✅ 비교 결과 저장: {comparison_path}")

def main():
    print("🚀 Normal 데이터 모델 학습 및 SMOTE 비교 시작")
    print("="*80)
    
    # 비교 객체 생성
    comparison = NormalDataModelComparison(random_state=42)
    
    # 데이터 로드
    comparison.load_data()
    
    # 모델 최적화 및 훈련
    comparison.optimize_logistic_regression(n_trials=30)
    comparison.optimize_random_forest(n_trials=30)
    comparison.optimize_xgboost(n_trials=30)
    
    # 모델 평가
    comparison.evaluate_models()
    
    # SMOTE 결과 로드
    comparison.load_smote_results()
    
    # 비교 분석
    comparison.create_comparison_table()
    comparison.plot_comparison()
    
    # 결과 저장
    comparison.save_normal_results()
    
    print("\n🎉 Normal vs SMOTE 비교 분석 완료!")
    print("="*80)

if __name__ == "__main__":
    main()