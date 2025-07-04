#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple XGBoost Modeling Pipeline
Based on the better-performing baseline from jongho.ipynb

Key improvements:
1. Automatic scale_pos_weight calculation for class imbalance
2. Version-agnostic XGBoost fitting with fallback mechanisms  
3. Optimal threshold finding using F1 score maximization
4. Robust error handling and data validation
5. Simple, focused approach without over-complex optimization
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import inspect
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, 
    precision_recall_curve
)
from pathlib import Path

class SimpleModelingPipeline:
    """
    Simple XGBoost modeling pipeline based on jongho.ipynb baseline
    """
    
    def __init__(self, data_path="data/final"):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the preprocessed data directory
        """
        self.data_path = Path(data_path)
        self.model = None
        self.optimal_threshold = 0.5
        self.optimal_f1 = 0.0
        
    def load_data(self):
        """Load preprocessed data from final directory"""
        print("📊 데이터 로드 중...")
        
        try:
            self.X_train = pd.read_csv(self.data_path / 'X_train.csv')
            self.X_val = pd.read_csv(self.data_path / 'X_val.csv')
            self.X_test = pd.read_csv(self.data_path / 'X_test.csv')
            
            self.y_train = pd.read_csv(self.data_path / 'y_train.csv').iloc[:, 0]
            self.y_val = pd.read_csv(self.data_path / 'y_val.csv').iloc[:, 0]
            self.y_test = pd.read_csv(self.data_path / 'y_test.csv').iloc[:, 0]
            
            print(f"✅ 데이터 로드 완료:")
            print(f"  Train: {self.X_train.shape}, 부실비율: {self.y_train.mean():.2%}")
            print(f"  Val:   {self.X_val.shape}, 부실비율: {self.y_val.mean():.2%}")
            print(f"  Test:  {self.X_test.shape}, 부실비율: {self.y_test.mean():.2%}")
            
        except Exception as e:
            print(f"❌ 데이터 로드 에러: {e}")
            raise
    
    def calculate_scale_pos_weight(self):
        """Calculate scale_pos_weight for class imbalance handling"""
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        print(f"클래스 분포 - 정상: {neg_count}, 부실: {pos_count}")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def train_model(self):
        """Train XGBoost model with robust version handling"""
        print("🚀 모델 학습 시작...")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = self.calculate_scale_pos_weight()
        
        # Define model with optimized parameters from baseline
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,  # Class imbalance handling
            verbosity=0  # Minimize logging output
        )
        
        eval_set = [(self.X_val, self.y_val)]
        
        # Version-agnostic fitting with robust error handling
        try:
            # Check if early_stopping_rounds parameter is supported (XGBoost ≤ 1.7.x)
            sig = inspect.signature(self.model.fit).parameters
            if "early_stopping_rounds" in sig:
                # XGBoost ≤ 1.7.x
                print("Using XGBoost ≤ 1.7.x early stopping")
                self.model.fit(
                    self.X_train, self.y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                # XGBoost ≥ 2.0
                print("Using XGBoost ≥ 2.0 callbacks")
                from xgboost.callback import EarlyStopping
                self.model.fit(
                    self.X_train, self.y_train,
                    eval_set=eval_set,
                    callbacks=[EarlyStopping(rounds=50, save_best=True)],
                    verbose=False
                )
        except Exception as e:
            print(f"Early stopping 실패, fallback으로 재시도: {e}")
            # Fallback: train without early stopping
            self.model.fit(self.X_train, self.y_train, verbose=False)
        
        print("✅ 모델 학습 완료!")
    
    def find_optimal_threshold(self):
        """Find optimal threshold using F1 score maximization"""
        print("🎯 최적 임계값 찾는 중...")
        
        # Get validation predictions
        y_val_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_val, y_val_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        self.optimal_f1 = f1_scores[optimal_idx]
        
        print(f"✅ 최적 임계값: {self.optimal_threshold:.4f} (F1: {self.optimal_f1:.4f})")
    
    def evaluate(self, split_name, X, y, use_optimal_threshold=True):
        """Evaluate model performance"""
        y_proba = self.model.predict_proba(X)[:, 1]
        
        if use_optimal_threshold:
            y_pred = (y_proba >= self.optimal_threshold).astype(int)
            threshold_used = self.optimal_threshold
        else:
            y_pred = self.model.predict(X)
            threshold_used = 0.5
        
        print(f"\n[{split_name} Set] (Threshold: {threshold_used:.4f})")
        print("Accuracy :", f"{accuracy_score(y, y_pred):.4f}")
        print("F1-score :", f"{f1_score(y, y_pred):.4f}")
        print("ROC AUC  :", f"{roc_auc_score(y, y_proba):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        return y_pred, y_proba
    
    def run_pipeline(self):
        """Run the complete modeling pipeline"""
        print("=" * 50)
        print("🚀 Simple XGBoost Modeling Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model()
        
        # Find optimal threshold
        self.find_optimal_threshold()
        
        # Evaluate on validation and test sets
        print("\n" + "=" * 50)
        print("📊 성능 평가 결과")
        print("=" * 50)
        
        # Validation evaluation with optimal threshold
        y_val_pred, y_val_proba = self.evaluate("Validation", self.X_val, self.y_val, use_optimal_threshold=True)
        
        # Test evaluation with optimal threshold  
        y_test_pred, y_test_proba = self.evaluate("Test", self.X_test, self.y_test, use_optimal_threshold=True)
        
        # Compare with default threshold (0.5)
        print("\n" + "=" * 30 + " 기본 임계값(0.5) 비교 " + "=" * 30)
        self.evaluate("Test (Default 0.5)", self.X_test, self.y_test, use_optimal_threshold=False)
        
        # Final summary
        print("\n" + "=" * 50)
        print("📋 최종 성능 요약")
        print("=" * 50)
        print(f"Test F1 Score (최적 임계값): {f1_score(self.y_test, y_test_pred):.4f}")
        print(f"Test ROC AUC: {roc_auc_score(self.y_test, y_test_proba):.4f}")
        print(f"사용된 최적 임계값: {self.optimal_threshold:.4f}")
        
        # 결과지표 저장
        self.save_results(y_test_pred, y_test_proba)
        
        return {
            'model': self.model,
            'optimal_threshold': self.optimal_threshold,
            'test_f1': f1_score(self.y_test, y_test_pred),
            'test_auc': roc_auc_score(self.y_test, y_test_proba)
        }
    
    def save_model(self, model_name="simple_xgb_model"):
        """Save the trained model"""
        try:
            self.model.save_model(f"{model_name}.json")
            joblib.dump(self.model, f"{model_name}.pkl")
            
            # Save threshold info
            threshold_info = {
                'optimal_threshold': self.optimal_threshold,
                'optimal_f1': self.optimal_f1
            }
            joblib.dump(threshold_info, f"{model_name}_threshold.pkl")
            
            print(f"\n💾 모델 저장 완료:")
            print(f"  - {model_name}.json")
            print(f"  - {model_name}.pkl") 
            print(f"  - {model_name}_threshold.pkl")
            print(f"  - 최적 임계값: {self.optimal_threshold:.4f}")
            
        except Exception as e:
            print(f"❌ 모델 저장 중 에러: {e}")
    
    def save_results(self, y_test_pred, y_test_proba):
        """결과지표 저장"""
        try:
            import os
            os.makedirs("outputs/modeling", exist_ok=True)
            
            results = {
                'Model': 'XGBoost',
                'Test_Accuracy': accuracy_score(self.y_test, y_test_pred),
                'Test_F1': f1_score(self.y_test, y_test_pred),
                'Test_ROC_AUC': roc_auc_score(self.y_test, y_test_proba),
                'Optimal_Threshold': self.optimal_threshold,
                'Train_Size': len(self.y_train),
                'Test_Size': len(self.y_test)
            }
            
            results_df = pd.DataFrame([results])
            results_df.to_csv("outputs/modeling/xgb_results.csv", index=False)
            print(f"💾 결과지표 저장: outputs/modeling/xgb_results.csv")
            
        except Exception as e:
            print(f"❌ 결과지표 저장 중 에러: {e}")


def main():
    """Main execution function"""
    # Initialize and run pipeline
    pipeline = SimpleModelingPipeline()
    results = pipeline.run_pipeline()
    
    # Save model
    pipeline.save_model()
    
    return results


if __name__ == "__main__":
    results = main()