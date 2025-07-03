#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Random Forest Modeling Pipeline
Based on the simple_modeling_pipeline.py structure with Random Forest

Key features:
1. Class imbalance handling with class_weight='balanced'
2. Optimal threshold finding using F1 score maximization
3. Robust error handling and data validation
4. Simple, focused approach without over-complex optimization
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, 
    precision_recall_curve
)
from pathlib import Path

class SimpleRandomForestPipeline:
    """
    Simple Random Forest modeling pipeline
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
    
    def check_class_distribution(self):
        """Check class distribution for monitoring"""
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        
        print(f"클래스 분포 - 정상: {neg_count}, 부실: {pos_count}")
        print(f"class_weight='balanced' 사용")
    
    def train_model(self):
        """Train Random Forest model"""
        print("🚀 모델 학습 시작...")
        
        # Check class distribution
        self.check_class_distribution()
        
        # Define model with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=500,  # Number of trees
            max_depth=10,  # Maximum depth of trees
            min_samples_split=2,  # Minimum samples to split a node
            min_samples_leaf=1,  # Minimum samples in leaf
            max_features='sqrt',  # Number of features to consider
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1,  # Use all available cores
            bootstrap=True,  # Bootstrap sampling
            oob_score=True  # Out-of-bag score for evaluation
        )
        
        # Train the model
        try:
            self.model.fit(self.X_train, self.y_train)
            print("✅ 모델 학습 완료!")
            print(f"OOB Score: {self.model.oob_score_:.4f}")
            
        except Exception as e:
            print(f"❌ 모델 학습 중 에러: {e}")
            raise
    
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
    
    def show_feature_importance(self, top_n=20):
        """Show top feature importances"""
        if self.model is None:
            print("❌ 모델이 학습되지 않았습니다.")
            return
        
        feature_names = self.X_train.columns
        importances = self.model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 상위 {top_n}개 중요 특성:")
        print("=" * 50)
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30}: {row['importance']:.4f}")
    
    def run_pipeline(self):
        """Run the complete modeling pipeline"""
        print("=" * 50)
        print("🚀 Simple Random Forest Modeling Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model()
        
        # Show feature importance
        self.show_feature_importance()
        
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
        print(f"OOB Score: {self.model.oob_score_:.4f}")
        
        # 결과지표 저장
        self.save_results(y_test_pred, y_test_proba)
        
        return {
            'model': self.model,
            'optimal_threshold': self.optimal_threshold,
            'test_f1': f1_score(self.y_test, y_test_pred),
            'test_auc': roc_auc_score(self.y_test, y_test_proba),
            'oob_score': self.model.oob_score_
        }
    
    def save_model(self, model_name="simple_random_forest_model"):
        """Save the trained model"""
        try:
            joblib.dump(self.model, f"{model_name}.pkl")
            
            # Save threshold info
            threshold_info = {
                'optimal_threshold': self.optimal_threshold,
                'optimal_f1': self.optimal_f1
            }
            joblib.dump(threshold_info, f"{model_name}_threshold.pkl")
            
            print(f"\n💾 모델 저장 완료:")
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
                'Model': 'Random_Forest',
                'Test_Accuracy': accuracy_score(self.y_test, y_test_pred),
                'Test_F1': f1_score(self.y_test, y_test_pred),
                'Test_ROC_AUC': roc_auc_score(self.y_test, y_test_proba),
                'Optimal_Threshold': self.optimal_threshold,
                'OOB_Score': self.model.oob_score_,
                'Train_Size': len(self.y_train),
                'Test_Size': len(self.y_test)
            }
            
            results_df = pd.DataFrame([results])
            results_df.to_csv("outputs/modeling/random_forest_results.csv", index=False)
            print(f"💾 결과지표 저장: outputs/modeling/random_forest_results.csv")
            
        except Exception as e:
            print(f"❌ 결과지표 저장 중 에러: {e}")


def main():
    """Main execution function"""
    # Initialize and run pipeline
    pipeline = SimpleRandomForestPipeline()
    results = pipeline.run_pipeline()
    
    # Save model
    pipeline.save_model()
    
    return results


if __name__ == "__main__":
    results = main()