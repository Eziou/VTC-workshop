import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb
from tqdm import tqdm

# 配置路径和参数
BASE_DIR = '/home/user/hrx/完整版/预训练/data'
TRAINVAL_DATA_PATH = os.path.join(BASE_DIR, 'trainval_data.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data.csv')
METHOD_DIR = '/home/user/hrx/完整版/方法'
RESULTS_DIR = os.path.join(METHOD_DIR, 'results_EWMA')
os.makedirs(RESULTS_DIR, exist_ok=True)
CV_RESULTS_PATH = os.path.join(RESULTS_DIR, 'cross_validation_results.csv')
TEST_RESULTS_PATH = os.path.join(RESULTS_DIR, 'test_results.csv')
PROGRESSIVE_RESULTS_PATH = os.path.join(RESULTS_DIR, 'progressive_results.csv')
FEATURE_IMPORTANCE_PATH = os.path.join(RESULTS_DIR, 'feature_importance_results.csv')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
ROUTER_MODEL_PATH = os.path.join(MODEL_DIR, 'router.txt')
EXPERT1_MODEL_PATH = os.path.join(MODEL_DIR, 'expert1.txt')
EXPERT2_MODEL_PATH = os.path.join(MODEL_DIR, 'expert2.txt')

PRETRAINED_ROUTER_PATH = os.path.join(BASE_DIR, 'pretrained_models/router_base.txt')
PRETRAINED_EXPERT1_PATH = os.path.join(BASE_DIR, 'pretrained_models/expert1_base_positive.txt')
PRETRAINED_EXPERT2_PATH = os.path.join(BASE_DIR, 'pretrained_models/expert2_base_negative.txt')

FEATURE_NAMES = ["K1K2 Signal", "Electronic Lock Signal", "Emergency Stop Signal",
                 "Access Control Signal", "THDV-M", "THDI-M"]
BASE_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 10, 'learning_rate': 0.05, 'max_depth': 4, 'min_data_in_leaf': 10,
    'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
    'lambda_l1': 1.0, 'lambda_l2': 2.0, 'verbose': -1, 'seed': 42, 'nthread': -1
}
EXPERT_PARAMS = BASE_PARAMS.copy()
EXPERT_PARAMS['learning_rate'] = 0.03

def load_data():
    trainval_df = pd.read_csv(TRAINVAL_DATA_PATH, header=None)
    test_df = pd.read_csv(TEST_DATA_PATH, header=None)
    X_trainval = trainval_df.iloc[:, 1:-1].values
    y_trainval = trainval_df.iloc[:, -1].values
    X_test = test_df.iloc[:, 1:-1].values
    y_test = test_df.iloc[:, -1].values
    print(f"Trainval data shape: {X_trainval.shape}, Label distribution: {np.bincount(y_trainval)}")
    print(f"Test data shape: {X_test.shape}, Label distribution: {np.bincount(y_test)}")
    return X_trainval, y_trainval, X_test, y_test

class JointEnhancedThreeLayerLGBM:
    def __init__(self):
        self.router = lgb.Booster(model_file=PRETRAINED_ROUTER_PATH)
        self.expert1 = lgb.Booster(model_file=PRETRAINED_EXPERT1_PATH)
        self.expert2 = lgb.Booster(model_file=PRETRAINED_EXPERT2_PATH)
        self.prev_weights = None
        self.alpha = 0.25
        print(f"Loaded pretrained models from {PRETRAINED_ROUTER_PATH}, {PRETRAINED_EXPERT1_PATH}, {PRETRAINED_EXPERT2_PATH}")

    def save_models(self):
        try:
            if self.router is not None:
                self.router.save_model(ROUTER_MODEL_PATH)
            if self.expert1 is not None:
                self.expert1.save_model(EXPERT1_MODEL_PATH)
            if self.expert2 is not None:
                self.expert2.save_model(EXPERT2_MODEL_PATH)
        except Exception as e:
            print(f"Error saving models: {e}")

    def compute_feedback_weights(self, y_true, final_probs):
        final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
        gradient = final_probs - y_true
        new_weights = 1.0 + np.abs(gradient) * 2.0
        if self.prev_weights is None:
            self.prev_weights = np.clip(new_weights, 0.2, 2.0)
        else:
            self.prev_weights = (1 - self.alpha) * self.prev_weights + self.alpha * new_weights
            self.prev_weights = np.clip(self.prev_weights, 0.2, 2.0)
        return self.prev_weights

    def joint_train(self, X_train, y_train, X_val, y_val, num_rounds=10):
        self.prev_weights = None  # 重置权重，避免跨样本集累积
        params = BASE_PARAMS.copy()
        params.update({'num_boost_round': 1})
        expert_params = EXPERT_PARAMS.copy()
        train_data_router = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data_router = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
        best_accuracy = 0.0
        patience = 3
        patience_counter = 0

        for round_idx in tqdm(range(num_rounds), desc="Joint Training"):
            route_probs = self.router.predict(X_train)
            routes = (route_probs > 0.5).astype(int)
            expert1_mask = (routes == 1)
            expert2_mask = (routes == 0)
            X_expert1, y_expert1 = X_train[expert1_mask], y_train[expert1_mask]
            X_expert2, y_expert2 = X_train[expert2_mask], y_train[expert2_mask]

            if len(X_expert1) < 10 or len(X_expert2) < 10:
                print(f"Round {round_idx}: Insufficient data for experts, skipping.")
                continue

            expert1_probs = self.expert1.predict(X_expert1) if len(X_expert1) > 0 else np.array([])
            expert2_probs = self.expert2.predict(X_expert2) if len(X_expert2) > 0 else np.array([])
            final_probs = np.zeros(len(X_train))
            if len(expert1_probs) > 0:
                final_probs[expert1_mask] = route_probs[expert1_mask] * expert1_probs
            if len(expert2_probs) > 0:
                final_probs[expert2_mask] = (1 - route_probs[expert2_mask]) * expert2_probs

            final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
            total_loss = -np.mean(y_train * np.log(final_probs) + (1 - y_train) * np.log(1 - final_probs))

            feedback_weights = self.compute_feedback_weights(y_train, final_probs)
            train_data_router.set_weight(feedback_weights)
            self.router = lgb.train(params, train_data_router, num_boost_round=1, init_model=self.router,
                                    valid_sets=[val_data_router])

            if len(X_expert1) >= 10:
                expert1_weights = feedback_weights[expert1_mask] * route_probs[expert1_mask]
                train_data_expert1 = lgb.Dataset(X_expert1, label=y_expert1, weight=expert1_weights, free_raw_data=False)
                self.expert1 = lgb.train(expert_params, train_data_expert1, num_boost_round=1, init_model=self.expert1,
                                         valid_sets=[lgb.Dataset(X_val, label=y_val, free_raw_data=False)])
            if len(X_expert2) >= 10:
                expert2_weights = feedback_weights[expert2_mask] * (1 - route_probs[expert2_mask])
                train_data_expert2 = lgb.Dataset(X_expert2, label=y_expert2, weight=expert2_weights, free_raw_data=False)
                self.expert2 = lgb.train(expert_params, train_data_expert2, num_boost_round=1, init_model=self.expert2,
                                         valid_sets=[lgb.Dataset(X_val, label=y_val, free_raw_data=False)])

            val_metrics = self.evaluate(X_val, y_val)
            print(f"Round {round_idx}, Loss: {total_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at round {round_idx}, best Val Accuracy: {best_accuracy:.4f}")
                    break

        self.save_models()
        return val_metrics

    def progressive_train(self, X_train, y_train, X_val, y_val, step_size=200, fold=None):
        performance_results = []
        feature_importance_results = []
        sample_sizes = np.arange(step_size, len(X_train) + 1, step_size)

        for size in tqdm(sample_sizes, desc="Progressive Training"):
            indices = np.arange(size)
            X_sub = X_train[indices]
            y_sub = y_train[indices]

            if len(np.unique(y_sub)) < 2:
                print(f"Sample size {size}: Only one class present, skipping.")
                continue

            start_time = time.time()
            val_metrics = self.joint_train(X_sub, y_sub, X_val, y_val)
            train_time = time.time() - start_time

            metrics = val_metrics.copy()
            metrics['sample_size'] = size
            metrics['training_time'] = train_time
            metrics['fold'] = fold
            performance_results.append(metrics)

            feature_importance = self.router.feature_importance(importance_type='gain')
            feature_importance_dict = {'sample_size': size, 'fold': fold}
            for i, importance in enumerate(feature_importance):
                feature_importance_dict[FEATURE_NAMES[i]] = importance
            feature_importance_results.append(feature_importance_dict)

            print(f"\nFold {fold}, Sample Size: {size}, Training Time: {train_time:.4f} seconds")
            print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, Prediction Time: {metrics['prediction_time']:.4f}")

        return performance_results, feature_importance_results

    def predict(self, X):
        start_time = time.time()
        route_probs = self.router.predict(X)
        routes = (route_probs > 0.5).astype(int)
        expert1_indices = (routes == 1)
        expert2_indices = (routes == 0)
        expert1_probs = self.expert1.predict(X[expert1_indices]) if np.sum(expert1_indices) > 0 else np.array([])
        expert2_probs = self.expert2.predict(X[expert2_indices]) if np.sum(expert2_indices) > 0 else np.array([])
        final_probs = np.zeros(len(X))
        if len(expert1_probs) > 0:
            final_probs[expert1_indices] = route_probs[expert1_indices] * expert1_probs
        if len(expert2_probs) > 0:
            final_probs[expert2_indices] = (1 - route_probs[expert2_indices]) * expert2_probs
        pred_time = time.time() - start_time
        return (final_probs > 0.5).astype(int), pred_time, final_probs

    def evaluate(self, X, y):
        final_preds, pred_time, final_probs = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, final_preds),
            'precision': precision_score(y, final_preds, zero_division=0),
            'recall': recall_score(y, final_preds, zero_division=0),
            'f1': f1_score(y, final_preds, zero_division=0),
            'prediction_time': pred_time
        }
        return metrics

def main():
    np.random.seed(42)
    print("===== Training Start =====")

    X_trainval, y_trainval, X_test, y_test = load_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    test_results = []
    progressive_results = []
    fold_feature_importance = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n===== Fold {fold + 1}/5 =====")
        X_train = X_trainval[train_idx]
        y_train = y_trainval[train_idx]
        X_val = X_trainval[val_idx]
        y_val = y_trainval[val_idx]

        model = JointEnhancedThreeLayerLGBM()
        progressive_performance, feature_importance_results = model.progressive_train(
            X_train, y_train, X_val, y_val, step_size=200, fold=fold + 1
        )
        progressive_results.extend(progressive_performance)
        fold_feature_importance.extend(feature_importance_results)

        val_metrics = model.evaluate(X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)
        val_metrics['fold'] = fold + 1
        test_metrics['fold'] = fold + 1
        cv_results.append(val_metrics)
        test_results.append(test_metrics)

        print(f"Fold {fold + 1} Validation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Fold {fold + 1} Test Metrics:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

    # 计算五折平均
    cv_df = pd.DataFrame(cv_results)
    test_df = pd.DataFrame(test_results)
    cv_mean = cv_df.mean(numeric_only=True).to_dict()
    test_mean = test_df.mean(numeric_only=True).to_dict()
    cv_mean['fold'] = 'Average'
    test_mean['fold'] = 'Average'
    cv_results.append(cv_mean)
    test_results.append(test_mean)

    # 保存结果
    pd.DataFrame(cv_results).to_csv(CV_RESULTS_PATH, index=False)
    print(f"Cross-validation results saved to {CV_RESULTS_PATH}")
    pd.DataFrame(test_results).to_csv(TEST_RESULTS_PATH, index=False)
    print(f"Test results saved to {TEST_RESULTS_PATH}")
    pd.DataFrame(progressive_results).to_csv(PROGRESSIVE_RESULTS_PATH, index=False)
    print(f"Progressive training results saved to {PROGRESSIVE_RESULTS_PATH}")
    pd.DataFrame(fold_feature_importance).to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Feature importance results saved to {FEATURE_IMPORTANCE_PATH}")

    print("\nFive-Fold Cross-Validation Average:")
    print(f"  Accuracy: {cv_mean['accuracy']:.4f}, Precision: {cv_mean['precision']:.4f}, "
          f"Recall: {cv_mean['recall']:.4f}, F1: {cv_mean['f1']:.4f}")
    print("Five-Fold Test Set Average:")
    print(f"  Accuracy: {test_mean['accuracy']:.4f}, Precision: {test_mean['precision']:.4f}, "
          f"Recall: {test_mean['recall']:.4f}, F1: {test_mean['f1']:.4f}")

    print("===== Training Completed =====")

if __name__ == "__main__":
    main()