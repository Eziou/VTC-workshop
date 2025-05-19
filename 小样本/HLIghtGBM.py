import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb

# 配置路径
BASE_DIR = '/home/user/hrx/小样本数据集'
TRAINVAL_DATA_PATH = os.path.join(BASE_DIR, 'ratio_1_1_train_1000.csv')  # 保持训练文件路径
TEST_DATA_PATH = os.path.join(BASE_DIR, 'ratio_1_1_test_1000.csv')      # 保持测试文件路径
RESULTS_DIR = '/home/user/hrx/小样本结果'
os.makedirs(RESULTS_DIR, exist_ok=True)
EVALUATION_METRICS_PATH = os.path.join(RESULTS_DIR, '1:1_1000.csv')     # 更新输出文件名

# 预训练模型路径
PRETRAINED_MODELS_DIR = '/home/user/hrx/完整版/预训练/data/pretrained_models'
PRETRAINED_ROUTER_PATH = os.path.join(PRETRAINED_MODELS_DIR, 'router_base.txt')
PRETRAINED_EXPERT1_PATH = os.path.join(PRETRAINED_MODELS_DIR, 'expert1_base_positive.txt')
PRETRAINED_EXPERT2_PATH = os.path.join(PRETRAINED_MODELS_DIR, 'expert2_base_negative.txt')

# 配置参数
BASE_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 5, 'learning_rate': 0.01, 'max_depth': 3, 'min_data_in_leaf': 20,
    'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
    'lambda_l1': 1.0, 'lambda_l2': 2.0, 'verbose': -1, 'seed': 42, 'nthread': -1
}
EXPERT_PARAMS = BASE_PARAMS.copy()
EXPERT_PARAMS['learning_rate'] = 0.005

# 加载数据
def load_data():
    trainval_df = pd.read_csv(TRAINVAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_trainval = trainval_df.iloc[:, 1:-1].values
    y_trainval = trainval_df.iloc[:, -1].values
    X_test = test_df.iloc[:, 1:-1].values
    y_test = test_df.iloc[:, -1].values
    print(f"X_trainval shape: {X_trainval.shape}, y_trainval shape: {y_trainval.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_trainval, y_trainval, X_test, y_test

# 模型定义
class JointEnhancedThreeLayerLGBM:
    def __init__(self):
        self.router = lgb.Booster(model_file=PRETRAINED_ROUTER_PATH)
        self.expert1 = lgb.Booster(model_file=PRETRAINED_EXPERT1_PATH)
        self.expert2 = lgb.Booster(model_file=PRETRAINED_EXPERT2_PATH)
        self.prev_weights = None
        self.alpha = 0.25

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
        self.prev_weights = None
        params = BASE_PARAMS.copy()
        expert_params = EXPERT_PARAMS.copy()
        train_data_router = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data_router = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
        train_data_expert1 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        train_data_expert2 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data_expert = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        # 使用 callbacks 实现早停
        self.router = lgb.train(
            params, train_data_router, num_boost_round=num_rounds,
            init_model=self.router, valid_sets=[val_data_router],
            callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)]
        )

        # 训练专家模型
        route_probs = self.router.predict(X_train)
        routes = (route_probs > 0.5).astype(int)
        expert1_mask = (routes == 1)
        expert2_mask = (routes == 0)
        X_expert1, y_expert1 = X_train[expert1_mask], y_train[expert1_mask]
        X_expert2, y_expert2 = X_train[expert2_mask], y_train[expert2_mask]

        if len(X_expert1) >= 10:
            expert1_weights = self.compute_feedback_weights(y_expert1, self.expert1.predict(X_expert1))
            train_data_expert1.set_weight(expert1_weights)
            self.expert1 = lgb.train(
                expert_params, train_data_expert1, num_boost_round=num_rounds,
                init_model=self.expert1, valid_sets=[val_data_expert],
                callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)]
            )
        if len(X_expert2) >= 10:
            expert2_weights = self.compute_feedback_weights(y_expert2, self.expert2.predict(X_expert2))
            train_data_expert2.set_weight(expert2_weights)
            self.expert2 = lgb.train(
                expert_params, train_data_expert2, num_boost_round=num_rounds,
                init_model=self.expert2, valid_sets=[val_data_expert],
                callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)]
            )

    def predict(self, X):
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
        return (final_probs > 0.5).astype(int), final_probs

    def evaluate(self, X, y):
        final_preds, final_probs = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, final_preds),
            'precision': precision_score(y, final_preds, zero_division=0),
            'recall': recall_score(y, final_preds, zero_division=0),
            'f1': f1_score(y, final_preds, zero_division=0),
        }
        return metrics

# 主函数（逐步训练）
def main():
    np.random.seed(42)
    X_trainval, y_trainval, X_test, y_test = load_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    test_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        X_train = X_trainval[train_idx]
        y_train = y_trainval[train_idx]
        X_val = X_trainval[val_idx]
        y_val = y_trainval[val_idx]
        print(f"Fold {fold}: X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

        model = JointEnhancedThreeLayerLGBM()
        step_size = 200
        for start in range(0, len(X_train), step_size):
            end = min(start + step_size, len(X_train))
            X_subset = X_train[start:end]
            y_subset = y_train[start:end]
            if len(np.unique(y_subset)) < 2:  # 确保子集包含两个类别
                continue
            model.joint_train(X_subset, y_subset, X_val, y_val, num_rounds=100)

        val_metrics = model.evaluate(X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)
        cv_results.append(val_metrics)
        test_results.append(test_metrics)

    # 计算并保存平均评估指标
    cv_df = pd.DataFrame(cv_results)
    test_df = pd.DataFrame(test_results)
    cv_mean = cv_df.mean().to_dict()
    test_mean = test_df.mean().to_dict()
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'CV_Average': [cv_mean['accuracy'], cv_mean['precision'], cv_mean['recall'], cv_mean['f1']],
        'Test_Average': [test_mean['accuracy'], test_mean['precision'], test_mean['recall'], test_mean['f1']]
    })
    metrics_df.to_csv(EVALUATION_METRICS_PATH, index=False)

if __name__ == "__main__":
    main()