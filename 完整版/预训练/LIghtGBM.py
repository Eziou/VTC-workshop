import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# 配置路径
DATA_DIR = '/home/user/hrx/完整版/预训练/data'
PRETRAIN_DATA_PATH = os.path.join(DATA_DIR, 'pretrain_data.csv')
MODEL_DIR = os.path.join(DATA_DIR, 'pretrained_models')
os.makedirs(MODEL_DIR, exist_ok=True)
ROUTER_MODEL_PATH = os.path.join(MODEL_DIR, 'router_base.txt')
EXPERT1_MODEL_PATH = os.path.join(MODEL_DIR, 'expert1_base_positive.txt')
EXPERT2_MODEL_PATH = os.path.join(MODEL_DIR, 'expert2_base_negative.txt')

# 优化后的参数（初始设置，可根据验证集进一步调整）
ROUTER_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 15, 'learning_rate': 0.1, 'max_depth': 6, 'min_data_in_leaf': 20,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 0.5, 'lambda_l2': 1.0, 'verbose': -1, 'seed': 42
}
EXPERT_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 15, 'learning_rate': 0.05, 'max_depth': 6, 'min_data_in_leaf': 20,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 0.5, 'lambda_l2': 1.0, 'verbose': -1, 'seed': 42
}


def load_and_split_data():
    """加载并分割预训练数据集"""
    df = pd.read_csv(PRETRAIN_DATA_PATH, header=None)
    print(f"Pretrain data shape: {df.shape}, Label distribution: {df.iloc[:, -1].value_counts().to_dict()}")

    # 数据格式：8 列，第 0 列是标号，1-6 列是特征，第 7 列是标签
    X = df.iloc[:, 1:-1].values  # 6 个特征
    y = df.iloc[:, -1].values  # 标签

    # 分成训练集 (80%) 和验证集 (20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train data shape: {X_train.shape}, Label distribution: {np.bincount(y_train)}")
    print(f"Val data shape: {X_val.shape}, Label distribution: {np.bincount(y_val)}")

    return X_train, X_val, y_train, y_val


def train_model(params, X_train, y_train, X_val, y_val, model_path, model_name):
    """训练 LightGBM 模型，使用早停并保存"""
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    # 训练模型，使用早停
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=10)]
    )

    # 保存模型
    model.save_model(model_path)
    print(f"{model_name} trained and saved to {model_path}")
    return model


def main():
    print("===== Pretraining Start =====")

    # 加载并分割数据
    X_train, X_val, y_train, y_val = load_and_split_data()

    # 训练路由器
    router = train_model(ROUTER_PARAMS, X_train, y_train, X_val, y_val, ROUTER_MODEL_PATH, "Router")

    # 用路由器分类训练数据
    route_probs = router.predict(X_train)
    routes = (route_probs > 0.5).astype(int)

    # 分割数据给专家模型
    X_expert1 = X_train[routes == 1]  # 路由器预测为正
    y_expert1 = y_train[routes == 1]
    X_expert2 = X_train[routes == 0]  # 路由器预测为负
    y_expert2 = y_train[routes == 0]

    # 检查分类结果
    print(f"Expert 1 data (predicted positive): {len(X_expert1)} samples, Label distribution: {np.bincount(y_expert1)}")
    print(f"Expert 2 data (predicted negative): {len(X_expert2)} samples, Label distribution: {np.bincount(y_expert2)}")

    # 训练专家模型
    if len(X_expert1) >= 10:
        expert1 = train_model(EXPERT_PARAMS, X_expert1, y_expert1, X_val, y_val, EXPERT1_MODEL_PATH,
                              "Expert 1 (positive)")
    else:
        print("Insufficient data for Expert 1")

    if len(X_expert2) >= 10:
        expert2 = train_model(EXPERT_PARAMS, X_expert2, y_expert2, X_val, y_val, EXPERT2_MODEL_PATH,
                              "Expert 2 (negative)")
    else:
        print("Insufficient data for Expert 2")

    print("===== Pretraining Completed =====")


if __name__ == "__main__":
    main()