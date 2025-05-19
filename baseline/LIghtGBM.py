import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import time

DATA_PATH = '/home/user/hrx/CSdata/data.csv'
ROUTER_BASE_PATH = '/home/user/hrx/test/models/router_base.txt'

def load_data():
    column_names = ['ID', 'K1K2驱动信号', '电子锁驱动信号', '急停信号',
                    '门禁信号', 'THDV-M', 'THDI-M', 'label']
    df = pd.read_csv(DATA_PATH, header=None, names=column_names)
    X = df.drop(['ID', 'label'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_and_save_router(X_train, X_test, y_train, y_test):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 10, 'learning_rate': 0.05, 'max_depth': 4, 'min_data_in_leaf': 10,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
        'lambda_l1': 1.0, 'lambda_l2': 2.0, 'verbose': -1, 'seed': 42
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    print("Training LightGBM model as router base...")
    model = lgb.train(
        params, train_data, num_boost_round=200, valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    model.save_model(ROUTER_BASE_PATH)
    print(f"Saved router base to {ROUTER_BASE_PATH}")

    start_time = time.time()
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    pred_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nRouter Base Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Prediction Time: {pred_time:.4f} seconds")

    return model

if __name__ == "__main__":
    np.random.seed(42)
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    train_and_save_router(X_train, X_test, y_train, y_test)