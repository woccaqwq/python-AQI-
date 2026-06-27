"""
模块4：机器学习短期趋势预测

【功能说明】
  使用随机森林回归算法，基于过去7天的空气质量与污染物数据，预测第8天
  的AQI值，并通过迭代预测延伸至未来7天。

【依赖库】
  numpy          - 数组操作
  pandas         - 数据处理
  scikit-learn   - 随机森林模型、评估指标
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 滑动窗口特征工程

def build_sliding_window(series, window=7):
    """
    将一维时间序列转换为监督学习的特征矩阵和标签向量。
    参数: series-一维时间序列（如某城市的AQI序列）
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])      # 特征矩阵，第i天到第i+6天的值
        y.append(series[i + window])        # 标签向量，第i+7天的值

    if len(X) == 0:
        return None, None, None

    # 保留最新窗口，后续预测使用
    last_window = series[-window:].tolist()
    return np.array(X), np.array(y), last_window


def prepare_features_for_city(df, city, window=7):
    """
    为指定城市构造完整的特征矩阵。
    标签：第t天的AQI实际值
    """
    city_df = df[df["城市"] == city].sort_values("日期")

    pollutants = ["AQI", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    all_features = []
    y_values = None
    last_windows = {}

    for p in pollutants:
        series = city_df[p].values
        X_p, y_p, last_w = build_sliding_window(series, window)

        # 以AQI的未来值作为标签（所有特征共享同一标签）
        if p == "AQI":
            y_values = y_p

        all_features.append(X_p)
        last_windows[p] = last_w

    if not all_features or y_values is None:
        return None, None, None

    # 横向拼接
    X = np.hstack(all_features)
    return X, y_values, last_windows


# 模型训练与评估

def train_model(X, y, focus_city="北京"):
    """
    训练随机森林回归模型并在测试集上评估。
    随机森林参数说明：
      n_estimators=150 : 150棵决策树（越多越稳定，但计算慢）
      max_depth=10     : 每棵树最多10层（限制复杂度，防止过拟合）
      random_state=42  : 固定随机种子（保证结果可复现）
      n_jobs=-1        : 使用所有CPU核心并行训练
    评估指标：
      RMSE: 均方根误差，值越小越好
          ——对大幅偏差特别敏感，适合发现"离谱"的预测
      MAE: 平均绝对误差，值越小越好
          ——直观反映预测值与真实值的平均差距（单位：AQI点数）
      R²: 决定系数，范围(-∞, 1]，越接近1越好
          ——衡量模型解释了多少数据变异，0.8表示解释了80%
    """
    # 时序划分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"  特征维度: {X.shape[1]} (7天 × 7种污染物)")

    # 创建并训练模型
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 在测试集上预测并评估
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  [{focus_city}] 模型评估结果:")
    print(f"    RMSE: {rmse:.2f}  —— 预测值与真实值的均方根偏差")
    print(f"    MAE:  {mae:.2f}  —— 预测值与真实值的平均绝对偏差")
    print(f"    R²:   {r2:.4f} —— 模型解释的变异比例")

    return model, {"RMSE": rmse, "MAE": mae, "R2": r2}


# 未来预测

def forecast_future(model, last_windows, days=7):
    """
    基于最近7天实际数据，迭代预测未来N天的AQI
    其他污染物处理：未来PM2.5/PM10等未知，简单沿用最新值
    """
    pollutants = ["AQI", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

    # 初始化各污染物的滑动窗口（取最近7天）
    current_windows = {}
    for p in pollutants:
        vals = last_windows.get(p, [0] * 7)     # 如果某种污染物意外缺失，用7个0兜底
        # 确保窗口正好7天    
        if len(vals) >= 7:
            current_windows[p] = vals[-7:]
        
    predictions = []
    for _ in range(days):
        # 构建49维特征向量
        features = []
        for p in pollutants:
            features.extend(current_windows[p])     # 列表拼接，与训练规格一致

        # 预测下一天AQI，结果浮点并保留一位小数
        pred = model.predict([features])[0]
        predictions.append(float(round(pred, 1)))

        # 滑动窗口：去掉第一天，加入新预测的AQI
        current_windows["AQI"] = current_windows["AQI"][1:] + [pred]
        # 其他污染物窗口也前移，最新值沿用
        for p in pollutants[1:]:
            current_windows[p] = current_windows[p][1:] + [current_windows[p][-1]]

    return predictions


# 主流程

def run_prediction(df_multi, focus_city="北京", window=7):
    """
    完整预测流程
    参数:df_multi-多级索引的清洗后数据
    """

    # 将多级索引还原为平面DataFrame
    df = df_multi.reset_index()

    # 步骤1: 特征工程
    print(f"\n[1/3] 构建滑动窗口特征 (窗口={window}天)...")
    X, y, last_windows = prepare_features_for_city(df, focus_city, window)

    if X is None:
        print("  [失败] 无法构建特征矩阵，请检查数据充足性")
        return None, None

    # 步骤2: 训练与评估
    print(f"[2/3] 训练随机森林模型...")
    model, metrics = train_model(X, y, focus_city)

    # 步骤3: 预测未来
    print(f"\n[3/3] 预测未来7天AQI趋势...")
    predictions = forecast_future(model, last_windows, days=7)

    # 显示预测结果
    print(f"\n  未来7天AQI预测值:")
    from datetime import datetime, timedelta
    today = datetime.now()
    for i, val in enumerate(predictions):
        date = today + timedelta(days=i + 1)
        print(f"    {date.strftime('%m月%d日')}: {val}")

    return predictions, metrics


# 独立运行入口
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_aqi.csv", encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    df_multi = df.set_index(["城市", "日期"]).sort_index()
    predictions, metrics = run_prediction(df_multi, focus_city="北京")
