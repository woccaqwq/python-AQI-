"""
模块2：数据清洗与融合

【功能说明】
  读取爬虫产出的各城市CSV文件，进行数据清洗、缺失值处理、多城市融合，
  并计算六大污染物之间的Pearson相关系数矩阵。

【依赖库】
  pandas  - 数据处理核心
  numpy   - 数值计算
  glob    - 文件搜索
"""

import pandas as pd
import numpy as np
import os
import glob


# 数据加载

def load_all_cities(data_dir="data"):
    """
    读取data/目录下所有城市CSV文件，合并为一个大DataFrame。
    返回: pandas.DataFrame - 合并后的数据，包含所有城市的记录
    列: 日期, 质量等级, AQI, PM2.5, PM10, SO2, NO2, CO, O3, 城市
    """
    # 获取所有CSV文件路径，排除清洗输出文件
    files = [f for f in glob.glob(os.path.join(data_dir, "*.csv"))
             if os.path.basename(f) != "cleaned_aqi.csv"]

    if not files:
        raise FileNotFoundError(f"{data_dir}/ 目录没有CSV文件，请先运行 crawler.py")

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")  # 兼容带BOM的UTF-8
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")  # 把日期列从str类型转成datetime类型，非日期格式NaT
        dfs.append(df)

    # 纵向拼接所有城市数据
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["日期"])          # 删除日期解析失败的行
    combined = combined.sort_values(["城市", "日期"]).reset_index(drop=True)    #重新双层排序并重新排前缀行号
    return combined


# 数据清洗

def clean_data(df):
    """
    对原始数据进行清洗：数值转换，缺失值填充。
    参数:df原始合并数据
    """
    # 步骤1: 将目标列转换为数值类型
    num_cols = ["AQI", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")# 遇到无法转换的值（如中文"--"）时转为NaN，而非报错

    # 步骤2: 按城市分组，各自独立填充缺失值
    def clean_group(g):
        """对单个城市的数据进行填充"""
        g = g.set_index("日期")                        # 以日期为索引，便于时间序列操作
        g[num_cols] = g[num_cols].ffill()              # 前向填充（用前一天填今天）
        g[num_cols] = g[num_cols].infer_objects(copy=False).interpolate(method="linear")  # 线性插值（两端平均值）
        return g.reset_index()                         # 恢复日期为普通列

    df = df.groupby("城市", group_keys=False).apply(clean_group, include_groups=True) #按城市组分类应用填充函数

    # 步骤3: 报告并删除仍缺失的行
    before = len(df)
    df = df.dropna(subset=num_cols)     #删除num_cols七列中任意一列仍为NaN的行
    print(f"  缺失值处理后：{before} → {len(df)} 行 (删除{before - len(df)}行)")

    return df


# 统计分析

def compute_correlation(df):
    """
    计算六大主要污染物之间的Pearson6×6的相关系数矩阵
    """
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    corr = df[pollutants].corr(method="pearson")
    return corr


# 数据结构构建

def build_multi_index(df):
    """
    构建(城市, 日期)双层索引的DataFrame。
    """
    return df.set_index(["城市", "日期"]).sort_index()


# 主流程

def run_cleaning(data_dir="data"):
    """
    完整数据清洗流程
    返回:
        df_multi-多级索引的清洗后数据（用于预测）
        corr-污染物相关系数矩阵（用于可视化热力图）
    """
    # 加载
    print("\n[1/3] 加载所有城市数据...")
    df = load_all_cities(data_dir)
    print(f"  加载 {df['城市'].nunique()} 个城市, {len(df)} 条记录")

    # 清洗
    print("\n[2/3] 数据清洗中...")
    df = clean_data(df)
    print(f"  清洗完成: {len(df)} 条有效记录")

    # 分析
    print("\n[3/3] 计算Pearson相关系数矩阵...")
    corr = compute_correlation(df)
    print("  六大污染物相关系数矩阵:")
    print(corr.round(3).to_string())  # 保留3位小数

    # 构建多级索引
    df_multi = build_multi_index(df)

    # 保存清洗结果
    df.to_csv("data/cleaned_aqi.csv", index=False, encoding="utf-8-sig")
    print("\n清洗后数据已保存: data/cleaned_aqi.csv")

    return df_multi, corr


# 独立运行入口
if __name__ == "__main__":
    run_cleaning()
