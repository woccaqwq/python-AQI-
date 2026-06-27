"""
模块3：Matplotlib多维可视化图

【功能说明】
  使用Matplotlib绘制四象限集成分析图表，从不同维度展现相关规律

【依赖库】
  matplotlib - 可视化核心
  numpy      - 数值计算
  pandas     - 数据处理
"""

import matplotlib
matplotlib.use("Agg")  # 非交互后端：不弹窗，直接保存为文件
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# 中文字体配置，原生不支持中文

def setup_chinese_font():
    """
    自动检测系统已安装的中文字体并配置Matplotlib使用。
    配置项说明：
      font.sans-serif:无衬线字体优先列表
      axes.unicode_minus:避免负号"-"被误渲染为其他字符
    """
    # 候选中文字体，按普及程度排序
    candidates = [
        "SimHei",                    # 黑体
        "Microsoft YaHei",           # 微软雅黑（Windows）
        "WenQuanYi Micro Hei",       # 文泉驿微米黑（Linux）
        "Noto Sans CJK SC",          # Google Noto简体中文（跨平台）
        "Source Han Sans SC",        # 思源黑体
        "PingFang SC",               # 苹方（macOS）
        "STSong", "FangSong", "KaiTi", "SimSun",  # 传统中文字体备选
    ]
    available = {f.name for f in fm.fontManager.ttflist}    #遍历matplotlib系统中所有已注册的字体文件供后续判断

    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  使用中文字体: {font_name}")
            return font_name

    # 兜底：模糊搜索所有含中文特征词的字体
    cjk_fonts = [f.name for f in fm.fontManager.ttflist
                 if any(kw in f.name for kw in ["Hei", "Song", "Ming", "Kai", "Fang", "CJK", "CN"])]
    if cjk_fonts:
        font_name = cjk_fonts[0]
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"  使用中文字体(自动): {font_name}")
        return font_name

    # 没有中文字体时也不中断
    print("  [警告] 未检测到中文字体，图表中文可能显示为方块")
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"


# 子图①：时序折线图（左上）

def plot_timeseries(ax, df, focus_city="北京", predictions=None):
    """
    绘制近90天AQI日趋势折线图，叠加7天预测虚线。
    参数:
        ax-matplotlib子图对象
        predictions-未来7天AQI预测值
    """
    # 构建数据透视表：行=日期，列=城市，值=AQI
    pivot = df.pivot_table(values="AQI", index="日期", columns="城市")

    # 确保每天只有一个值，然后取最近90天
    daily = pivot.resample("D").mean()
    recent = daily.iloc[-90:]

    # 背景城市（灰色淡线）
    bg_cities = [c for c in recent.columns if c != focus_city]
    for city in bg_cities:
        ax.plot(recent.index, recent[city], color="gray", alpha=0.12, linewidth=0.8)

    # 焦点城市（红色高亮）
    if focus_city in recent.columns:
        ax.plot(recent.index, recent[focus_city], color="#E74C3C", linewidth=2.2,
                label=focus_city, zorder=5)

    # 7天预测虚线（接在历史数据末端）
    if predictions is not None and len(predictions) > 0:
        last_date = recent.index[-1]  # 历史数据的最后一天
        # 预测日期序列
        pred_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                   periods=len(predictions), freq="D")
        ax.plot(pred_dates, predictions, color="#E74C3C", linewidth=2.2,
                linestyle="--", label="预测趋势", zorder=5)
        ax.scatter(pred_dates, predictions, color="#E74C3C", s=25, zorder=5)

    # 标注与美化
    ax.set_title("① 近90天AQI日趋势与7天预测", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("AQI")
    ax.grid(True, alpha=0.3)    #加浅色网格线，方便肉眼对日期和AQI数值定位


# 子图②：排名条形图（右上）

def plot_ranking(ax, df):
    """
    绘制31城市AQI均值排名：Top10最优（绿色） + Top10最差（红色）。
    """
    # 计算每个城市的平均AQI，升序排列（最低=最好，最高=最差）
    city_avg = df.groupby("城市")["AQI"].mean().sort_values()

    # 取头10名和尾10名
    best10 = city_avg.head(10)
    worst10 = city_avg.tail(10)

    # 拼接：先展示最优10城，再展示最差10城
    cities_list = list(best10.index) + list(worst10.index)
    values_list = list(best10.values) + list(worst10.values)
    colors = (["#27AE60"] * 10 + ["#E74C3C"] * 10)  # 绿最优，红最差

    # 绘制水平条形图
    y_pos = range(len(cities_list))
    ax.barh(y_pos, values_list, color=colors, alpha=0.85, height=0.65)

    # 刻度标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities_list, fontsize=8)
    ax.set_xlabel("平均AQI")
    ax.set_title("② 全国城市AQI排名 (Top10最优·最差)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # 翻转Y轴：最优排在最上面

    # 全国中位数参考线
    ax.axvline(x=city_avg.median(), color="gray", linestyle="--", alpha=0.5, label="全国中位数")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # 在每个条形右侧标注数值
    for i, v in enumerate(values_list):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=7)


# 子图③：雷达图（左下）

def plot_radar(fig, df, focus_city="北京"):
    """
    绘制焦点城市六大污染物浓度分布的雷达图（极坐标）。
    参数:fig-matplotlib画布对象
    """
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    labels = pollutants  

    # 计算焦点城市的均值、全国均值、全国最大值
    city_data = df[df["城市"] == focus_city][pollutants].mean()
    national_avg = df[pollutants].mean()
    max_vals = df[pollutants].max()

    # 归一化：每个污染物除以各自的全国最大值使其值域[0,1]
    city_norm = []
    national_norm = []
    for p in pollutants:
        mx = max_vals[p]
        city_norm.append(city_data[p] / mx if mx > 0 else 0)
        national_norm.append(national_avg[p] / mx if mx > 0 else 0)

    # 构建雷达图角度坐标
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]       # 首尾相接，使多边形闭合
    city_norm += city_norm[:1]
    national_norm += national_norm[:1]

    # 创建极坐标子图
    ax = fig.add_subplot(2, 2, 3, projection="polar")
    ax.fill(angles, city_norm, alpha=0.25, color="#3498DB")
    ax.plot(angles, city_norm, "o-", color="#3498DB", linewidth=2, label=focus_city)
    ax.plot(angles, national_norm, "o-", color="gray", linewidth=1.5, label="全国均值", alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("③ 六大污染物浓度分布", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return ax


# 子图④：相关性热力图（右下）

def plot_heatmap(ax, df):
    """
    绘制六大污染物Pearson相关系数热力图。
    """
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    labels = pollutants
    corr = df[pollutants].corr(method="pearson")  # 6×6 Pearson相关矩阵

    # imshow: 将矩阵渲染为热力图
    # RdYlBu_r = 红-黄-蓝反转色图(红色=低/负相关, 蓝色=高/正相关)
    # vmin/vmax: 固定色阶范围到[-1, 1]
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")

    # 在每个格子内标注相关系数值
    for i in range(len(pollutants)):
        for j in range(len(pollutants)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if abs(val) > 0.5 else "black")

    # 刻度标签
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("④ 污染物Pearson相关性热力图", fontsize=14, fontweight="bold")

    # 添加色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("相关系数", fontsize=10)


# 主画布编排

def create_dashboard(df, focus_city="北京", predictions=None, output_path="output/dashboard.png"):
    """
    绘制完整的四象限可视化大屏并保存为PNG文件
    """
 
    setup_chinese_font()

    # 创建2×2画布，总尺寸14×10
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("全国核心城市空气质量(AQI)多维统计分析与预测系统",
                 fontsize=18, fontweight="bold", y=0.98)

    print(f"  焦点城市: {focus_city}")

    # 依次绘制四个子图
    print("  绘制子图①: 时序折线图...")
    plot_timeseries(axes[0, 0], df, focus_city, predictions)

    print("  绘制子图②: 城市排行榜...")
    plot_ranking(axes[0, 1], df)

    print("  绘制子图③: 污染物雷达图...")
    # 雷达图需要极坐标，先删除默认的普通子图，再创建极坐标子图
    axes[1, 0].remove()
    plot_radar(fig, df, focus_city)

    print("  绘制子图④: 相关性热力图...")
    plot_heatmap(axes[1, 1], df)

    # 调整子图间距，避免标签重叠
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93)

    # 保存为PNG文件（DPI=150保证清晰度）
    import os
    os.makedirs("output", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\n图表已保存: {output_path}")
    plt.close(fig)  # 释放内存
    return output_path


# 独立运行入口
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_aqi.csv", encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    create_dashboard(df, focus_city="北京")
