"""
主函数 —— 全国核心城市空气质量(AQI)多维统计分析与预测系统脚本
"""
import os

# 固定的31个核心城市列表
ALL_CITIES = [
    "北京", "上海", "天津", "重庆",
    "哈尔滨", "长春", "沈阳", "呼和浩特",
    "石家庄", "太原", "济南", "郑州",
    "南京", "合肥", "武汉", "长沙",
    "南昌", "杭州", "福州", "南宁",
    "广州", "海口", "贵阳", "昆明",
    "成都", "拉萨", "西宁", "兰州",
    "银川", "西安", "乌鲁木齐",
]


def print_city_list():
    """以多列格式打印31个城市名"""
    print("\n  可用城市（输入序号或城市名均可）:")
    for row in range(0, len(ALL_CITIES), 6):
        line = ""
        for col in range(6):
            idx = row + col
            if idx < len(ALL_CITIES):
                line += f"  {idx+1:2d}.{ALL_CITIES[idx]:　<5}"
        print(line)
    print("  直接回车默认选择 1.北京")


def choose_city():
    """用序号或城市名选择焦点城市"""
    print_city_list()
    while True:
        choice = input("\n  请选择焦点城市 (默认 1): ").strip()
        if not choice:
            return "北京"
        # 尝试按序号匹配
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(ALL_CITIES):
                return ALL_CITIES[idx]
        # 尝试按城市名匹配
        if choice in ALL_CITIES:
            return choice
        print(f"  [错误] '{choice}' 不是有效选择，请重新输入")


def run_pipeline(focus_city):
    """为指定城市运行清洗→预测→可视化，返回图表路径"""
    # 数据清洗与融合
    print("\n" + "=" * 60)
    print("  [数据清洗]")
    print("=" * 60)

    from data_cleaner import run_cleaning
    df_multi, corr = run_cleaning()

    # 机器学习预测
    print("\n" + "=" * 60)
    print("  [机器学习预测]")
    print("=" * 60)

    from ml_predict import run_prediction
    predictions, metrics = run_prediction(df_multi, focus_city=focus_city)

    # 可视化
    print("\n" + "=" * 60)
    print("  [可视化]")
    print("=" * 60)

    from visualization import create_dashboard
    df_flat = df_multi.reset_index()
    output_path = f"output/{focus_city}.png"
    create_dashboard(df_flat, focus_city=focus_city, predictions=predictions,
                     output_path=output_path)

    return output_path, predictions, metrics


def main():
    print("=" * 60)
    print("  全国核心城市空气质量(AQI)多维统计分析与预测系统")
    print("=" * 60)

    # 第1步：数据采集
    print("\n" + "=" * 60)
    print("  [第1步] 数据采集")
    print("=" * 60)

    expected_files = 31
    data_files = []
    if os.path.exists("data"):
        data_files = [f for f in os.listdir("data")
                      if f.endswith(".csv") and f != "cleaned_aqi.csv"]

    if len(data_files) >= expected_files:
        print(f"  发现 {len(data_files)} 个城市CSV文件，跳过爬取")
        print("  (如需重新爬取，请删除 data/ 目录下的 .csv 文件)")
    else:
        print("  开始爬取数据...")
        for f in data_files:
            os.remove(os.path.join("data", f))
        import crawler
        crawler.crawl_all()

    # 第2步：城市选择 + 清洗 + 预测 + 可视化
    focus_city = choose_city()
    print(f"\n  焦点城市: {focus_city}")
    output_path, predictions, metrics = run_pipeline(focus_city)

    # 总结
    print("\n" + "=" * 60)
    print("  " + focus_city + " — 分析完成!")
    print("=" * 60)
    if predictions:
        print(f"  未来7天AQI预测: {predictions}")
    if metrics:
        print(f"  模型RMSE: {metrics['RMSE']:.2f}  "
              f"MAE: {metrics['MAE']:.2f}  "
              f"R²: {metrics['R2']:.4f}")
    print(f"  图表文件: {output_path}")
    print("=" * 60)

    # 第3步：是否继续查看其他城市
    while True:
        again = input("\n  是否切换其他城市？(y/n，直接回车=n): ").strip().lower()
        if not again or again == 'n':
            break
        focus_city = choose_city()
        print(f"\n  焦点城市: {focus_city}")
        output_path, predictions, metrics = run_pipeline(focus_city)
        print("\n" + "=" * 60)
        print("  " + focus_city + " — 分析完成!")
        print("=" * 60)
        if predictions:
            print(f"  未来7天AQI预测: {predictions}")
        if metrics:
            print(f"  模型RMSE: {metrics['RMSE']:.2f}  "
                  f"MAE: {metrics['MAE']:.2f}  "
                  f"R²: {metrics['R2']:.4f}")
        print(f"  图表文件: {output_path}")
        print("=" * 60)

    print("\n  感谢使用！\n")


# 程序入口
if __name__ == "__main__":
    main()
