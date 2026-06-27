import os
import sys
import threading
import time
from datetime import date, timedelta
from tkinter import Tk, Frame, Label, Button, StringVar, OptionMenu, PhotoImage, Canvas, Scrollbar, DISABLED, NORMAL, W, E, NW

import pandas as pd

# 常量

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

DATA_DIR = "data"
OUTPUT_DIR = "output"

# 工具函数

def check_data_files():
    """检查 data/ 目录中是否有 >=31 个城市 CSV"""
    if not os.path.exists(DATA_DIR):
        return False
    files = [f for f in os.listdir(DATA_DIR)
             if f.endswith(".csv") and f != "cleaned_aqi.csv"]
    return len(files) >= 31


def has_chart(city):
    """该城市图表是否已存在"""
    path = os.path.join(OUTPUT_DIR, f"{city}.png")
    return os.path.exists(path) and os.path.getsize(path) > 0


def get_indicators(city):
    """从清洗数据中提取指定城市的摘要指标"""
    csv_path = os.path.join(DATA_DIR, "cleaned_aqi.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    city_df = df[df["城市"] == city]
    if len(city_df) == 0:
        return None
    avg_aqi = round(float(city_df["AQI"].mean()), 1)
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    primary = max(pollutants, key=lambda p: city_df[p].mean())
    return {"avg_aqi": avg_aqi, "primary": primary}


# GUI 主窗口

class AQIApp:
    def __init__(self, root):
        self.root = root
        root.title("全国核心城市 AQI 多维分析与预测系统")
        root.geometry("1100x650")
        root.configure(bg="#ecf0f1")

        self.focus_city = StringVar(value="北京")

        # 顶部栏
        top = Frame(root, bg="#2c3e50", padx=12, pady=8)
        top.pack(fill="x")

        Label(top, text="全国核心城市 AQI 多维分析系统", fg="#ecf0f1", bg="#2c3e50",
              font=("Microsoft YaHei", 16, "bold")).pack(side="left", padx=(0, 24))

        Label(top, text="焦点城市:", fg="#bdc3c7", bg="#2c3e50",
              font=("Microsoft YaHei", 11)).pack(side="left")

        city_menu = OptionMenu(top, self.focus_city, "北京", *ALL_CITIES,
                               command=self.on_city_change)
        city_menu.config(bg="#34495e", fg="#ecf0f1", font=("Microsoft YaHei", 11))
        city_menu.pack(side="left", padx=8)

        self.btn_refresh = Button(top, text="刷新数据", command=self.on_refresh,
                                  bg="#e67e22", fg="#fff", font=("Microsoft YaHei", 10),
                                  relief="flat", padx=14, pady=2)
        self.btn_refresh.pack(side="left", padx=12)

        # 主体区
        body = Frame(root, bg="#ecf0f1")
        body.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # 左侧：图表区域
        left = Frame(body, bg="#fff", relief="flat", bd=1)
        left.pack(side="left", fill="both", expand=True)
        Label(left, text="分析图表", bg="#fff", fg="#7f8c8d",
              font=("Microsoft YaHei", 10)).pack(pady=4)

        self.canvas = Canvas(left, bg="#fafafa", highlightthickness=0)
        self.scrollbar = Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.chart_label = Label(self.canvas, bg="#fafafa")
        self.canvas.create_window((0, 0), window=self.chart_label, anchor=NW)

        # 右侧：数据面板
        right = Frame(body, bg="#fff", relief="flat", bd=1, width=280)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        Label(right, text="数据摘要", bg="#fff", fg="#2c3e50",
              font=("Microsoft YaHei", 12, "bold")).pack(pady=8)

        self.lbl_aqi = self._make_card(right, "AQI 均值", "--")
        self.lbl_primary = self._make_card(right, "首要污染物", "--")
        self.lbl_rmse = self._make_card(right, "模型 RMSE", "--")
        self.lbl_mae = self._make_card(right, "模型 MAE", "--")
        self.lbl_r2 = self._make_card(right, "模型 R²", "--")

        Label(right, text="未来 7 天 AQI 预测", bg="#fff", fg="#2c3e50",
              font=("Microsoft YaHei", 11, "bold")).pack(pady=(16, 4))
        self.pred_frame = Frame(right, bg="#fff")
        self.pred_frame.pack(fill="x", padx=10)
        self.pred_labels = []
        for i in range(7):
            lbl = Label(self.pred_frame, text=f"  D+{i+1}: --", bg="#fff", fg="#7f8c8d",
                        font=("Consolas", 11), anchor=W)
            lbl.pack(fill="x")
            self.pred_labels.append(lbl)

        # 状态栏
        self.status = StringVar(value="就绪")
        status_bar = Label(root, textvariable=self.status, bg="#34495e", fg="#bdc3c7",
                           font=("Microsoft YaHei", 9), anchor=W, padx=10, pady=3)
        status_bar.pack(fill="x", side="bottom")

    # 辅助方法

    def _make_card(self, parent, label, value):
        frame = Frame(parent, bg="#ecf0f1", padx=10, pady=6)
        frame.pack(fill="x", padx=10, pady=3)
        Label(frame, text=label, bg="#ecf0f1", fg="#7f8c8d",
              font=("Microsoft YaHei", 9)).pack(anchor=W)
        val = Label(frame, text=value, bg="#ecf0f1", fg="#2c3e50",
                    font=("Microsoft YaHei", 16, "bold"))
        val.pack(anchor=W)
        return val

    def _run_bg(self, func, done_msg="就绪"):
        """在后台线程运行耗时操作"""
        def wrapper():
            self.root.after(0, lambda: self.status.set("正在运行..."))
            self.root.after(0, lambda: self.btn_refresh.config(state=DISABLED))
            func()
            self.root.after(0, lambda: self.status.set(done_msg))
            self.root.after(0, lambda: self.btn_refresh.config(state=NORMAL))
        threading.Thread(target=wrapper, daemon=True).start()

    # 核心逻辑

    def load_city(self):
        """加载当前选中城市的图表与数据"""
        city = self.focus_city.get()
        self.show_chart(city)

        # 后台计算指标
        def work():
            from ml_predict import run_prediction
            from data_cleaner import run_cleaning

            self.root.after(0, lambda: self.status.set(f"正在分析 {city}..."))

            df_multi, _ = run_cleaning(DATA_DIR)
            preds, metrics = run_prediction(df_multi, focus_city=city)

            indicators = get_indicators(city) or {}
            avg_aqi = indicators.get("avg_aqi", "--")
            primary = indicators.get("primary", "--")

            def update_ui():
                self.lbl_aqi.config(text=str(avg_aqi))
                self.lbl_primary.config(text=primary)
                self.lbl_rmse.config(text=f"{metrics['RMSE']:.2f}" if metrics else "--")
                self.lbl_mae.config(text=f"{metrics['MAE']:.2f}" if metrics else "--")
                self.lbl_r2.config(text=f"{metrics['R2']:.4f}" if metrics else "--")
                if preds:
                    for i, v in enumerate(preds):
                        self.pred_labels[i].config(text=f"  D+{i+1}: {v}")
                self.root.after(0, lambda: self.status.set("就绪"))
            self.root.after(0, update_ui)

        self._run_bg(work)

    def show_chart(self, city):
        """在画布中显示图表 PNG"""
        path = os.path.join(OUTPUT_DIR, f"{city}.png")
        if not os.path.exists(path):
            self.chart_label.config(image="", text=f"  图表尚未生成，正在后台生成...\n  城市: {city}")
            return
        self.img = PhotoImage(file=path)
        self.chart_label.config(image=self.img, text="")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def on_city_change(self, city):
        """城市下拉切换时触发"""
        if not has_chart(city):
            self.status.set(f"正在生成 {city} 的图表...")
            self._run_bg(lambda: self._generate_chart(city), "就绪")
        else:
            self.load_city()

    def _generate_chart(self, city):
        """生成指定城市的图表（后台线程）"""
        try:
            from main import run_pipeline
            run_pipeline(city)
            self.root.after(0, lambda: self.show_chart(city))
            self.root.after(0, self.load_city)
        except Exception as e:
            self.root.after(0, lambda: self.status.set(f"生成失败: {e}"))

    def on_refresh(self):
        """点击刷新按钮"""
        def work():
            import crawler
            self.root.after(0, lambda: self.status.set("清空旧数据..."))
            if os.path.exists(DATA_DIR):
                for f in os.listdir(DATA_DIR):
                    if f.endswith(".csv"):
                        os.remove(os.path.join(DATA_DIR, f))
            if os.path.exists(OUTPUT_DIR):
                for f in os.listdir(OUTPUT_DIR):
                    if f.endswith(".png"):
                        os.remove(os.path.join(OUTPUT_DIR, f))

            self.root.after(0, lambda: self.status.set("重新爬取数据..."))
            crawler.crawl_all()

            city = self.focus_city.get()
            self._generate_chart(city)

        self._run_bg(work, "就绪")


# 启动入口

def main():
    # 自动检查数据，缺失则爬取
    if not check_data_files():
        print("正在爬取数据，请稍候...")
        import crawler
        crawler.crawl_all()

    root = Tk()
    app = AQIApp(root)
    # 启动后自动加载默认城市
    app.load_city()
    root.mainloop()


if __name__ == "__main__":
    main()
