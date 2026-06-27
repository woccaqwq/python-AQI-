"""
模块1：全国31个核心城市空气质量数据爬虫

【功能说明】
  本模块负责从"天气后报网"(tianqihoubao.com) 定向抓取全国31个核心城市的
  逐日空气质量数据（AQI、PM2.5、PM10、SO2、NO2、CO、O3）及质量等级。

【数据源】
  天气后报网是一个气象历史数据镜像站点，页面结构为纯HTML表格，无需处理
  JavaScript，因此使用requests+BeautifulSoup即可稳定采集。

【依赖库】
  requests  - HTTP请求
  BeautifulSoup (bs4) - HTML解析
  pandas    - 数据整理与CSV导出
  dateutil  - 月份递增计算
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import date, timedelta


# 全局配置

# 31个省会、直辖城市：中文名：拼音（用于构造URL）
CITIES = {
    "北京": "beijing", "上海": "shanghai", "天津": "tianjin", "重庆": "chongqing",
    "哈尔滨": "haerbin", "长春": "changchun", "沈阳": "shenyang", "呼和浩特": "huhehaote",
    "石家庄": "shijiazhuang", "太原": "taiyuan", "济南": "jinan", "郑州": "zhengzhou",
    "南京": "nanjing", "合肥": "hefei", "武汉": "wuhan", "长沙": "changsha",
    "南昌": "nanchang", "杭州": "hangzhou", "福州": "fuzhou", "南宁": "nanning",
    "广州": "guangzhou", "海口": "haikou", "贵阳": "guiyang", "昆明": "kunming",
    "成都": "chengdu", "拉萨": "lasa", "西宁": "xining", "兰州": "lanzhou",
    "银川": "yinchuan", "西安": "xian", "乌鲁木齐": "wulumuqi",
}

# HTTP请求头：伪装成Chrome浏览器，避免被简单反爬拦截
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 目标网站的URL模板
# {pinyin}: 城市拼音  {year}: 4位年份  {month:02d}: 2位月份（如01, 12）
BASE_URL = "http://www.tianqihoubao.com/aqi/{pinyin}-{year}{month:02d}.html"

DATA_DIR = "data"             # 数据保存目录
REQUEST_INTERVAL = 2          # 同一城市不同月份间的请求间隔（秒）防止被封禁
MAX_RETRIES = 3               # 单次请求失败后的最大重试次数
RETRY_DELAY = 5               # 重试前等待秒数


# 核心函数

def fetch_city_month(city_cn, pinyin, year, month):
    """
    解析网站
    网页表格结构:
        日期 | AQI | 质量等级 | PM2.5 | PM10 | SO2 | NO2 | CO | O3
    """
    url = BASE_URL.format(pinyin=pinyin, year=year, month=month)

    # 第1步：HTTP请求（最多重试3次）
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            # 让requests自动检测响应编码，避免强制gbk导致的乱码
            resp.encoding = resp.apparent_encoding
            if resp.status_code == 200:
                break  # 请求成功，跳出重试循环
            else:
                print(f"  [失败] {city_cn} {year}-{month:02d} HTTP {resp.status_code} "
                      f"(第{attempt}次尝试)")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            # 常见异常：网络超时、连接被重置等
            print(f"  [错误] {city_cn} {year}-{month:02d}: {e} "
                  f"(第{attempt}次尝试)")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    else:
        # for-else语法：循环未通过break退出时执行
        print(f"  [放弃] {city_cn} {year}-{month:02d} 重试{MAX_RETRIES}次后仍失败")
        return None

    # 第2步：解析HTML表格
    soup = BeautifulSoup(resp.text, "lxml")     #指定HTML解析器的参数，速度更快兼容更好
    table = soup.select_one("table.b")  # CSS选择器：class="b"的table元素
    if table is None:
        print(f"  [无数据] {city_cn} {year}-{month:02d}")
        return None

    # 第3步：逐行提取数据
    rows = table.select("tr")[1:]      # 跳过第一行表头
    records = []
    for row in rows:
        cells = row.select("td")       # 获取当前行所有单元格
        if len(cells) < 10:            # 有效数据行至少10列
            continue
        try:
            date_str = cells[0].get_text(strip=True)  # strip: 去除首尾空白
            records.append({
                "日期": date_str,
                "质量等级": cells[2].get_text(strip=True),  # 第3列=质量等级（优/良/轻度污染等）
                "AQI": cells[1].get_text(strip=True),      # 第2列=AQI数值
                "PM2.5": cells[3].get_text(strip=True),
                "PM10": cells[4].get_text(strip=True),
                "SO2": cells[5].get_text(strip=True),
                "NO2": cells[6].get_text(strip=True),
                "CO": cells[7].get_text(strip=True),
                "O3": cells[8].get_text(strip=True),
            })
        except Exception:
            continue  # 单行解析失败不影响其他行

    if records:
        df = pd.DataFrame(records)
        df["城市"] = city_cn       # 添加城市标签
        print(f"  [成功] {city_cn} {year}-{month:02d} → {len(df)}条")
        return df
    return None


def crawl_city(city_cn, pinyin):
    """抓取单个城市从2025年1月到昨天（但会受限于网站本身所提供的数据可能有所偏差）的所有数据"""
    all_data = []
    for year in range(2025, date.today().year + 1):
        end_month = date.today().month if year == date.today().year else 12
        for month in range(1, end_month + 1):
            df = fetch_city_month(city_cn, pinyin, year, month)
            if df is not None:
                all_data.append(df)
            time.sleep(REQUEST_INTERVAL)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def crawl_all():
    """抓取所有31个城市数据"""
    end_date = date.today() - timedelta(days=1)
    total = len(CITIES)
    os.makedirs(DATA_DIR, exist_ok=True)      # 创建data目录，已存在则确认跳过
    print(f"  数据范围: 2025-01 ~ {end_date}")
    for i, (city_cn, pinyin) in enumerate(CITIES.items(), 1):
        print(f"[{i}/{total}] 正在抓取: {city_cn} ({pinyin})")
        df = crawl_city(city_cn, pinyin)
        if df is not None and len(df) > 0:
            path = os.path.join(DATA_DIR, f"{city_cn}.csv")     # 设置路径data/城市.csv
            df.to_csv(path, index=False, encoding="utf-8-sig")  # 向csv填充数据
            print(f"  已保存: {path} ({len(df)}条记录)")
        else:
            print(f"  {city_cn}: 无数据")
        if i < total:
            time.sleep(2)


# 独立运行入口
if __name__ == "__main__":
    crawl_all()
