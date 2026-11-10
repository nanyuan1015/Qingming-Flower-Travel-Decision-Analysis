[README.md](https://github.com/user-attachments/files/34607648/README.md)
# 清明赏花旅游决策分析

本项目围绕清明假期赏花出行，整理多城市历史天气记录与花期数据，构建可复现的数据分析流程，覆盖天气特征汇总、花期趋势基线预测、历史花期重叠分析和可视化输出。分析主题对应清明天气、花期、出行决策与赏花经济四类问题；当前实现重点落在天气数据整理、花期探索分析和观赏窗口计算。

## 项目内容

- 汇总西安、杭州、武汉、郑州、毕节和吐鲁番六个有原始天气文件城市的清明时段记录，比较气温、湿度和天气现象代码。
- 整理五类花卉的历史始花日期与持续时间，使用按年份滚动验证评估线性趋势基线，并生成探索性2026趋势预测。
- 计算历史花期与清明假期的重叠时长，作为出行日期与观赏窗口分析的基础。
- 输出CSV结果和图表，便于复查、扩展和用于后续路线决策建模。

## 方法与技术

Python、Pandas、NumPy、scikit-learn、Matplotlib、Seaborn。分析包括Excel数据读取、日期清洗、描述性统计、按时间顺序的滚动验证、线性趋势基线和结果可视化。

## 目录结构

```text
.
├── data/
│   ├── weather/       # 多城市历史天气记录
│   └── phenology/     # 花期历史数据
├── src/
│   └── analyze_project.py
├── results/
│   ├── figures/       # 可视化图表
│   └── *.csv          # 整理数据、验证结果和分析汇总
├── requirements.txt
└── README.md
```

## 运行

建议 Python 3.10 或更新版本：

```bash
python -m pip install -r requirements.txt
python src/analyze_project.py
```

脚本从项目内 `data/` 读取数据，并在 `results/` 输出CSV、PNG图表和运行摘要。


