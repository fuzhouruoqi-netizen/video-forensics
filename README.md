# video-forensics

一个轻量的**视频亮度取证分析**脚本：从逐帧亮度序列里找拼接/插入/剪切痕迹，并通过频谱判断是否存在周期性闪烁（工频干扰、重编码或合成）。

## 功能

1. **逐帧亮度曲线** —— 均值 / 中位数 / 标准差，输出 CSV + 三联图
2. **帧间亮度突变检测** —— 用 MAD（中位数绝对偏差）自适应阈值，比固定阈值更抗场景光照变化
3. **亮度闪烁频谱** —— 去趋势 + 汉宁窗 + FFT，找出 1 Hz 以上的主频峰值

## 依赖

```bash
pip install numpy opencv-python matplotlib
```

## 用法

```bash
# 分析当前目录下的 test.mp4
python analyze_brightness.py

# 分析指定视频
python analyze_brightness.py D:\path\to\video.mp4
```

输出两个文件（与输入视频同名）：

- `<name>_brightness.csv` —— 列：`frame,time_s,mean,median,std`
- `<name>_brightness.png` —— 亮度曲线 / 帧间差 / 频谱 三联图

## 输出示例

终端会打印分辨率、帧率、总帧数、亮度范围，以及检测到的突变点与闪烁主频：

```
视频: test.mp4
  分辨率 1920x1080  帧率 25.00  总帧数 750  采样 750 帧
  亮度均值范围: 41.3 ~ 96.7
  相邻帧亮度差: 中位 0.42, 阈值 3.15
  ⚠ 检测到 2 处突变（疑似拼接/插入/剪切）：
     第 312 帧  时间 12.44s  Δ=8.31
  闪烁主频（前 3）:
     50.00 Hz   幅度 12.44
```

## 实现说明

- `read_brightness()`：长视频会按 `max_samples` 自动降采样，避免跑太久
- `find_jumps()`：`threshold = max(median(Δ) + 6×MAD, 3.0)`
- `flicker_spectrum()`：减去 1 秒滑动平均去趋势，只保留 1 Hz 以上成分

## 仓库内容

| 文件 | 说明 |
|---|---|
| `analyze_brightness.py` | 主脚本 |
| `test.mp4` | 测试视频（约 1 MB） |
| `test_brightness.csv` | 示例输出：逐帧亮度表 |
| `test_brightness.png` | 示例输出：三联图 |

## License

仅供学习与检验技术研究使用。
