# -*- coding: utf-8 -*-
"""
视频亮度取证分析
----------------
输入一个视频，输出三样东西：
  1. 逐帧亮度曲线（均值 / 中位数 / 标准差）
  2. 相邻帧亮度突变点（疑似拼接、剪切、插入）
  3. 亮度序列的频谱（找工频闪烁，判断是否经过重编码/合成）

用法：
    python analyze_brightness.py 视频路径.mp4
不带参数时默认分析当前目录下的 test.mp4
"""

import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")            # 不弹窗，直接存图；想看窗口就注释掉这行
import matplotlib.pyplot as plt


def read_brightness(path, max_samples=3000):
    """逐帧读取，返回 (亮度均值, 亮度中位数, 亮度标准差, fps, 分辨率)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 太长的视频每隔 step 帧取一帧，避免跑太久
    step = max(1, total // max_samples)

    means, medians, stds = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(gray.mean()))
            medians.append(float(np.median(gray)))
            stds.append(float(gray.std()))
        i += 1
    cap.release()

    return (np.array(means), np.array(medians), np.array(stds),
            fps / step, (width, height), total)


def find_jumps(series, k=6.0, min_delta=3.0):
    """用中位数绝对偏差(MAD)找异常突变，比固定阈值更抗干扰"""
    diff = np.abs(np.diff(series))
    if len(diff) == 0:
        return np.array([], dtype=int), diff, 0.0
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) * 1.4826
    threshold = max(med + k * mad, min_delta)
    idx = np.where(diff > threshold)[0]
    return idx, diff, threshold


def _moving_average(x, w):
    """奇数窗口的滑动平均，用于去掉缓慢的整体亮度漂移"""
    w = max(3, int(w) | 1)
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


def flicker_spectrum(series, fps, top_n=3):
    """先去趋势，再对亮度序列做 FFT，找周期性闪烁的主频"""
    if len(series) < 32:
        return np.array([]), np.array([]), []

    # ① 去趋势：减掉 1 秒窗的滑动平均，只留下比 1Hz 快的成分
    x = series - _moving_average(series, max(5, int(round(fps))))
    x = x - x.mean()

    # ② 加汉宁窗抑制频谱泄漏
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)

    # ③ 丢掉直流和过低频
    mask = freqs > 1.0
    freqs, spec = freqs[mask], spec[mask]
    if len(spec) == 0:
        return freqs, spec, []

    order = np.argsort(spec)[::-1][:top_n]
    peaks = sorted(zip(freqs[order], spec[order]), key=lambda t: -t[1])
    return freqs, spec, peaks


def analyze(path, out_dir="."):
    means, medians, stds, fps, size, total = read_brightness(path)
    if len(means) == 0:
        raise SystemExit("没读到任何帧")

    t = np.arange(len(means)) / fps
    jumps, diff, thr = find_jumps(means)
    freqs, spec, peaks = flicker_spectrum(means, fps)

    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(out_dir, f"{base}_brightness.csv")
    png_path = os.path.join(out_dir, f"{base}_brightness.png")

    np.savetxt(csv_path,
               np.column_stack([np.arange(len(means)), t, means, medians, stds]),
               delimiter=",", fmt="%.4f",
               header="frame,time_s,mean,median,std", comments="")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    # 1) 亮度曲线
    ax = axes[0]
    ax.plot(t, means, lw=1.0, color="#2b7de9", label="frame mean luminance")
    ax.plot(t, medians, lw=0.8, color="#8fb8f0", alpha=0.7, label="median")
    if len(jumps):
        ax.scatter(t[jumps + 1], means[jumps + 1], color="#e5484d", s=28, zorder=5,
                   label=f"jump ({len(jumps)})")
    ax.set_ylabel("brightness (0-255)")
    ax.set_title(f"{base}   fps={fps:.2f}   size={size[0]}x{size[1]}   frames={total}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # 2) 帧间差
    ax = axes[1]
    ax.plot(t[1:], diff, lw=0.9, color="#7a5af8", label="|Δ mean|")
    ax.axhline(thr, color="#e5484d", ls="--", lw=1, label=f"threshold={thr:.2f}")
    if len(jumps):
        ax.scatter(t[jumps + 1], diff[jumps], color="#e5484d", s=28, zorder=5)
    ax.set_ylabel("frame-to-frame Δ")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # 3) 频谱
    ax = axes[2]
    if len(freqs):
        ax.plot(freqs, spec, lw=0.9, color="#12a594")
        for f, a in peaks[:3]:
            ax.annotate(f"{f:.1f} Hz", xy=(f, a), xytext=(f + max(1, freqs[-1] * 0.02), a),
                        fontsize=8, color="#0b6d61")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.set_title("brightness flicker spectrum")
    ax.grid(alpha=0.25)

    fig.savefig(png_path, dpi=140)

    # -------- 终端摘要 --------
    print(f"视频: {os.path.basename(path)}")
    print(f"  分辨率 {size[0]}x{size[1]}  帧率 {fps:.2f}  总帧数 {total}  采样 {len(means)} 帧")
    print(f"  亮度均值范围: {means.min():.1f} ~ {means.max():.1f}")
    print(f"  相邻帧亮度差: 中位 {np.median(diff):.2f}, 阈值 {thr:.2f}")
    if len(jumps):
        print(f"  ⚠ 检测到 {len(jumps)} 处突变（疑似拼接/插入/剪切）：")
        for j in jumps[:10]:
            print(f"     第 {j + 1} 帧  时间 {t[j + 1]:.2f}s  Δ={diff[j]:.2f}")
    else:
        print("  未发现明显的亮度突变")
    if peaks:
        print("  闪烁主频（前 3）:")
        for f, a in peaks[:3]:
            print(f"     {f:6.2f} Hz   幅度 {a:.2f}")
    print(f"\n已保存：\n  图  {os.path.abspath(png_path)}\n  表  {os.path.abspath(csv_path)}")
    return {"means": means, "jumps": jumps, "threshold": thr, "peaks": peaks}


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    analyze(video)
