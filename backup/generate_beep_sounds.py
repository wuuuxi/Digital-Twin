import wave
import struct
import math
import os


def generate_beep(filename, frequency, duration=0.15, volume=0.3, sample_rate=44100):
    """
    生成纯音beep音效 - 修复杂音版本

    Parameters:
    -----------
    filename : str
        输出文件名
    frequency : int
        频率 (Hz)
    duration : float
        时长 (秒)
    volume : float
        音量 (0.0-1.0)
    sample_rate : int
        采样率
    """
    num_samples = int(sample_rate * duration)

    # 生成正弦波数据
    samples = []

    # 计算淡入淡出长度（更长的淡出以减少杂音）
    fade_in_samples = int(sample_rate * 0.02)  # 20ms淡入
    fade_out_samples = int(sample_rate * 0.03)  # 30ms淡出

    for i in range(num_samples):
        t = i / sample_rate

        # 改进的包络：更平滑的淡入淡出
        if i < fade_in_samples:
            # 淡入：使用正弦曲线更平滑
            envelope = math.sin((i / fade_in_samples) * (math.pi / 2))
        elif i > num_samples - fade_out_samples:
            # 淡出：使用余弦曲线更平滑
            envelope = math.cos(((i - (num_samples - fade_out_samples)) / fade_out_samples) * (math.pi / 2))
        else:
            envelope = 1.0

        # 正弦波 - 确保在零交叉点结束
        cycles = frequency * t
        # 计算相位，确保在完整周期结束
        phase = 2 * math.pi * (cycles % 1.0)
        value = math.sin(phase)

        # 应用音量和包络
        value = value * volume * envelope

        # 转换为16位整数
        sample = int(value * 32767)
        samples.append(sample)

    # 确保最后几个样本为零，避免截断杂音
    for i in range(min(10, len(samples))):
        samples[-(i + 1)] = 0

    # 写入WAV文件
    with wave.open(filename, 'w') as wav_file:
        # 设置参数：通道数, 采样宽度(字节), 采样率, 帧数, 压缩类型, 压缩名称
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))

        # 写入数据
        for sample in samples:
            wav_file.writeframes(struct.pack('<h', sample))

    print(f"  Generated: {filename} ({frequency}Hz, {duration}s)")


def generate_smooth_beep(filename, frequency, duration=0.15, volume=0.3, sample_rate=44100):
    """
    生成更平滑的beep音效 - 专门针对杂音问题优化
    """
    num_samples = int(sample_rate * duration)

    # 计算确保在零交叉点结束的样本数
    samples_per_cycle = sample_rate / frequency
    # 调整样本数以确保在零交叉点结束
    adjusted_samples = int((num_samples // samples_per_cycle) * samples_per_cycle)
    if adjusted_samples < num_samples * 0.9:  # 如果调整后长度变化太大，使用原长度
        adjusted_samples = num_samples

    samples = []

    # 更长的淡入淡出
    fade_in_samples = int(sample_rate * 0.03)  # 30ms淡入
    fade_out_samples = int(sample_rate * 0.04)  # 40ms淡出

    for i in range(adjusted_samples):
        t = i / sample_rate

        # 非常平滑的包络
        if i < fade_in_samples:
            # 二次曲线淡入
            progress = i / fade_in_samples
            envelope = progress * progress
        elif i > adjusted_samples - fade_out_samples:
            # 二次曲线淡出
            progress = (adjusted_samples - i) / fade_out_samples
            envelope = progress * progress
        else:
            envelope = 1.0

        # 正弦波
        value = math.sin(2 * math.pi * frequency * t)

        # 应用音量和包络
        value = value * volume * envelope

        # 转换为16位整数
        sample = int(value * 32767)
        samples.append(sample)

    # 强制最后20个样本为零
    for i in range(min(20, len(samples))):
        samples[-(i + 1)] = 0

    # 写入WAV文件
    with wave.open(filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(samples), 'NONE', 'not compressed'))
        for sample in samples:
            wav_file.writeframes(struct.pack('<h', sample))

    print(f"  Generated smooth: {filename} ({frequency}Hz, {len(samples) / sample_rate:.2f}s)")


def main(output_dir):
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("=" * 60)
    print("Generating Improved Beep Sound Files (No Pop/Click)")
    print("=" * 60)

    # 1. 使用改进版本生成音效
    generate_smooth_beep(
        os.path.join(output_dir, 'beep_high.wav'),
        frequency=1200,  # 举起阶段 - 高音
        duration=0.15,
        volume=0.3
    )

    generate_smooth_beep(
        os.path.join(output_dir, 'beep_low.wav'),
        frequency=800,  # 放下阶段 - 低音
        duration=0.15,
        volume=0.3
    )

    # 2. 峰谷值音效（稍长一些，更明显）
    generate_smooth_beep(
        os.path.join(output_dir, 'beep_peak.wav'),
        frequency=1500,  # 最高点 - 更高音
        duration=0.2,
        volume=0.35
    )

    generate_smooth_beep(
        os.path.join(output_dir, 'beep_valley.wav'),
        frequency=600,  # 最低点 - 更低音
        duration=0.2,
        volume=0.35
    )

    print("=" * 60)
    print("All sound files generated successfully!")
    print("Improvements applied:")
    print("  - Longer fade-in/fade-out (30ms/40ms)")
    print("  - Quadratic envelope curves for smoother transitions")
    print("  - Zero-crossing optimization")
    print("  - Forced zero samples at end to eliminate pops")
    print("=" * 60)

    print("\nFiles created:")
    for filename in ['beep_high.wav', 'beep_low.wav', 'beep_peak.wav', 'beep_valley.wav']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"    {filename} ({size} bytes)")
        else:
            print(f"    {filename} (not found)")


if __name__ == "__main__":
    # 设置输出目录
    output_dir = 'workspace'
    main(output_dir)
