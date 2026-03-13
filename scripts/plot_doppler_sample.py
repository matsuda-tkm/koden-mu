#!/usr/bin/env python
"""ドップラースペクトルのサンプル表示スクリプト。

process_iq.py で生成した .npz ファイルを読み込み、
先頭方位・先頭レンジゲートのドップラー速度スペクトルをプロットする。

Usage:
    uv run plot_doppler_sample.py <input_file.npz>
"""

import signal
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main() -> None:
    """メイン処理：npzファイルを読み込み、ドップラースペクトルを表示する。"""
    input_path = Path(sys.argv[1])
    data = np.load(input_path)
    doppler: np.ndarray = data["doppler"]
    velocity: np.ndarray = data["velocity"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(velocity, doppler[0, 0, :])
    ax.set_xlabel("Doppler Velocity [m/s]")
    ax.set_ylabel("Power Spectrum [dB]")
    plt.show()


if __name__ == "__main__":
    main()
