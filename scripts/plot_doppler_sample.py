#!/usr/bin/env python
"""ドップラースペクトルのサンプル表示スクリプト。

process_iq.py で生成した .npz ファイルを読み込み、
先頭方位・先頭レンジゲートのドップラー速度スペクトルをプロットする。

Usage:
    uv run scripts/plot_doppler_sample.py <input_file.npz>
        [--azi-idx AZI_IDX] [--rng-idx RNG_IDX]
        [--output OUTPUT_FIG]
"""

import argparse
import signal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main() -> None:
    """メイン処理：npzファイルを読み込み、ドップラースペクトルを表示する。"""
    parser = argparse.ArgumentParser(
        description="NPZファイルからドップラースペクトルを表示・保存する。"
    )
    parser.add_argument("input_file", help="入力NPZファイルのパス")
    parser.add_argument(
        "--azi-idx",
        type=int,
        default=0,
        help="方位インデックス（デフォルト: 0）",
    )
    parser.add_argument(
        "--rng-idx",
        type=int,
        default=0,
        help="レンジインデックス（デフォルト: 0）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="出力画像のパス（省略時は表示のみ）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    data = np.load(input_path)
    doppler: np.ndarray = data["doppler"]
    velocity: np.ndarray = data["velocity"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(velocity, doppler[args.azi_idx, args.rng_idx, :])
    ax.set_xlabel("Doppler Velocity [m/s]")
    ax.set_ylabel("Power Spectrum [dB]")
    ax.set_title(f"{input_path.stem}  azi={args.azi_idx}  rng={args.rng_idx}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"保存完了: {output_path}")

    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
