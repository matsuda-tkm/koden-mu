#!/usr/bin/env python
"""IQデータの読み込み・処理・可視化スクリプト。

バイナリ形式のIQデータファイルを読み込み、ドップラー処理および
PPI（Plan Position Indicator）形式での電力マップを生成・保存する。

Usage:
    uv run process_iq.py <input_file> [--output-npz OUTPUT_NPZ] [--output-fig OUTPUT_FIG]
"""

import argparse
import signal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from koden_mu.radar import (
    compute_doppler,
    coordinate,
    doppler_velocity,
    make_colormap,
    read_iq,
    to_dbm,
)

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main() -> None:
    """メイン処理：IQデータを読み込み、処理・保存・描画を行う。"""
    parser = argparse.ArgumentParser(
        description="IQデータを処理してPPI画像とNPZを出力する。"
    )
    parser.add_argument("input_file", help="入力バイナリIQデータファイルのパス")
    parser.add_argument(
        "--output-npz",
        default="data_processed",
        help="NPZ出力先ディレクトリ（デフォルト: data_processed/）",
    )
    parser.add_argument(
        "--output-fig",
        default="fig",
        help="PNG出力先ディレクトリ（デフォルト: fig/）",
    )
    args = parser.parse_args()

    azi_num = 10
    input_path = Path(args.input_file)
    npz_dir = Path(args.output_npz)
    fig_dir = Path(args.output_fig)
    npz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_img = fig_dir / input_path.with_suffix(".png").name
    output_dat = npz_dir / input_path.with_suffix(".npz").name

    iq = read_iq(str(input_path), azi_num)
    doppler = compute_doppler(iq)
    velocity = doppler_velocity()
    dbm = to_dbm(iq)
    x, y = coordinate(azi_num)

    np.savez_compressed(
        output_dat, doppler=doppler, velocity=velocity, power=dbm, x=x, y=y
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    p = ax.pcolormesh(x, y, dbm, vmin=-130, vmax=-50.0, cmap=make_colormap())
    cbar = fig.colorbar(p, ax=ax, orientation="vertical")
    cbar.set_label("Power [dBm]", fontname="Arial", fontsize=10)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(0.0, 20.0)
    ax.set_aspect("equal", "datalim")
    plt.savefig(output_img, dpi=300)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
