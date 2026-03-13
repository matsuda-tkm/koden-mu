#!/usr/bin/env python
"""ディレクトリ内の.binファイルを一括処理して電力マップGIFを生成するスクリプト。

指定ディレクトリ内の全 .bin ファイルを昇順に読み込み、各ファイルの PPI
電力マップをフレームとして合成し、GIFアニメーションとして出力する。

Usage:
    uv run scripts/make_power_gif.py <input_dir> [--output OUTPUT_GIF]
                                     [--fps FPS] [--azi-num AZI_NUM]
                                     [--vmin VMIN] [--vmax VMAX]
                                     [--skip-first]
"""

import argparse
import signal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from koden_mu.radar import coordinate, make_colormap, read_iq, to_dbm
from koden_mu.radar import RNG_NUM

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main() -> None:
    """メイン処理：ディレクトリ内の全.binを処理してGIFを出力する。"""
    parser = argparse.ArgumentParser(
        description="ディレクトリ内の .bin ファイルを処理して電力マップGIFを生成する。"
    )
    parser.add_argument("input_dir", help="入力 .bin ファイルが格納されたディレクトリ")
    parser.add_argument(
        "--output",
        default=None,
        help="出力GIFのパス（省略時は <input_dir>/<dirname>.gif）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="GIFのフレームレート（デフォルト: 2.0 fps）",
    )
    parser.add_argument(
        "--azi-num",
        type=int,
        default=10,
        help="処理する方位数（デフォルト: 10）",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-130.0,
        help="カラースケールの最小値 [dBm]（デフォルト: -130）",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=-50.0,
        help="カラースケールの最大値 [dBm]（デフォルト: -50）",
    )
    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="最初のファイル（_0000.bin）をスキップする",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"エラー: ディレクトリが存在しません: {input_dir}")

    bin_files = sorted(p for p in input_dir.glob("*.bin") if p.is_file())
    if not bin_files:
        raise SystemExit(f"エラー: .bin ファイルが見つかりません: {input_dir}")

    if args.skip_first:
        bin_files = bin_files[1:]
        if not bin_files:
            raise SystemExit("エラー: スキップ後に処理対象のファイルがありません。")

    output_path = (
        Path(args.output)
        if args.output
        else input_dir / f"{input_dir.resolve().name}.gif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"処理対象: {len(bin_files)} ファイル -> {output_path}")

    cmap = make_colormap()
    x, y = coordinate(args.azi_num)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(0.0, 20.0)
    ax.set_aspect("equal", "datalim")

    dummy = np.full((args.azi_num, RNG_NUM), args.vmin)
    mesh = ax.pcolormesh(x, y, dummy, vmin=args.vmin, vmax=args.vmax, cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical")
    cbar.set_label("Power [dBm]", fontname="Arial", fontsize=10)
    title = ax.set_title("")

    def update(frame_idx: int) -> tuple:
        bin_path = bin_files[frame_idx]
        print(f"  [{frame_idx + 1}/{len(bin_files)}] {bin_path.name}")
        iq = read_iq(str(bin_path), args.azi_num)
        dbm = to_dbm(iq)
        mesh.set_array(dbm.ravel())
        title.set_text(bin_path.stem)
        return mesh, title

    ani = FuncAnimation(
        fig,
        update,
        frames=len(bin_files),
        blit=False,
        repeat=False,
    )

    interval_ms = int(1000 / args.fps)
    writer = PillowWriter(fps=args.fps)
    ani.save(str(output_path), writer=writer, dpi=150)
    plt.close(fig)

    print(f"GIF保存完了: {output_path}  (interval={interval_ms}ms/frame)")


if __name__ == "__main__":
    main()
