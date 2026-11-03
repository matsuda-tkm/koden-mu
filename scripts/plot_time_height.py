#!/usr/bin/env python
"""ディレクトリ内の .bin を一括処理し、時間・高度図を作成するスクリプト。

鉛直上向きで観測した IQ データを対象とする。各ファイルの全スウィープから
レンジごとの受信電力と平均ドップラー速度を求め、ファイルの更新時刻を横軸に
して並べる。上段に受信電力、下段に平均ドップラー速度を描いた PNG を出力する。

Usage:
    uv run scripts/plot_time_height.py <input_dir> --output OUTPUT_PNG
                                       [--npz OUTPUT_NPZ] [--max-range KM]
                                       [--clutter-bins N] [--snr-min DB]
                                       [--no-mark-short-files]
"""

import argparse
import datetime as dt
import signal
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from koden_mu.radar import (
    BLOCK_SIZE,
    HIT_NUM,
    RNG_NUM,
    RNG_STEP,
    compute_profile,
    read_sweeps,
)

signal.signal(signal.SIGINT, signal.SIG_DFL)

JST = dt.timezone(dt.timedelta(hours=9))


def list_files(input_dir: Path) -> list[Path]:
    """解析に使う .bin ファイルをファイル名順に返す。

    _0000.bin は駆動範囲のデータがそろっていない場合があるため除外する。
    1 方位分（HIT_NUM ヒット）に満たないファイルも除外する。
    """
    min_size = BLOCK_SIZE * 8 * HIT_NUM
    return sorted(
        p
        for p in input_dir.glob("*.bin")
        if not p.name.endswith("_0000.bin") and p.stat().st_size >= min_size
    )


def fix_timestamps(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """後から上書きされたファイルの時刻を、前後のファイルの時刻から補間する。

    上書きされたファイルは更新時刻が取得時刻と大きく異なるため、
    前後 10 ファイルの中央値から 120 秒以上ずれた時刻を補間の対象にする。

    Returns:
        (補間後の時刻, 補間したファイルを True とする配列)
    """
    idx = np.arange(len(times))
    med = np.array([np.median(times[max(0, i - 10) : i + 11]) for i in idx])
    bad = np.abs(times - med) > 120
    fixed = times.copy()
    if bad.any() and (~bad).any():
        fixed[bad] = np.interp(idx[bad], idx[~bad], times[~bad])
    return fixed, bad


def plot(
    times: list[dt.datetime],
    rng: np.ndarray,
    moving: np.ndarray,
    mean_velocity: np.ndarray,
    short_times: list[dt.datetime],
    args: argparse.Namespace,
) -> None:
    """受信電力と平均ドップラー速度の時間・高度図を上下に並べて保存する。"""
    nr = int(round(args.max_range / RNG_STEP))
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
    )
    vmin, vmax = np.nanpercentile(moving[:, :nr], [5, 99.5])
    pc = axes[0].pcolormesh(
        times,
        rng[:nr],
        moving[:, :nr].T,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )
    fig.colorbar(pc, ax=axes[0], label="Power [dBm]")
    axes[0].set_title(f"Received power (|v| <= {args.clutter_bins} bins removed)")

    pc = axes[1].pcolormesh(
        times,
        rng[:nr],
        mean_velocity[:, :nr].T,
        cmap="turbo",
        vmin=-2,
        vmax=10,
        shading="nearest",
    )
    fig.colorbar(pc, ax=axes[1], label="Mean Doppler velocity [m/s] (+: downward)")
    axes[1].set_title(f"Mean Doppler velocity (SNR >= {args.snr_min:g} dB)")
    axes[1].set_facecolor("0.85")

    for ax in axes:
        ax.set_ylabel("Range ≈ height above radar [km]")
        for t in short_times:
            ax.axvline(t, color="k", ls="--", lw=0.8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=JST))
    axes[1].xaxis.set_major_locator(
        mdates.MinuteLocator(byminute=range(0, 60, 10), tz=JST)
    )
    axes[1].set_xlabel("Time (JST)")
    fig.suptitle(
        f"{Path(args.input_dir).name}  {times[0]:%Y-%m-%d %H:%M} - {times[-1]:%H:%M} JST"
    )
    fig.savefig(args.output, dpi=150)
    print("saved", args.output)


def main() -> None:
    """メイン処理：ディレクトリ内の全 .bin を処理して時間・高度図を出力する。"""
    parser = argparse.ArgumentParser(
        description="鉛直観測の .bin ファイルを一括処理して時間・高度図を作成する。"
    )
    parser.add_argument("input_dir", help="入力 .bin ファイルが格納されたディレクトリ")
    parser.add_argument("--output", required=True, help="出力PNGのパス")
    parser.add_argument(
        "--npz", help="計算結果を保存するNPZのパス（省略時は保存しない）"
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=4.24,
        help="表示する最大距離 [km]（デフォルト: 4.24、短パルスの範囲）",
    )
    parser.add_argument(
        "--clutter-bins",
        type=int,
        default=2,
        help="静止物の反射として除外するゼロ速度付近のビン数（片側、デフォルト: 2）",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=-3.0,
        help="平均ドップラー速度を表示する信号対雑音比の下限 [dB]（デフォルト: -3）",
    )
    parser.add_argument(
        "--mark-short-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="サイズが通常より小さいファイルの位置に破線を引く（デフォルト: 引く）",
    )
    args = parser.parse_args()

    files = list_files(Path(args.input_dir))
    if not files:
        parser.error(f"解析できる .bin ファイルがありません: {args.input_dir}")

    names, mtimes, sizes, profiles = [], [], [], []
    for i, path in enumerate(files):
        iq = read_sweeps(str(path))
        if iq.shape[0] < HIT_NUM:
            continue
        profiles.append(compute_profile(iq, args.clutter_bins, args.snr_min))
        names.append(path.name)
        mtimes.append(path.stat().st_mtime)
        sizes.append(path.stat().st_size)
        if i % 100 == 0:
            print(f"{i}/{len(files)} {path.name}")

    times, fixed = fix_timestamps(np.array(mtimes))
    if fixed.any():
        print("timestamps interpolated:", ", ".join(np.array(names)[fixed]))
    sizes = np.array(sizes)
    # 記録が途中で途切れたファイルは、通常のファイルの半分程度のサイズになる
    short = sizes < np.median(sizes) * 0.75
    # 最後のファイルは記録の終了で小さくなるため、破線の対象から外す
    short[-1] = False
    rng = np.arange(RNG_NUM) * RNG_STEP
    result = {k: np.array([p[k] for p in profiles]) for k in profiles[0]}

    if args.npz:
        np.savez_compressed(
            args.npz,
            time=times,
            range=rng,
            file_name=np.array(names),
            short_file=short,
            **result,
        )
        print("saved", args.npz)

    jst_times = [dt.datetime.fromtimestamp(t, JST) for t in times]
    short_times = (
        [t for t, s in zip(jst_times, short) if s] if args.mark_short_files else []
    )
    plot(
        jst_times,
        rng,
        result["moving"],
        result["mean_velocity"],
        short_times,
        args,
    )


if __name__ == "__main__":
    main()
