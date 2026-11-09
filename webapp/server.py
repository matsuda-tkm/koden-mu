#!/usr/bin/env python
"""IQ データの解析結果をブラウザで表示する Web サーバー。

起動時に指定した親フォルダの直下（と親フォルダ自身）から .bin ファイルを含む
フォルダを探し、画面で選んだフォルダの時間・高度図、高度プロファイル、
ドップラースペクトル、PPI 電力マップを表示する。記録中のフォルダでは、
取り込みが完了したファイルを順に処理して図に追加する。

Usage:
    uv run webapp/server.py [--root DIR] [--port 8000]
"""

import argparse
import datetime as dt
import gzip
import json
import os
import signal
import threading
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly

from koden_mu.radar import (
    AZI_STEP,
    HIT_NUM,
    PRF,
    RADAR_FREQ,
    RNG_NUM,
    RNG_STEP,
    SPEED_OF_LIGHT,
    OBS_DELAY_RATE,
    estimate_observation_time,
    fix_timestamps,
    mean_spectrum,
    read_sweeps,
)
from watcher import FILE_RE, Watcher

signal.signal(signal.SIGINT, signal.SIG_DFL)

JST = dt.timezone(dt.timedelta(hours=9))
STATIC_DIR = Path(__file__).parent / "static"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
MAX_COLUMNS = 3000  # 時間・高度図に送る列数の上限。超えた場合は間引く
# 1 ファイルの解析は一時的に約 100 MB を使うため、同時に 1 つだけ実行する
ANALYZE_LOCK = threading.Lock()


def to_json_array(a: np.ndarray, decimals: int = 2) -> list:
    """NaN を None に置き換えた入れ子のリストに変換する（JSON の null になる）。"""
    a = np.round(np.asarray(a, dtype=float), decimals)
    obj = a.astype(object)
    obj[np.isnan(a)] = None
    return obj.tolist()


def jst_str(t: float) -> str:
    """UNIX 時間を Plotly の日付軸で使う JST の文字列に変換する。"""
    return dt.datetime.fromtimestamp(t, JST).strftime("%Y-%m-%d %H:%M:%S")


def n_range(query: dict) -> int:
    """表示する最大距離 [km] をレンジゲート数に変換する。"""
    km = float(query.get("max_range", ["4.24"])[0])
    return int(np.clip(round(km / RNG_STEP), 1, RNG_NUM))


def find_folders(root: Path) -> list[str]:
    """親フォルダ自身と直下のフォルダのうち、.bin ファイルを含むものを返す。"""

    def has_bin(d: Path) -> bool:
        try:
            with os.scandir(d) as it:
                return any(e.is_file() and FILE_RE.match(e.name) for e in it)
        except OSError:
            return False

    folders = ["."] if has_bin(root) else []
    try:
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        subdirs = []
    folders += [p.name for p in subdirs if not p.name.startswith(".") and has_bin(p)]
    return folders


def build_time_height(items: list, nr: int, axis: str) -> dict:
    """処理済みファイルの結果を、時間・高度図の描画用データにまとめる。

    axis が "obs" のときは推定観測時刻、"mtime" のときはファイルの更新時刻を
    横軸にする。
    """
    if not items:
        return {
            "time": [],
            "name": [],
            "short_time": [],
            "gaps": [],
            "first_time": None,
        }
    names = [n for n, _ in items]
    sizes = np.array([e.size for _, e in items])
    times, fixed = fix_timestamps(np.array([e.mtime for _, e in items]))
    # 記録が途中で途切れたファイルは、通常のファイルの半分程度のサイズになる
    short = sizes < np.median(sizes) * 0.75
    # 最後のファイルは記録の終了で小さくなることがあるため、途切れの対象から外す
    short[-1] = False
    # 途切れたファイルと、別の記録（ファイル名の日時が異なる）の最初のファイルで区間を区切る
    series = np.array([FILE_RE.match(n)["series"] for n in names])
    seg_start = short.copy()
    seg_start[1:] |= series[1:] != series[:-1]
    obs = estimate_observation_time(times, seg_start)
    x = obs if axis == "obs" else times

    moving = np.array([e.profile["moving"][:nr] for _, e in items])
    velocity = np.array([e.profile["mean_velocity"][:nr] for _, e in items])
    vmin, vmax = np.nanpercentile(moving, [5, 99.5])

    # 欠測を探す。ファイルの間隔が通常の 3 倍（最低 20 秒）を超えた時間を欠測とする。
    # 推定観測時刻では、途切れの位置で失われた時間（約 5 分）もここで見つかる
    step = float(np.median(np.diff(x))) if len(x) > 1 else 8.0
    gap_after = np.flatnonzero(np.diff(x) > max(3 * step, 20.0))
    gaps = [
        {
            "start": jst_str(x[g] + step / 2),
            "end": jst_str(x[g + 1] - step / 2),
            "seconds": round(float(x[g + 1] - x[g] - step)),
            # truncated: 記録ソフトのバッファが一杯になり失われた時間（推定）
            # stopped: ファイルが書き込まれていない時間
            "kind": "truncated" if short[g + 1] else "stopped",
        }
        for g in gap_after
    ]

    idx = np.arange(len(items))
    if len(idx) > MAX_COLUMNS:
        idx = np.unique(np.linspace(0, len(idx) - 1, MAX_COLUMNS).round().astype(int))

    # 欠測の位置に空の列を入れる。入れないと、欠測の前後の列が欠測の時間まで広がる
    cols = {"time": [], "name": [], "update": [], "obs": [], "m": [], "v": []}
    empty = np.full(nr, np.nan)

    def add(t, name, update, obs_t, m, v):
        for k, val in zip(cols, (t, name, update, obs_t, m, v)):
            cols[k].append(val)

    for k, i in enumerate(idx):
        if k > 0:
            for g in gap_after[(gap_after >= idx[k - 1]) & (gap_after < i)]:
                add(jst_str(x[g] + step), None, None, None, empty, empty)
        add(
            jst_str(x[i]),
            names[i],
            jst_str(times[i]),
            jst_str(obs[i]),
            moving[i],
            velocity[i],
        )

    return {
        "axis": axis,
        "alpha": OBS_DELAY_RATE,
        "time": cols["time"],
        "name": cols["name"],
        "update_time": cols["update"],
        "obs_time": cols["obs"],
        "range": to_json_array(np.arange(nr) * RNG_STEP, 3),
        "moving": to_json_array(np.array(cols["m"]).T),
        "mean_velocity": to_json_array(np.array(cols["v"]).T),
        "power_range": [round(float(vmin), 1), round(float(vmax), 1)],
        "short_time": [jst_str(x[i]) for i in np.flatnonzero(short)],
        "gaps": gaps,
        "decimated": len(idx) < len(items),
        "n_fixed_time": int(fixed.sum()),
        "first_time": jst_str(x[0]),
        "last_time": jst_str(x[-1]),
    }


@lru_cache(maxsize=8)
def analyze_file(path: str, size: int, mtime: float) -> dict[str, np.ndarray]:
    """1 ファイルのドップラースペクトルと方位ごとの受信電力を計算する。

    size と mtime はキャッシュのキーとして使う。ファイルが書き換えられた場合は
    計算し直す。
    """
    iq, angle = read_sweeps(path, return_angles=True)
    spec, power = mean_spectrum(iq)
    del iq
    spec = np.fft.fftshift(spec, axes=0)
    n_azi = power.shape[0]
    velocity = (
        np.fft.fftshift(np.fft.fftfreq(HIT_NUM))
        * PRF
        * (SPEED_OF_LIGHT / RADAR_FREQ)
        / 2
    )
    azi = angle[: n_azi * HIT_NUM].reshape(n_azi, HIT_NUM).mean(axis=1)
    # 往復の走査で同じ角度を複数回通る場合は、角度ごとに平均する
    key = np.round(azi / AZI_STEP) * AZI_STEP
    angles = np.unique(key) + 0.0  # -0.0 を 0.0 にする
    ppi = np.array([power[key == a].mean(axis=0) for a in angles])
    with np.errstate(divide="ignore"):
        return {
            "spectrum": 10 * np.log10(spec).astype(np.float32),
            "velocity": velocity,
            "angle": angles,
            "ppi": 10 * np.log10(ppi).astype(np.float32),
        }


class Handler(BaseHTTPRequestHandler):
    """画面の HTML と、解析結果を返す API を提供する。"""

    root: Path
    watcher: Watcher

    def log_message(self, fmt: str, *args) -> None:  # アクセスログは出さない
        pass

    def send_body(self, body: bytes, content_type: str, cache: bool = False) -> None:
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024:
            body = gzip.compress(body, compresslevel=5)
            encoding = "gzip"
        else:
            encoding = None
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj: dict) -> None:
        self.send_body(json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self.send_body(
                    (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8"
                )
            elif url.path == "/app.js":
                self.send_body(
                    (STATIC_DIR / "app.js").read_bytes(),
                    "text/javascript; charset=utf-8",
                )
            elif url.path == "/plotly.min.js":
                self.send_body(
                    PLOTLY_JS.read_bytes(), "text/javascript; charset=utf-8", cache=True
                )
            elif url.path == "/api/folders":
                self.send_json(
                    {"root": str(self.root), "folders": find_folders(self.root)}
                )
            elif url.path == "/api/timeheight":
                self.api_time_height(query)
            elif url.path == "/api/file":
                self.api_file(query)
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "見つかりません")
        except BrokenPipeError, ConnectionResetError:
            pass
        except Exception as e:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
            raise

    def folder_path(self, query: dict) -> Path | None:
        """画面から指定されたフォルダを、一覧にあるフォルダに限って返す。"""
        folder = query.get("folder", [""])[0]
        if folder not in find_folders(self.root):
            self.send_error_json(
                HTTPStatus.NOT_FOUND, f"フォルダがありません: {folder}"
            )
            return None
        return (self.root / folder).resolve()

    def api_time_height(self, query: dict) -> None:
        path = self.folder_path(query)
        if path is None:
            return
        state = self.watcher.select(path)
        version, items, status = state.snapshot()
        nr = n_range(query)
        axis = "mtime" if query.get("axis", ["obs"])[0] == "mtime" else "obs"
        tag = f"{version}-{nr}-{axis}"
        result = {"tag": tag, "status": status}
        if query.get("tag", [""])[0] != tag:
            result["data"] = build_time_height(items, nr, axis)
        self.send_json(result)

    def api_file(self, query: dict) -> None:
        path = self.folder_path(query)
        if path is None:
            return
        name = query.get("name", [""])[0]
        entry = self.watcher.select(path).entries.get(name)
        if entry is None or entry.profile is None:
            self.send_error_json(
                HTTPStatus.NOT_FOUND, f"処理済みのファイルではありません: {name}"
            )
            return
        nr = n_range(query)
        st = (path / name).stat()
        with ANALYZE_LOCK:
            res = analyze_file(str(path / name), st.st_size, st.st_mtime)
        p = entry.profile
        self.send_json(
            {
                "name": name,
                "time": jst_str(entry.mtime),
                "range": to_json_array(np.arange(nr) * RNG_STEP, 3),
                "profile": {k: to_json_array(p[k][:nr]) for k in p},
                "velocity": to_json_array(res["velocity"], 3),
                "spectrum": to_json_array(res["spectrum"][:, :nr].T, 1),
                "angle": to_json_array(res["angle"], 2),
                "ppi": to_json_array(res["ppi"][:, :nr], 1),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IQ データの解析結果をブラウザで表示する Web サーバーを起動する。"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="IQ データのフォルダを探す親フォルダ（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument("--port", type=int, default=8000, help="待ち受けるポート番号")
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
        "--settle",
        type=float,
        default=60.0,
        help="後続ファイルがない場合に、最終更新から何秒経てば完了とみなすか（デフォルト: 60）",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).parent / "cache"),
        help="処理結果のキャッシュを保存するフォルダ（デフォルト: webapp/cache）",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="起動時にブラウザを開かない"
    )
    args = parser.parse_args()

    Handler.root = Path(args.root).resolve()
    Handler.watcher = Watcher(
        Path(args.cache_dir), args.clutter_bins, args.snr_min, args.settle
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"親フォルダ: {Handler.root}")
    print(f"{url} を開いてください（終了は Ctrl+C）")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
