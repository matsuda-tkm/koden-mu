"""IQ データのフォルダを監視し、取り込みが完了したファイルを処理するモジュール。

記録ソフトは .bin ファイルを 1 個ずつ順に書き込む。書き込み中のファイルを
読まないように、次のどちらかを満たしたファイルだけを完了とみなして処理する。

- 同じ記録の後続ファイル（番号が大きいファイル）があり、前回の走査から
  サイズと更新時刻が変わっていない
- 最終更新から settle 秒以上経っている（記録を終えた後の最後のファイルなど）

記録の終了後に更新時刻やサイズが変わるファイルがあるため、処理済みの
ファイルでもサイズか更新時刻が変わった場合は処理し直す。
"""

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from koden_mu.radar import BLOCK_SIZE, HIT_NUM, RNG_NUM, compute_profile, read_sweeps

FILE_RE = re.compile(r"^(?P<series>.+)_(?P<num>\d{4})\.bin$")
PROFILE_KEYS = ("total", "moving", "mean_velocity", "snr")
MIN_SIZE = BLOCK_SIZE * 8 * HIT_NUM  # 1 方位分（HIT_NUM ヒット）のバイト数


@dataclass
class Entry:
    """処理済みファイル 1 個分の結果。"""

    size: int
    mtime: float
    profile: dict[str, np.ndarray] | None  # 解析に使わないファイルは None
    skip_reason: str = ""


class FolderState:
    """1 つのフォルダの走査状態と処理結果を保持する。"""

    def __init__(
        self,
        path: Path,
        cache_dir: Path,
        clutter_bins: int,
        snr_min: float,
        settle: float,
    ) -> None:
        self.path = path
        self.clutter_bins = clutter_bins
        self.snr_min = snr_min
        self.settle = settle
        digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
        self.cache_path = cache_dir / f"{path.name}_{digest}.npz"
        self.lock = threading.Lock()
        self.entries: dict[str, Entry] = {}
        self.prev_stat: dict[str, tuple[int, float]] = {}
        self.writing: list[str] = []  # 取り込み中と判定したファイル
        self.queue: list[str] = []  # 完了していて未処理のファイル
        self.version = 0  # 処理結果が変わるたびに増やす
        self.scanned = False  # フォルダを 1 回以上確認したか
        self.cache_dirty = False
        self._load_cache()

    def _load_cache(self) -> None:
        """前回までの処理結果を読み込む。処理条件が異なるキャッシュは使わない。"""
        if not self.cache_path.exists():
            return
        try:
            with np.load(self.cache_path) as z:
                if (
                    int(z["clutter_bins"]) != self.clutter_bins
                    or float(z["snr_min"]) != self.snr_min
                ):
                    return
                # z[key] はアクセスのたびにファイルから読み直すため、先に全部読む
                c = {k: z[k] for k in z.files}
            for i, name in enumerate(c["name"]):
                skip = str(c["skip_reason"][i])
                profile = None if skip else {k: c[k][i] for k in PROFILE_KEYS}
                self.entries[str(name)] = Entry(
                    int(c["size"][i]), float(c["mtime"][i]), profile, skip
                )
        except Exception as e:  # 壊れたキャッシュは捨てて処理し直す
            print(f"キャッシュを読み込めませんでした ({self.cache_path}): {e}")
            self.entries = {}

    def save_cache(self) -> None:
        """処理結果をキャッシュファイルに保存する。"""
        with self.lock:
            if not self.cache_dirty:
                return
            items = sorted(self.entries.items())
            self.cache_dirty = False
        nan_row = np.full(RNG_NUM, np.nan, dtype=np.float32)
        arrays = {
            k: np.array(
                [e.profile[k] if e.profile else nan_row for _, e in items],
                dtype=np.float32,
            ).reshape(len(items), RNG_NUM)
            for k in PROFILE_KEYS
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            name=np.array([n for n, _ in items], dtype=str),
            size=np.array([e.size for _, e in items], dtype=np.int64),
            mtime=np.array([e.mtime for _, e in items]),
            skip_reason=np.array([e.skip_reason for _, e in items], dtype=str),
            clutter_bins=self.clutter_bins,
            snr_min=self.snr_min,
            **arrays,
        )
        tmp.replace(self.cache_path)

    def scan(self) -> None:
        """フォルダを走査し、処理待ちのファイルと取り込み中のファイルを更新する。"""
        stats: dict[str, tuple[int, float]] = {}
        last_num: dict[str, int] = {}
        for p in self.path.glob("*.bin"):
            m = FILE_RE.match(p.name)
            if not m:
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            stats[p.name] = (st.st_size, st.st_mtime)
            num = int(m["num"])
            last_num[m["series"]] = max(last_num.get(m["series"], -1), num)

        now = time.time()
        queue, writing = [], []
        for name, stat in sorted(stats.items()):
            m = FILE_RE.match(name)
            has_next = int(m["num"]) < last_num[m["series"]]
            old = now - stat[1] >= self.settle
            stable = old or self.prev_stat.get(name) == stat
            if not (stable and (has_next or old)):
                writing.append(name)
                continue
            entry = self.entries.get(name)
            if entry is None or (entry.size, entry.mtime) != stat:
                queue.append(name)

        with self.lock:
            removed = set(self.entries) - set(stats)
            for name in removed:
                del self.entries[name]
            if removed:
                self.version += 1
                self.cache_dirty = True
            self.prev_stat = stats
            self.scanned = True
            self.queue = queue
            self.writing = writing

    def process(self, name: str) -> None:
        """1 ファイルを処理して結果を登録する。"""
        path = self.path / name
        try:
            st = path.stat()
        except FileNotFoundError:
            return
        size, mtime = st.st_size, st.st_mtime
        profile, skip = None, ""
        if name.endswith("_0000.bin"):
            # 1 ファイル目は駆動範囲のデータがそろっていない場合がある
            skip = "1 ファイル目（_0000）"
        elif size < MIN_SIZE:
            skip = "1 方位分に満たない"
        else:
            try:
                iq = read_sweeps(str(path))
            except Exception as e:
                print(f"読み込めませんでした ({path}): {e}")
                iq = None
                skip = "読み込みエラー"
            if iq is not None and iq.shape[0] < HIT_NUM:
                skip = "1 方位分に満たない"
            elif iq is not None:
                result = compute_profile(iq, self.clutter_bins, self.snr_min)
                profile = {k: result[k].astype(np.float32) for k in PROFILE_KEYS}
        with self.lock:
            self.entries[name] = Entry(size, mtime, profile, skip)
            if name in self.queue:
                self.queue.remove(name)
            self.version += 1
            self.cache_dirty = True

    def snapshot(self) -> tuple[int, list[tuple[str, Entry]], dict]:
        """処理結果と状態をまとめて返す。"""
        with self.lock:
            items = sorted(
                (n, e) for n, e in self.entries.items() if e.profile is not None
            )
            status = {
                "scanned": self.scanned,
                "processed": len(items),
                "skipped": sum(1 for e in self.entries.values() if e.profile is None),
                "queued": len(self.queue),
                "writing": list(self.writing),
            }
            return self.version, items, status


class Watcher:
    """画面で選択中のフォルダを、バックグラウンドのスレッドで処理し続ける。

    CPU を使い過ぎないように、処理するのは最後に選択されたフォルダだけにする。
    """

    def __init__(
        self,
        cache_dir: Path,
        clutter_bins: int,
        snr_min: float,
        settle: float,
        interval: float = 2.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.clutter_bins = clutter_bins
        self.snr_min = snr_min
        self.settle = settle
        self.interval = interval
        self.states: dict[Path, FolderState] = {}
        self.active: FolderState | None = None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def select(self, path: Path) -> FolderState:
        """フォルダを処理対象にして、その状態を返す。"""
        with self.lock:
            state = self.states.get(path)
            if state is None:
                state = FolderState(
                    path, self.cache_dir, self.clutter_bins, self.snr_min, self.settle
                )
                self.states[path] = state
            if self.active is not state:
                self.active = state
                self.wake.set()
            return state

    def _run(self) -> None:
        last_save = time.time()
        while True:
            state = self.active
            if state is not None:
                try:
                    state.scan()
                    for name in list(state.queue):
                        if self.active is not state:
                            break
                        state.process(name)
                        if time.time() - last_save > 30:
                            state.save_cache()
                            last_save = time.time()
                    state.save_cache()
                except Exception as e:
                    print(f"処理中にエラーが発生しました ({state.path}): {e}")
            self.wake.wait(self.interval)
            self.wake.clear()
