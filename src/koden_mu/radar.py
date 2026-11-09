"""レーダーIQデータ処理の共通関数・定数モジュール。

バイナリIQデータの読み込み、電力変換、ドップラー処理、
PPI座標生成、カラーマップ生成などの共通処理を提供する。
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# レーダーシステム定数
RNG_NUM: int = 2048  # レンジゲート数
HIT_NUM: int = 200  # ヒット数（コヒーレント積分数）
RNG_STEP: float = 0.02  # レンジ分解能 [km/gate]
AZI_STEP: float = 1.0  # 方位分解能 [deg/step]
RADAR_FREQ: float = 9.4e9  # 送信周波数 [Hz]
SPEED_OF_LIGHT: float = 2.998e8  # 光速 [m/s]
PRF: float = 2400.0  # パルス繰り返し周波数 [Hz]
CAL_BOUNDARY: int = 213  # キャリブレーション切替レンジゲート
CAL_NEAR_DB: float = -387.7  # 近距離補正係数 [dB]
CAL_FAR_DB: float = -398.1  # 遠距離補正係数 [dB]
# 記録ソフトの書き込みの遅れが増える割合。2026-11-03 の追加観測で測った
# 取り込み（約 3.1 MB/s）と書き込み（約 2.1 MB/s）の速さから求めた値
OBS_DELAY_RATE: float = 0.30

_RADAR_COLORS: list[str] = [
    "#FFFFFF",
    "#A0D2FF",
    "#218CFF",
    "#0041FF",
    "#FAF500",
    "#FF9900",
    "#FF2800",
    "#B40068",
]


def read_iq(input_file: str, azi_num: int) -> np.ndarray:
    """バイナリファイルからIQデータを読み込む。

    ファイルヘッダを逐次解析し、IフラグとQフラグに対応するデータを
    (方位数, レンジ数, ヒット数) の配列に格納する。
    読み込み後、対数圧縮された整数値を線形振幅へ変換し、
    レンジゲートごとのキャリブレーション係数を乗算して返す。

    Args:
        input_file: 読み込むバイナリIQデータファイルのパス。
        azi_num: 処理する方位数（スキャン数）。

    Returns:
        複素IQ配列。shape は (azi_num, RNG_NUM, HIT_NUM)。
    """
    chunk = RNG_NUM // 4
    data = np.zeros((2, azi_num, RNG_NUM, HIT_NUM), dtype=np.int16)
    with open(input_file, "rb") as f:
        while True:
            f.seek(4, 1)
            iq_flag = int.from_bytes(f.read(2), "little")
            deg = int.from_bytes(f.read(2), "little") * 0.25
            div_num = int.from_bytes(f.read(4), "little")
            sweep_num = int.from_bytes(f.read(4), "little")
            cnt_azi = sweep_num // HIT_NUM
            cnt_pls = sweep_num % HIT_NUM
            if (iq_flag == 0) and (div_num == 0) and (deg == 0) and (sweep_num == 0):
                break
            if cnt_azi >= azi_num:
                break
            data[iq_flag, cnt_azi, div_num * chunk : (div_num + 1) * chunk, cnt_pls] = (
                np.frombuffer(f.read(chunk * 2), np.int16)
            )
    amplitude = np.sign(data) * 2 ** (np.abs(data) / 2.0**9)
    iq = amplitude[0] + 1j * amplitude[1]
    iq[:, :CAL_BOUNDARY, :] *= 10 ** (CAL_NEAR_DB / 20)
    iq[:, CAL_BOUNDARY:, :] *= 10 ** (CAL_FAR_DB / 20)
    return iq


def to_dbm(iq: np.ndarray) -> np.ndarray:
    """IQ信号の平均電力をdBm単位に変換する。

    ヒット方向（axis=2）に沿って電力を平均してから対数変換する。

    Args:
        iq: 複素IQ配列。shape は (..., hit_num)。

    Returns:
        平均電力配列 [dBm]。shape は iq の axis=2 を除いたもの。
    """
    return 10 * np.log10(np.mean(np.abs(iq) ** 2, axis=2))


def compute_doppler(iq: np.ndarray) -> np.ndarray:
    """IQ配列からドップラー電力スペクトル [dB] を計算する。

    ヒット軸（axis=2）に対してFFTを適用し、fftshift相当のロールを行う。

    Args:
        iq: 複素IQ配列。shape は (azi_num, rng_num, hit_num)。

    Returns:
        ドップラー電力スペクトル [dB]。shape は iq と同じ。
    """
    spectrum = 20 * np.log10(np.abs(np.fft.fft(iq, axis=2)))
    return np.roll(spectrum, HIT_NUM // 2 - 1, axis=2)


def coordinate(azi_num: int) -> tuple[np.ndarray, np.ndarray]:
    """PPI表示用の直交座標グリッドを生成する。

    レンジ・方位の等間隔グリッドを極座標から直交座標（x, y）へ変換する。

    Args:
        azi_num: 方位方向のグリッド点数。

    Returns:
        (x, y) のタプル。それぞれ shape は (azi_num+1, RNG_NUM+1) の
        2次元配列 [km]。
    """
    rng_arr = np.arange(RNG_NUM + 1) * RNG_STEP
    azi_arr = np.arange(azi_num + 1) * AZI_STEP - azi_num / 2
    azi, rng = np.meshgrid(np.deg2rad(azi_arr), rng_arr)
    azi, rng = azi.T, rng.T
    return rng * np.sin(azi), rng * np.cos(azi)


def make_colormap() -> LinearSegmentedColormap:
    """レーダー表示用のカスタムカラーマップを生成する。

    白→水色→青→濃青→黄→橙→赤→紫 の順に遷移するカラーマップを返す。

    Returns:
        LinearSegmentedColormap インスタンス。
    """
    color_list = list(zip(np.linspace(0, 1, len(_RADAR_COLORS)), _RADAR_COLORS))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)


def doppler_velocity() -> np.ndarray:
    """ドップラー速度軸の配列を生成する。

    FFT後の周波数ビンを、レーダーパラメータ（周波数・PRF）に基づく
    ドップラー速度 [m/s] へ変換する。

    Returns:
        shape (HIT_NUM,) のドップラー速度配列 [m/s]。
        ナイキスト速度 (Vmax) の範囲にロールシフトされた順で並ぶ。
    """
    v_max = (SPEED_OF_LIGHT / RADAR_FREQ) * PRF / 2
    velo = v_max * np.arange(HIT_NUM) / HIT_NUM
    velo[HIT_NUM // 2 + 1 :] -= v_max
    return np.roll(velo, HIT_NUM // 2 - 1)


BLOCK_SIZE: int = 1040  # 1ブロックのバイト数（ヘッダー16B + データ1024B）


# 記録値（uint16 として解釈した int16）から線形振幅への変換表
_CODES = np.arange(2**16, dtype=np.uint16).view(np.int16).astype(float)
_AMPLITUDE_TABLE = np.sign(_CODES) * 2.0 ** (np.abs(_CODES) / 2.0**9)


def read_sweeps(
    input_file: str, return_angles: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """バイナリファイルの全スウィープを読み込み、校正済みの複素IQ配列を返す。

    ヘッダーを一括で解析するため、read_iq より高速に読み込める。
    書き込みが途中で終わったファイルにも対応するため、I/Q と 4 分割の
    8 ブロックがすべてそろっているスウィープだけを返す。

    Args:
        input_file: 読み込むバイナリIQデータファイルのパス。
        return_angles: True のとき、各スウィープの方位角も返す。

    Returns:
        複素IQ配列。shape は (スウィープ数, RNG_NUM)。
        スウィープは記録順に並び、HIT_NUM 個ごとに 1 方位（傾き）に対応する。
        return_angles が True のときは (IQ配列, 方位角 [deg]) のタプルを返す。
        方位角は -180〜180 deg に変換する（鉛直観測では鉛直からの傾き）。
    """
    chunk = RNG_NUM // 4
    raw = np.fromfile(input_file, dtype=np.uint8)
    n_block = len(raw) // BLOCK_SIZE
    if n_block == 0:
        empty = np.zeros((0, RNG_NUM), dtype=complex)
        return (empty, np.zeros(0)) if return_angles else empty
    blocks = raw[: n_block * BLOCK_SIZE].reshape(n_block, BLOCK_SIZE)
    iq_flag = blocks[:, 4:6].copy().view("<u2").ravel()
    deg = blocks[:, 6:8].copy().view("<u2").ravel() * 0.25
    div_num = blocks[:, 8:12].copy().view("<u4").ravel()
    sweep_num = blocks[:, 12:16].copy().view("<u4").ravel()
    payload = blocks[:, 16:].copy().view("<i2")

    valid = np.flatnonzero((iq_flag <= 1) & (div_num <= 3))
    # 同じスウィープの同じ部分のブロックが複数ある場合は、最後のブロックを使う。
    # 一括代入ではインデックスが重複したときにどの値が残るか保証されないため、
    # 先に重複を除く
    slot = sweep_num[valid].astype(np.int64) * 8 + iq_flag[valid] * 4 + div_num[valid]
    _, last = np.unique(slot[::-1], return_index=True)
    rows = valid[len(valid) - 1 - last]
    f, d, s = iq_flag[rows], div_num[rows], sweep_num[rows]

    n_sweep = int(s.max()) + 1 if len(rows) else 0
    data = np.zeros((2, n_sweep, 4, chunk), dtype=np.int16)
    data[f, s, d] = payload[rows]
    filled = np.zeros((n_sweep, 8), dtype=bool)
    filled[s, f * 4 + d] = True
    # 1 スウィープ内のブロックで方位角が 0.25 deg 異なる場合がある。
    # ファイル内で最後のブロックの方位角を、そのスウィープの方位角とする
    last_row = np.full(n_sweep, -1)
    np.maximum.at(last_row, s, rows)
    angle = np.where(last_row >= 0, deg[last_row], 0.0)
    complete = filled.all(axis=1)
    data = data.reshape(2, n_sweep, RNG_NUM)[:, complete].view(np.uint16)
    del raw, blocks, payload

    # 記録値（16 bit）から振幅への変換は、取り得る全値の変換表を引いて求める。
    # 配列全体で計算すると、ファイルの十数倍の一時配列ができてメモリを消費するため
    iq = np.empty(data.shape[1:], dtype=complex)
    iq.real = _AMPLITUDE_TABLE[data[0]]
    iq.imag = _AMPLITUDE_TABLE[data[1]]
    iq[:, :CAL_BOUNDARY] *= 10 ** (CAL_NEAR_DB / 20)
    iq[:, CAL_BOUNDARY:] *= 10 ** (CAL_FAR_DB / 20)
    if return_angles:
        return iq, (angle[complete] + 180) % 360 - 180
    return iq


def mean_spectrum(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HIT_NUM ヒットごとのドップラー電力スペクトルを全方位で平均する。

    メモリの消費を抑えるため、1 方位ずつ FFT して足し合わせる。

    Args:
        iq: read_sweeps が返す複素IQ配列。shape は (スウィープ数, RNG_NUM)。

    Returns:
        (スペクトル, 方位ごとの平均電力) のタプル。
            スペクトル: shape (HIT_NUM, RNG_NUM) の線形電力。速度ビンは FFT の順
            （fftshift していない）に並ぶ
            方位ごとの平均電力: shape (方位数, RNG_NUM) の線形電力
    """
    n_azi = iq.shape[0] // HIT_NUM
    spec = np.zeros((HIT_NUM, RNG_NUM))
    power = np.empty((n_azi, RNG_NUM))
    for a in range(n_azi):
        block = iq[a * HIT_NUM : (a + 1) * HIT_NUM]
        power[a] = np.mean(np.abs(block) ** 2, axis=0)
        spec += np.abs(np.fft.fft(block, axis=0)) ** 2
    spec /= HIT_NUM**2 * n_azi
    return spec, power


def compute_profile(
    iq: np.ndarray, clutter_bins: int = 2, snr_min_db: float = -3.0
) -> dict[str, np.ndarray]:
    """1ファイル分のIQから、レンジごとの電力と平均ドップラー速度を計算する。

    HIT_NUM ヒットごとに FFT したスペクトルを全方位で平均してから計算する。
    ゼロ速度付近（片側 clutter_bins ビン）は静止物の反射とみなして除外する。
    平均ドップラー速度は、速度ビン方向の中央値を雑音レベルとして差し引いた
    スペクトルの重心とする。

    Args:
        iq: read_sweeps が返す複素IQ配列。shape は (スウィープ数, RNG_NUM)。
        clutter_bins: 除外するゼロ速度付近のビン数（片側）。
        snr_min_db: 平均ドップラー速度を求める信号対雑音比の下限 [dB]。
            下回るレンジは NaN にする。

    Returns:
        以下のキーを持つ辞書。値はすべて shape (RNG_NUM,)。
            total: 全速度ビンの平均電力 [dBm]
            moving: ゼロ速度付近を除いた電力 [dBm]
            mean_velocity: 平均ドップラー速度 [m/s]。正がレーダーへ近づく向き
            snr: 信号対雑音比 [dB]
    """
    spec, power = mean_spectrum(iq)
    total = power.mean(axis=0)

    keep = np.abs(np.fft.fftfreq(HIT_NUM) * HIT_NUM) > clutter_bins
    velocity = np.fft.fftfreq(HIT_NUM) * PRF * (SPEED_OF_LIGHT / RADAR_FREQ) / 2
    noise = np.median(spec, axis=0)
    signal = np.clip(spec[keep] - noise, 0, None)
    signal_power = signal.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * np.log10(signal_power / (noise * keep.sum()))
        mean_velocity = (velocity[keep, None] * signal).sum(axis=0) / signal_power
    mean_velocity[~(snr >= snr_min_db)] = np.nan
    return {
        "total": 10 * np.log10(total),
        "moving": 10 * np.log10(spec[keep].sum(axis=0)),
        "mean_velocity": mean_velocity,
        "snr": snr,
    }


def fix_timestamps(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """後から上書きされたファイルの時刻を、前後のファイルの時刻から補間する。

    上書きされたファイルは更新時刻が取得時刻と大きく異なるため、
    前後 10 ファイルの中央値から 120 秒以上ずれた時刻を補間の対象にする。

    Args:
        times: ファイル名順に並べた各ファイルの更新時刻（UNIX 時間 [s]）。

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


def estimate_observation_time(
    times: np.ndarray, segment_start: np.ndarray, alpha: float = OBS_DELAY_RATE
) -> np.ndarray:
    """ファイルの更新時刻から、観測時刻を推定する。

    記録ソフトは書き込みが取り込みに追いつかないため、未書き込みのデータが
    バッファに溜まり、更新時刻は観測時刻より少しずつ遅れていく。約 17 分で
    バッファが一杯になると、書きかけのファイルが途中で切れ、溜まっていたデータ
    （約 5 分）が失われて最新の観測から記録が再開される。この遅れを、区間の
    先頭からの経過時間に比例するとみなして補正する（t' = t - alpha (t - t_s)）。

    Args:
        times: ファイル名順に並べた各ファイルの更新時刻（UNIX 時間 [s]）。
        segment_start: 区間の先頭のファイルを True とする配列。記録が途中で
            途切れたファイルと、記録の最初のファイルが区間の先頭になる。
        alpha: 遅れが増える割合。

    Returns:
        推定観測時刻（UNIX 時間 [s]）。
    """
    idx = np.arange(len(times))
    start = np.where(segment_start, idx, 0)
    start[0] = 0
    seg_time = times[np.maximum.accumulate(start)]
    return times - alpha * (times - seg_time)
