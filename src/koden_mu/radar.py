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
        最大不曖昧速度 (Vmax) の範囲にロールシフトされた順で並ぶ。
    """
    v_max = (SPEED_OF_LIGHT / RADAR_FREQ) * PRF / 2
    velo = v_max * np.arange(HIT_NUM) / HIT_NUM
    velo[HIT_NUM // 2 + 1 :] -= v_max
    return np.roll(velo, HIT_NUM // 2 - 1)
