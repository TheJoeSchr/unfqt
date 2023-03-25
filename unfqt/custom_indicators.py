"""
Solipsis Custom Indicators and Maths
Some indicators are direct copies from freqtrade/technical but I did not want to rely on that repository
to be stable and I needed to make some fixes/changes to fibonacci retracements and PMAX to suit my needs.

extended by @TheJoeSchr 
"""
from typing import List, Union
from pandas import Series
import pandas as pd
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from pandas import DataFrame, Series
from pandas_ta.utils import get_drift, get_offset, verify_series, zero
from pandas_ta.volatility import atr
from pandas_ta.overlap import ma
# from pandas_ta import adx


def find_higher_tf_divergences(dataframe,
                               lower_tf_postfix='_btc_5m',
                               higher_tf_postfix='_btc_1h'):
    dataframe['low_peaks'+lower_tf_postfix] = np.NaN
    dataframe['high_peaks'+lower_tf_postfix] = np.NaN
    dataframe.loc[dataframe.index.isin(detect_peaks(
        dataframe['high'+lower_tf_postfix][:-1], valley=False, show=False)), 'high_peaks'+lower_tf_postfix] = dataframe['high'+lower_tf_postfix]
    dataframe.loc[dataframe.index.isin(detect_peaks(
        dataframe['low'+lower_tf_postfix][:-1], valley=True, show=False)), 'low_peaks'+lower_tf_postfix] = dataframe['low'+lower_tf_postfix]

    dataframe['peaks_osc'+lower_tf_postfix] = np.where(
        dataframe['low_peaks'+lower_tf_postfix] == dataframe['low'+lower_tf_postfix], dataframe['osc'+lower_tf_postfix], np.NaN)
    dataframe['peaks_osc'+higher_tf_postfix] = np.where(
        dataframe['low_peaks'+higher_tf_postfix].shift(-12) == dataframe['low'+lower_tf_postfix], dataframe['osc'+lower_tf_postfix], np.NaN)
    # index.isin: brings in latest 5m low peak
    dataframe['peaks_osc'+higher_tf_postfix+'_shifted'] = dataframe.loc[(dataframe['low_peaks'+higher_tf_postfix].shift(-12) == dataframe['low'+lower_tf_postfix]) | (
        dataframe.index.isin(dataframe.loc[dataframe['low_peaks'+lower_tf_postfix].notna()][-1::].index.values)), 'osc'+lower_tf_postfix].shift()
    # index.isin: brings in latest 5m low peak
    dataframe['peaks_low'+higher_tf_postfix+'_shifted'] = dataframe.loc[(dataframe['low_peaks'+higher_tf_postfix].shift(-12) == dataframe['low'+lower_tf_postfix]) | (
        dataframe.index.isin(dataframe.loc[dataframe['low_peaks'+lower_tf_postfix].notna()][-1::].index.values)), 'low'+lower_tf_postfix].shift()

    dataframe['found_divergence_hidden'+lower_tf_postfix] = 0
    dataframe['found_divergence_regular'+lower_tf_postfix] = 0

    dataframe.loc[(dataframe['peaks_low'+higher_tf_postfix+'_shifted'] > dataframe['low'+lower_tf_postfix]) & (
        dataframe['peaks_osc'+higher_tf_postfix+'_shifted'] <= dataframe['peaks_osc'+lower_tf_postfix]), 'found_divergence_regular'+lower_tf_postfix] = 1

    dataframe.loc[(dataframe['peaks_low'+higher_tf_postfix+'_shifted'] < dataframe['low'+lower_tf_postfix]) & (
        dataframe['peaks_osc'+higher_tf_postfix+'_shifted'] >= dataframe['peaks_osc'+lower_tf_postfix]), 'found_divergence_hidden'+lower_tf_postfix] = 1

    dataframe.loc[(dataframe['found_divergence_hidden'+lower_tf_postfix].shift() == 1) | (
        dataframe['found_divergence_regular'+lower_tf_postfix].shift() == 1), 'found_divergence_marker'+lower_tf_postfix] = dataframe['low'+lower_tf_postfix]

    # makes rolling().sum() easier than with NaN
    dataframe['found_divergence_marker' +
              lower_tf_postfix].fillna(0, inplace=True)
    return dataframe


def vwap_fast(dataframe: DataFrame):
    """
    by @rk
    """
    split_indices = list(dataframe.loc[
        (
            (dataframe['date'].dt.second == 0)
            &
            (dataframe['date'].dt.minute == 0)
            &
            (dataframe['date'].dt.hour == 0)
        )].index)
    split_indices.insert(0, 0)
    split_indices.append(len(dataframe))
    vwap_slices = []
    for i in range(1, len(split_indices)):
        start_idx = split_indices[i - 1]
        end_idx = split_indices[i]
        slice = dataframe[start_idx:end_idx]
        hlc3 = (slice['high'] + slice['low'] + slice['close']) / 3
        wp = hlc3 * slice['volume']
        vwap = wp.cumsum() / slice['volume'].cumsum()
        vwap_slices.append(vwap)
    vwap = pd.concat(vwap_slices)
    return vwap


def williams_fractal(dataframe):
    """
    DANGEROUS: needs to be used with .shift(2) because of lookahead bias
    add 'fractals_high' and 'fractals_low' to dataframe
    """
    dataframe.loc[(dataframe['high'] > dataframe['high'].shift(1)) &
                  (dataframe['high'] > dataframe['high'].shift(2)) &
                  (dataframe['high'] > dataframe['high'].shift(-1)) &
                  (dataframe['high'] > dataframe['high'].shift(-2)),
                  'fractal_high'
                  ] = 1
    dataframe.loc[((dataframe['low'] < dataframe['low'].shift(1)) &
                   (dataframe['low'] < dataframe['low'].shift(2)) &
                   (dataframe['low'] < dataframe['low'].shift(-1)) &
                   (dataframe['low'] < dataframe['low'].shift(-2))),
                  'fractal_low'] = 1
    return dataframe


"""
Misc. Helper Functions
"""


def same_length(bigger, shorter):
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))


"""
Maths
"""


def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))


"""
TA Indicators
"""


def fib_ret(df, lookback=72, field='close') -> DataFrame:
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L735
    Modifed to have a specific lookback period to prevent lookahead bias.
    """
    thresholds = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0]

    window_min = df[field].rolling(lookback).min()
    window_max = df[field].rolling(lookback).max()
    # I have no idea what this is for, maaybe graphing or jupyter?
    fib_levels = [window_min + t * (window_max - window_min)
                  for t in thresholds]

    data = (df[field] - window_min) / (window_max - window_min)

    # this return is broken and I don't know how to fix it.
    # return data.apply(lambda x: max(t for t in thresholds if x >= t))
    return data


def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 /
                         (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]


def VIDYA(dataframe, length=9, select=True):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L936
    """
    df = dataframe.copy()

    alpha = 2 / (length + 1)
    df['momm'] = df['close'].diff()
    df['m1'] = np.where(df['momm'] >= 0, df['momm'], 0.0)
    df['m2'] = np.where(df['momm'] >= 0, 0.0, -df['momm'])

    df['sm1'] = df['m1'].rolling(length).sum()
    df['sm2'] = df['m2'].rolling(length).sum()

    df['chandeMO'] = 100 * (df['sm1'] - df['sm2']) / (df['sm1'] + df['sm2'])
    if select:
        df['k'] = abs(df['chandeMO']) / 100
    else:
        df['k'] = df['close'].rolling(length).std()
    df.fillna(0.0, inplace=True)

    df['VIDYA'] = 0.0
    for i in range(length, len(df)):

        df['VIDYA'].iat[i] = alpha * df['k'].iat[i] * df['close'].iat[i] + \
            (1 - alpha * df['k'].iat[i]) * df['VIDYA'].iat[i - 1]

    return df['VIDYA']


def vwma(df, window):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L75
    """
    return (df['close'] * df['volume']).rolling(window).sum() / df.volume.rolling(window).sum()


def zema(dataframe, period, field='close'):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
    Modified slightly to use ta.EMA instead of technical ema
    """
    df = dataframe.copy()

    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']

    return df['zema']


def PMAX(dataframe, period=10, multiplier=3, length=12, MAtype=1, src=1):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1059
    Modified to return different series instead of a modified dataframe as well as not use unique
    field names based on the input parameters.
    returns: atr, ma, pmax_value, pmax_location
    src==1 --> close
    src==2 --> hl2
    src==3 --> ohlc4
    MAtype==1 --> EMA
    MAtype==2 --> DEMA
    MAtype==3 --> T3
    MAtype==4 --> SMA
    MAtype==5 --> VIDYA
    MAtype==6 --> TEMA
    MAtype==7 --> WMA
    MAtype==8 --> VWMA
    MAtype==9 --> zema
    """
    df = dataframe.copy()
    mavalue = 'ma'
    atr = 'atr'
    pm = 'pm'
    pmx = 'pmx'

    df[atr] = ta.ATR(df, timeperiod=period)

    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        df[mavalue] = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        df[mavalue] = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        df[mavalue] = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        df[mavalue] = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        df[mavalue] = VIDYA(df, length=length)
    elif MAtype == 6:
        df[mavalue] = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        df[mavalue] = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        df[mavalue] = vwma(df, length)
    elif MAtype == 9:
        df[mavalue] = zema(df, period=length)

    # Compute basic upper and lower bands
    df['basic_ub'] = df[mavalue] + (multiplier * df[atr])
    df['basic_lb'] = df[mavalue] - (multiplier * df[atr])
    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if (
            df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1]
            or df[mavalue].iat[i - 1] > df['final_ub'].iat[i - 1]) else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if (
            df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1]
            or df[mavalue].iat[i - 1] < df['final_lb'].iat[i - 1]) else df['final_lb'].iat[i - 1]

    # Set the Pmax value
    df[pm] = 0.00
    for i in range(period, len(df)):
        df[pm].iat[i] = (
            df['final_ub'].iat[i] if (df[pm].iat[i - 1] == df['final_ub'].iat[i - 1]
                                      and df[mavalue].iat[i] <= df['final_ub'].iat[i])
            else df['final_lb'].iat[i] if (
                df[pm].iat[i - 1] == df['final_ub'].iat[i - 1]
                and df[mavalue].iat[i] > df['final_ub'].iat[i]) else df['final_lb'].iat[i]
            if (df[pm].iat[i - 1] == df['final_lb'].iat[i - 1]
                and df[mavalue].iat[i] >= df['final_lb'].iat[i]) else df['final_ub'].iat[i]
            if (df[pm].iat[i - 1] == df['final_lb'].iat[i - 1]
                and df[mavalue].iat[i] < df['final_lb'].iat[i]) else 0.00)

    # Mark the trend direction up/down
    df[pmx] = np.where((df[pm] > 0.00), np.where(
        (df[mavalue] < df[pm]), 'down', 'up'), np.NaN)
    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub',
            'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df[atr], df[mavalue], df[pm], df[pmx]


def macross(dataframe: DataFrame, fast: int = 20, slow: int = 50) -> Series:
    """
    Moving Average Cross
    Port of: https://www.tradingview.com/script/PcWAuplI-Moving-Average-Cross/
    """
    df = dataframe.copy()

    # Fast MAs
    upper_fast = ta.EMA(df['high'], timeperiod=fast)
    lower_fast = ta.EMA(df['low'], timeperiod=fast)

    # Slow MAs
    upper_slow = ta.EMA(df['high'], timeperiod=slow)
    lower_slow = ta.EMA(df['low'], timeperiod=slow)

    # Crosses
    crosses_lf_us = qtpylib.crossed_above(
        lower_fast, upper_slow) | qtpylib.crossed_below(lower_fast, upper_slow)
    crosses_uf_ls = qtpylib.crossed_above(
        upper_fast, lower_slow) | qtpylib.crossed_below(upper_fast, lower_slow)

    dir_1 = np.where(crosses_lf_us, 1, np.nan)
    dir_2 = np.where(crosses_uf_ls, -1, np.nan)

    dir = np.where(dir_1 == 1, dir_1, np.nan)
    dir = np.where(dir_2 == -1, dir_2, dir_1)

    res = Series(dir).fillna(method='ffill').to_numpy()

    return res


def mastreak(dataframe: DataFrame, period: int = 4, field='close') -> Series:
    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """
    df = dataframe.copy()

    avgval = zema(df, period, field)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                      -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = same_length(df['close'], streak)

    return res


def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """
    df = dataframe.copy()

    df['previous_close'] = df['close'].shift()

    df['close_change'] = (df['close'] - df['previous_close']
                          ) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100

    df['delta'] = df['high_change'] - df['low_change']

    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, rangema, lower


def SSLChannels(dataframe, length=10, mode='sma'):
    """
    Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1025
    Author: xmatthias
    Pinescript Author: ErwinBeckers
    SSL Channels.
    Average over highs and lows form a channel - lines "flip" when close crosses either of the 2 lines.
    Trading ideas:
        * Channel cross
        * as confirmation based on up > down for long
    Usage:
        dataframe['sslDown'], dataframe['sslUp'] = SSLChannels(dataframe, 10)
    """
    if mode not in ('sma'):
        raise ValueError(f"Mode {mode} not supported yet")

    df = dataframe.copy()

    if mode == 'sma':
        df['smaHigh'] = df['high'].rolling(length).mean()
        df['smaLow'] = df['low'].rolling(length).mean()

    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,
                         np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()

    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

# From CombinedBinHAAndCluc


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


"""
Market Cipher, from @drafty in Freqtrade discord
"""


def market_cipher(self, dataframe) -> DataFrame:
    #dataframe['volume_rolling'] = dataframe['volume'].shift(14).rolling(14).mean()
    #
    dataframe['ap'] = (dataframe['high'] +
                       dataframe['low'] + dataframe['close']) / 3
    dataframe['esa'] = ta.EMA(dataframe['ap'], self.n1)
    dataframe['d'] = ta.EMA(
        (dataframe['ap'] - dataframe['esa']).abs(), self.n1)
    dataframe['ci'] = (dataframe['ap'] - dataframe['esa']) / \
        (0.015 * dataframe['d'])
    dataframe['tci'] = ta.EMA(dataframe['ci'], self.n2)

    dataframe['wt1'] = dataframe['tci']
    dataframe['wt2'] = ta.SMA(dataframe['wt1'], 4)
    dataframe['wt1-wt2'] = dataframe['wt1'] - dataframe['wt2']

    dataframe['crossed_above'] = qtpylib.crossed_above(
        dataframe['wt2'], dataframe['wt1'])
    dataframe['crossed_below'] = qtpylib.crossed_below(
        dataframe['wt2'], dataframe['wt1'])
    #dataframe['slope_gd'] = ta.LINEARREG_ANGLE(dataframe['crossed_above'] * dataframe['wt2'], 10)

    return dataframe


def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def adx(high, low, close, length=None, scalar=None, mamode=None, drift=None, offset=None, lensig=None, **kwargs):
    """Indicator: ADX"""
    # Validate Arguments
    length = length if length and length > 0 else 14
    lensig = lensig if lensig and lensig > 0 else length
    mamode = mamode if isinstance(mamode, str) else "rma"
    scalar = float(scalar) if scalar else 100
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return

    # Calculate Result
    atr_ = atr(high=high, low=low, close=close, length=length)

    up = high - high.shift(drift)  # high.diff(drift)
    dn = low.shift(drift) - low    # low.diff(-drift).shift(drift)

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn

    pos = pos.apply(zero)
    neg = neg.apply(zero)

    k = scalar / atr_
    dmp = k * ma(mamode, pos, length=length)
    dmn = k * ma(mamode, neg, length=length)

    dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
    adx = ma(mamode, dx, length=lensig)

    # Offset
    if offset != 0:
        dmp = dmp.shift(offset)
        dmn = dmn.shift(offset)
        adx = adx.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        adx.fillna(kwargs["fillna"], inplace=True)
        dmp.fillna(kwargs["fillna"], inplace=True)
        dmn.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        adx.fillna(method=kwargs["fill_method"], inplace=True)
        dmp.fillna(method=kwargs["fill_method"], inplace=True)
        dmn.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    adx.name = f"ADX_{lensig}"
    dmp.name = f"DMP_{length}"
    dmn.name = f"DMN_{length}"

    adx.category = dmp.category = dmn.category = "trend"

    # Prepare DataFrame to return
    data = {adx.name: adx, dmp.name: dmp, dmn.name: dmn}
    adxdf = DataFrame(data)
    adxdf.name = f"ADX_{lensig}"
    adxdf.category = "trend"

    return adxdf
