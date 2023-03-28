import functools
from typing import Union
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pandas
from itertools import compress
from numbers import Number
from typing import Sequence, Optional, Union, Callable


def angle(a, b):
    """Convert angle between points `a` and `b` to an angle in degrees, relative to x axis."""
    deg = np.rad2deg(np.arctan2(b[1] - a[1], b[0] - a[0]))
    deg = (90 - abs(deg)) * (+1 if deg > 0 else -1)
    return deg


def pivot(s: Series, cmp, left=1, right=1):
    conditions = []
    for i in range(left):
        conditions.append(cmp(s, s.shift(+i + 1)))
    for i in range(right):
        conditions.append(cmp(s, s.shift(-i - 1)))
    return functools.reduce(lambda a, b: a & b, conditions)


def pivotlow(s: Series, left=1, right=1):
    return pivot(s, np.less, left, right)


def pivothigh(s: Series, left=1, right=1):
    return pivot(s, np.greater, left, right)


def repeat_condition(self, series: pandas.Series, length: int):
    '''
    HOWTO:
    dataframe.loc[
        (
            (dataframe['go_long'] > 1) &
            repeat_condition(dataframe['minus_di_5m'] < 1.2 * dataframe['minus_di_5m'].shift(1), 5)
        ),
        'buy'] = 0
    '''
    conditions = []
    for i in range(length):
        conditions.append(series.shift(i))
    return functools.reduce(lambda a, b: a & b, conditions)


def line2arr(line, size=-1):
    '''
    Creates an numpy array from a backtrader line

    This method wraps the lines array in numpy. This can
    be used for conditions.
    '''
    if size <= 0:
        return np.array(line.array)
    else:
        return np.array(line.get(size=size))


def na(val):
    '''
    RETURNS
    true if x is not a valid number (x is NaN), otherwise false.
    '''
    return val != val


def nz(x, y=None):
    '''
    RETURNS
    Two args version: returns x if it's a valid (not NaN) number, otherwise y
    One arg version: returns x if it's a valid (not NaN) number, otherwise 0
    ARGUMENTS
    x (val) Series of values to process.
    y (float) Value that will be inserted instead of all NaN values in x series.
    '''
    if isinstance(x, np.generic):
        return x.fillna(y or 0)
    if x != x:
        if y is not None:
            return y
        return 0
    return x


def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    Return the number of bars since `condition` sequence was last `True`,
    or if never, return `default`.
        >>> barssince(self.data.Close > self.data.Open)
        3
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


def barssince_while(condition, occurrence=0):
    '''
    Impl of barssince

    RETURNS
    Number of bars since condition was true.
    REMARKS
    If the condition has never been met prior to the current bar, the function returns na.
    '''
    cond_len = len(condition)
    occ = 0
    since = 0
    res = float('nan')
    while cond_len - (since+1) >= 0:
        cond = condition[cond_len-(since+1)]
        # check for nan cond != cond == True when nan
        if cond and not cond != cond:
            if occ == occurrence:
                res = since
                break
            occ += 1
        since += 1
    return res


def valuewhen(condition, source, occurrence=0):
    # NOTE: checked, not .shift(-occurrence) is correct
    return source \
        .reindex(condition[condition].index) \
        .shift(occurrence) \
        .reindex(source.index) \
        .ffill()


def valuewhen_iter(condition, source, occurrence=0):
    '''
    Impl of valuewhen
    + added occurrence

    RETURNS
    Source value when condition was true
    '''
    res = float('nan')
    since = barssince(condition, occurrence)
    if since is not None:
        res = source[-(since+1)]
    return res


def calc_streaks(series: pd.Series):

    # logic tables
    geq = series >= series.shift(1)  # True if rising
    eq = series == series.shift(1)  # True if equal
    logic_table = pd.concat([geq, eq], axis=1)

    streaks = [0]  # holds the streak duration, starts with 0

    for row in logic_table.iloc[1:].itertuples():  # iterate through logic table
        if row[2]:  # same value as before
            streaks.append(0)
            continue
        last_value = streaks[-1]
        if row[1]:  # higher value than before
            streaks.append(last_value + 1 if last_value >=
                           0 else 1)  # increase or reset to +1
        else:  # lower value than before
            streaks.append(last_value - 1 if last_value <
                           0 else -1)  # decrease or reset to -1

    return streaks


def cross_up(a: Series, b: Union[Series, int, float]):
    b_prev = b.shift(1) if isinstance(b, Series) else b
    return (a.shift(1) <= b_prev) & (a > b)


def cross_down(a: Series, b: Union[Series, int, float]):
    b_prev = b.shift(1) if isinstance(b, Series) else b
    return (a.shift(1) >= b_prev) & (a < b)


def cross_any(a: Series, b: Union[Series, int, float]):
    return cross_up(a, b) | cross_down(a, b)


def going_up(series: Series, period: int):
    conditions = []
    for i in range(period):
        conditions.append(series.shift(i + 1) < series.shift(i))
    return functools.reduce(lambda a, b: a & b, conditions)


def going_down(series: Series, period: int):
    conditions = []
    for i in range(period):
        conditions.append(series.shift(i + 1) > series.shift(i))
    return functools.reduce(lambda a, b: a & b, conditions)


def typical_price(dataframe: DataFrame):
    return (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3


def repeat_condition(condition: Series, length: int, op: str = 'and', step: int = 1):
    """
    Repeat condition `length` times shifting it by `step`.
    :param condition:
    :param length:
    :param op: 'or' or 'and'
    :param step:
    :return:
    """
    conditions = []
    for i in range(length):
        conditions.append(condition.shift(i * step))
    if op == 'or':
        def reducer(a, b):
            return a | b
    elif op == 'and':
        def reducer(a, b):
            return a & b
    else:
        raise Exception('op should be "or" or "and"')
    return functools.reduce(reducer, conditions)


def value_when(condition, source, occurrence):
    return source \
        .reindex(condition[condition].index) \
        .shift(-occurrence) \
        .reindex(source.index) \
        .ffill()


def heikin_ashi(dataframe: DataFrame):
    hk = DataFrame()
    hk['close'] = (dataframe['open'] + dataframe['close'] +
                   dataframe['high'] + dataframe['low']) / 4
    hk['open'] = (dataframe['open'].shift(1) + dataframe['close'].shift(1)) / 2
    hk['low'] = dataframe['low']
    hk['low'] = hk[['low', 'open', 'close']].min(axis=1)
    hk['high'] = dataframe['high']
    hk['high'] = hk[['high', 'open', 'close']].max(axis=1)
    return hk


def ssl(high, low, close, length=10):
    df = DataFrame()
    df['sma_high'] = ta.sma(high, length=length)
    df['sma_low'] = ta.sma(low, length=length)
    df['hlv'] = np.where(close > df['sma_high'], 1,
                         np.where(close < df['sma_low'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['ssl_down'] = np.where(df['hlv'] < 0, df['sma_high'], df['sma_low'])
    df['ssl_up'] = np.where(df['hlv'] < 0, df['sma_low'], df['sma_high'])
    return df['ssl_down'], df['ssl_up']


def bov(df: DataFrame):
    """
    Balance of volume indicator by @rk. Splits entire volume into buying and selling volume.
    :param df: dataframe.
    :return: (buying_volume, selling_volume)
    """
    v: Series = df['volume']
    power = ta.bop(df['open'], df['high'], df['low'], df['close'])
    # bop range is [-1, +1], convert it to [0, +1]
    power = 0.5 + power / 2
    vol_up = v * power              # volume in the direction of the candle
    # volume against the direction of the candle
    vol_dn = v * (1 - power)
    return vol_up, vol_dn


def vres(df: DataFrame):
    """
    Volume-derived price movement resistance indicator by @rk. Measures how much volume
    is required to move price. Indicator value range is undefined. Values should be
    interpreted by comparing them to one another. Value fluctuates from negative to
    positive, sign indicating candle direction. Values closer to 0 mean reduced
    resistance while values further from 0 mean increased resistance.

    Consider two candles, red and green, of equal size. vres_red.abs() > vres_green.abs()
    signals that downwards resistance is greater and it is easier for price to move
    up, therefore it is a bullish sign.

    :param df:
    :return:
    """
    v = df['volume']
    h = df['high'] - df['low']
    power = ta.bop(df['open'], df['high'], df['low'], df['close'])
    # bop range is [-1, +1], convert it to [0, +1], < 0.5 - sellers in control, > 0.5 - buyers in control
    power = 0.5 + power / 2
    # Convert power to % of control in direction of the candle.
    power = np.where(df['open'] < df['close'], 1 - power, power)
    return v * power / h


def vwbop(df: DataFrame, length=14, kind='sma'):
    """
    Volume-weighted balance of power indicator by @rk.
    :param df:
    :return:
    """
    bop = ta.bop(df['open'], df['high'], df['low'], df['close'])
    volume = df['volume'] * bop.abs()
    pv = bop * volume
    if length > 1:
        return ta.ma(kind, close=pv, length=length) / ta.ma(kind, close=volume, length=length)
    return pv


def swbop(df: DataFrame, length=14, kind='sma'):
    """
    Candle-size-weighted balance of power indicator by @rk.
    :param df:
    :param length:
    :param kind:
    :return:
    """
    bop = ta.bop(df['open'], df['high'], df['low'], df['close'])
    size = df['high'] - df['low']
    pv = bop * size
    return ta.ma(kind, close=pv, length=length) / ta.ma(kind, close=size, length=length)


def vwtp(df: DataFrame, length=14, kind='sma'):
    """
    Volume-weighted typical price.
    :param df:
    :param length:
    :param kind:
    :return:
    """
    tprice = (df['high'] + df['low'] + df['close']) / 3
    volume = df['volume']
    pv = tprice * volume
    return ta.ma(kind, close=pv, length=length) / ta.ma(kind, close=volume, length=length)
