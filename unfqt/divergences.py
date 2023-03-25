from typing import List, Union
import numpy as np
from pandas import DataFrame, Series
from unfqt.strat_utils import valuewhen
from detecta import detect_peaks


def find_divergence(is_low_peak: Series,
                    price: Series,
                    osc: Series,
                    lookback: int = 1):

    # bullish regular divergence (price LL vs osc HL)
    osc_higher_low = osc > valuewhen(is_low_peak, osc, lookback)
    price_lower_low = price < valuewhen(is_low_peak, price, lookback)

    found_regular_divergence_mask = osc_higher_low & \
        price_lower_low & is_low_peak

    # hidden divergence (price HL vs osc LL)
    osc_lower_low = osc < valuewhen(is_low_peak, osc, lookback)
    price_higher_low = price > valuewhen(is_low_peak, price, lookback)

    found_hidden_divergence_mask = osc_lower_low & \
        price_higher_low & is_low_peak

    # [:-1]: is needed, because is_low_peak is
    # using duplicated last candle for detect_peaks
    # TODO: refactor `[:-1]` to level above, so that logic isn't split.
    regular = np.where(found_regular_divergence_mask[:-1], 1, np.NaN)
    hidden = np.where(found_hidden_divergence_mask[:-1], 1, np.NaN)
    return regular, hidden


def find_bullish_div(dataframe,
                     price_label='low',
                     osc_label='rsi',
                     *,
                     label_regular="bullish_regular_divergence",
                     label_hidden="bullish_hidden_divergence",
                     lookback: int = 1):
    dataframe = dataframe.copy()
    regular = Series(data=None, index=dataframe.index, dtype=np.float64)
    hidden = Series(data=None, index=dataframe.index, dtype=np.float64)
    # we need to duplicate last entries of series
    # to enable detect_peak to work with last candle
    price = Series([*dataframe[price_label], dataframe[price_label].iat[-1]])
    osc = Series([*dataframe[osc_label], dataframe[osc_label].iat[-1]])
    # find peaks and re-append index
    is_low_peak = price.iloc[detect_peaks(
        price, valley=True, show=False)] != np.NaN

    for lookback_i in range(0, lookback+1):
        reg, hid = find_divergence(is_low_peak, price, osc, lookback_i)
        regular = Series(regular).combine_first(Series(reg))
        hidden = Series(hidden).combine_first(Series(hid))

    dataframe[label_regular] = regular
    dataframe[label_hidden] = hidden
    return regular, hidden, dataframe


def combine_bullish_divergences(dataframe,
                                price_label: str = 'low',
                                oscillators: List[str] = ['rsi'],
                                lookback: int = 1):
    regular = Series(data=0, index=dataframe.index, dtype=np.float64)
    hidden = Series(data=0, index=dataframe.index, dtype=np.float64)
    for osc in oscillators:
        reg, hid, _ = find_bullish_div(
            dataframe, price_label, osc, lookback=lookback)
        regular = regular + reg.fillna(0)
        hidden = hidden + hid.fillna(0)
    return regular, hidden
