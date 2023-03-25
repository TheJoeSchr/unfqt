# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open
from freqtrade.enums import RunMode
from datetime import datetime
from technical.indicators import bollinger_bands, VIDYA
import talib.abstract as ta
import logging
import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pta  # used as dataframe.ta
from pandas import DataFrame
from typing import Dict, List, Optional, Tuple, Union

from unfqt.helper_mixin import HelperMixin

# --------------------------------
# Import Strategy from different file


class CustomTakeprofitMixin(HelperMixin):
    custom_stoploss_config = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    # needs to be overridden
    def custom_takeprofit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        """
        needs to be overriden
        """
        pass

    def is_takeprofit_reached(self, current_profit_pct, takeprofit_price_abs, open_rate_abs, **kwargs) -> bool:
        takeprofit_pct = takeprofit_price_abs / open_rate_abs - 1
        if(current_profit_pct >= takeprofit_pct):
            return True
        return False

    def stoploss_from_open(self, open_relative_stop: float, current_profit: float) -> float:
        return stoploss_from_open(open_relative_stop, current_profit)

    def stoploss_from_absolute(self, current_rate, stoploss_price):
        if stoploss_price >= current_rate:
            return 1
        return (stoploss_price / current_rate) - 1


class WaitForTakeProfitStoploss(CustomTakeprofitMixin):
    custom_stoploss_config = {
        'risk_reward_ratio': np.NaN,
        'atr_ratio': np.NaN,
        'min_rollback': np.NaN,
        'set_to_break_even_at_profit': np.NaN
    }
    stoploss = -0.30
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        custom_stoploss using a risk/reward ratio
        """
        result = DO_NOTHING
        super().custom_stoploss(pair, trade, current_time, current_rate, current_profit)
        # get initial stoploss at trade.open_date
        trade_candle = self.get_trade_candle(trade)
        if not trade_candle.empty:
            if self.is_takeprofit_reached(current_profit,
                                          trade_candle['takeprofit_rate'], trade.open_rate):
                result = self.custom_takeprofit(pair, trade, current_time,
                                                current_rate, current_profit, **kwargs)

        return result

    def __init__(self, config: dict) -> None:
        super().__init__(config)


# set as close as possible to current price to trigger emergency sell on exchange
FORCE_SELL = 0.0000001
DO_NOTHING = 1  # no new stoploss gets set


class LockSellUntilTakeprofitMixin(WaitForTakeProfitStoploss):
    use_sell_signal = True

    def custom_takeprofit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        result = DO_NOTHING
        current_candle = self.find_candle_datetime(
            current_time, trade.pair, now=current_time)
        if not current_candle.empty:
            should_sell = current_candle['sell'] > 0 if 'sell' in current_candle else False
            if(should_sell):
                result = FORCE_SELL
        return result

    def __init__(self, config: dict) -> None:
        super().__init__(config)


class SetTrailingStoplossMixin(LockSellUntilTakeprofitMixin):
    use_sell_signal = True

    def custom_takeprofit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        result = super().custom_takeprofit(pair, trade, current_time,
                                           current_rate, current_profit, **kwargs)

        # check if we already got a sell signal
        if(abs(result) != DO_NOTHING):
            return result

        current_candle = self.find_candle_datetime(
            current_time, trade.pair, now=current_time)
        if not current_candle.empty:
            # calculate current trailing stoploss
            if 'trailing_stoploss_rate' in current_candle:
                result = self.stoploss_from_absolute(
                    current_rate, current_candle['trailing_stoploss_rate'])

        return result

    def __init__(self, config: dict) -> None:
        super().__init__(config)


class SetBreakEvenWhenTrailingStoplossIsNotSet(SetTrailingStoplossMixin):
    use_sell_signal = True

    def custom_takeprofit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        result = super().custom_takeprofit(pair, trade, current_time,
                                           current_rate, current_profit, **kwargs)

        break_even_pct = (trade.fee_open + trade.fee_close)
        stoploss_break_even = self.stoploss_from_open(
            break_even_pct, current_profit)
        # break_even maybe still to big if takeprofit is very tiny
        # at this point return 0 would not set a new stoploss
        if(stoploss_break_even == 0):
            return result
        # super custom_takeprofit either did nothing
        # or set stoploss below break_even_pct
        # so we replace it with break_even_pct
        if(abs(result) == DO_NOTHING or abs(result) > abs(stoploss_break_even)):
            result = stoploss_break_even
        return result

    def __init__(self, config: dict) -> None:
        super().__init__(config)


class TakeprofitTrailingByATRMixin(CustomTakeprofitMixin):
    custom_stoploss_config = {
        'takeprofit_atr_ratio': 3,
        'min_rollback': 3,
        'trailing_atr_ratio': 1,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['min'] = dataframe['low'].rolling(
            self.custom_stoploss_config['min_rollback']).min()
        dataframe['atr'] = dataframe.ta.atr()
        dataframe['takeprofit_rate'] = (
            dataframe['atr'] * self.custom_stoploss_config['takeprofit_atr_ratio']) + dataframe['low']
        dataframe['trailing_stoploss_rate'] = dataframe['min'] - \
            (dataframe['atr']*self.custom_stoploss_config['trailing_atr_ratio'])
        return dataframe

    def __init__(self, config: dict) -> None:
        super().__init__(config)


class TrailingByInversedPSARMixin(CustomTakeprofitMixin):
    custom_stoploss_config = {
        'takeprofit_atr_ratio': 3,
        'min_rollback': 3,
        'trailing_atr_ratio': 1,
        'risk_reward_ratio': 2
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        if 'psar' not in dataframe.columns:
            dataframe['psar'] = ta.SAR(dataframe)

        dataframe['psar_inverse'] = dataframe.psar.where(~(dataframe['high'] < dataframe['psar']) & (
            dataframe.psar.shift() < dataframe.low.shift()), dataframe.psar.shift()-(dataframe.psar-dataframe.psar.shift()))
        dataframe['min'] = dataframe['low'].rolling(
            self.custom_stoploss_config['min_rollback']).min()
        dataframe['atr'] = dataframe.ta.atr()
        atr_trail = dataframe['min'] - \
            (dataframe['atr']*self.custom_stoploss_config['trailing_atr_ratio'])
        dataframe.psar_inverse.where(dataframe.psar_inverse <
                                     dataframe.low, atr_trail, inplace=True)
        dataframe.psar_inverse.ffill(inplace=True)
        dataframe['trailing_stoploss_rate'] = dataframe['psar_inverse']
        dataframe['stoploss_rate'] = dataframe['min'] - \
            (dataframe['psar_inverse'])
        dataframe['takeprofit_rate'] = ((dataframe['close']-dataframe['stoploss_rate'])
                                        * self.custom_stoploss_config['risk_reward_ratio'])+dataframe['close']
        return dataframe

    def __init__(self, config: dict) -> None:
        super().__init__(config)


DECAY_STOPLOSS = 0.30
