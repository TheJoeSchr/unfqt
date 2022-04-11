import pandas as pd

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import IStrategy
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

from functools import wraps

# anti-future bias wrapper
def analyze_df_in_batches(f, window=500):
    @wraps(f)
    def wrapper(df, *args, **kwargs):
        lines = []
        for i in range(len(df) - window):
            slice = df[i : i + window]
            res = f(slice, *args, **kwargs)
            lines.append(res.iloc[[-1]])
        return pd.concat(lines)

    return wrapper


def get_trade_opened_candle(self, trade: "Trade"):
    """
    search for nearest row of trade.open_date
    """
    trade_candle = find_candle_near_datetime(self, trade.open_date_utc, pair=trade.pair)
    return trade_candle


def get_buy_signal_candle(self, trade: "Trade", timeframe="5m"):
    """
    search for nearest buy candle next to trade.open_date
    """
    buy_candle = find_candle_near_datetime(
        self,
        trade.open_date_utc - timedelta(minutes=timeframe_to_minutes(self.timeframe)),
        pair=trade.pair,
    )
    return buy_candle


def find_candle_near_datetime(self, query_date: datetime, pair: str):
    result = None
    # get dataframe
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    candle = search_dataframe_index(
        self,
        query_date,
        dataframe,
    )
    result = candle if candle.empty else candle.squeeze()
    return result


def search_dataframe_index(self, query_date: datetime, dataframe):
    df = dataframe[["date"]].set_index("date")
    try:
        date_mask = df.index.unique().get_loc(query_date, method="ffill")
        candle = dataframe.iloc[date_mask]  # use iloc because date_mask :int
    except KeyError:  # trade.open_date may not exist if trade hasn't been opened yet
        candle = pd.DataFrame(index=dataframe.index)
    return candle


def find_candle_datetime_faster(self, query_date: datetime, now: datetime, dataframe):
    """
    fast, but looks into future,
    if dataframe hasn't been sanitzed
    via e.g. using self.dp.get_analyzed_dataframe
    """

    if now and now == query_date:
        candle = dataframe.iloc[-1]
    else:
        candle_date = timeframe_to_prev_date(self.timeframe, query_date)
        candle = dataframe.loc[dataframe.date == candle_date]
    return candle
