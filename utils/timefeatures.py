from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self, steps_per_day=144):  # 添加 steps_per_day 参数
        self.steps_per_day = steps_per_day

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [0, 1]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [0, 1]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [0, 1)"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 使用传入的 steps_per_day 来计算
        tod = [i % self.steps_per_day / self.steps_per_day for i in range(len(index))]
        tod = np.array(tod)
        return tod


class DayOfWeek(TimeFeature):
    """Day of the week encoded as value between [0, 1)"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 7.0 


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [0, 1]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 31.0 


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [0, 1]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 366.0 


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [0, 1]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 12.0 


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [0, 1]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 
    
class SeasonOfYear(TimeFeature):
    """Season of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        #return (index.month // 3) / 3.0 - 0.5
        return (((index.month // 3) + 3) % 4) / 3.0 - 0.5


def time_features_from_frequency_str(freq_str: str, steps_per_day=24) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            # 在这里传递 steps_per_day 参数
            return [cls(steps_per_day) for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h', steps_per_day=144):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq, steps_per_day)])

