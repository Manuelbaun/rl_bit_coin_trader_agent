import pandas as pd
from pathlib import Path
from datetime import datetime


current_dir = Path(Path(__file__).resolve()).parent
data_paths = current_dir / "data"


def get_stock_data(filename):
    df = pd.read_csv(data_paths / filename)

    # d2 = df.set_index('Date').diff().shift();
    # print(d2)

    df["Date"] = pd.to_datetime(df["Date"])
    # date = datetime.strptime(d, "%m/%d/%y %H:%M:%S")
    df["year"] = pd.DatetimeIndex(df["Date"]).year
    df["month"] = pd.DatetimeIndex(df["Date"]).month
    df["day_of_year"] = pd.DatetimeIndex(df["Date"]).dayofyear
    df["day_of_week"] = pd.DatetimeIndex(df["Date"]).dayofweek
    df["week"] = pd.DatetimeIndex(df["Date"]).week
    df["week_day"] = pd.DatetimeIndex(df["Date"]).weekday
    df["week_of_year"] = pd.DatetimeIndex(df["Date"]).weekofyear
    df["hour"] = pd.DatetimeIndex(df["Date"]).hour
    df["minute"] = pd.DatetimeIndex(df["Date"]).minute
    df["second"] = pd.DatetimeIndex(df["Date"]).second

    df = df.drop(["Date"], axis=1)

    return df


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []

    lines = open(key, "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec


def getBitCoinData(filename):
    df = pd.read_csv(data_paths / filename, header=0, encoding="ascii", skipinitialspace=True)
    df.info()
    df["Date"] = pd.to_datetime(df["Date"])

    # created an extra index column => why?
    df = df.sort_values(by="Date", ascending=True).reset_index()
    df.set_index("Date", inplace=True)
    df = df.drop(["Unix Timestamp", "Symbol", "index"], axis=1)

    return df
