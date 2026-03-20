
import pandas as pd
import numpy as np

import requests
from datetime import datetime, timedelta

# --- CONFIGURATION ---



def get_macro_data(START_DATE, END_DATE, dow):
    API_KEY = 'ac6f670ee74102e4f03870510072acb9'
    SERIES_IDS_MACRO = [ 'DFF', 'DGS1MO','DGS3MO','DGS1', 'DGS2','DGS5', 'DGS10', 'AAA', 'MORTGAGE30US','IRLTLT01JPM156N']

    # S&P 500, VIX, High Yield Spread (not the Yield), Oil, Oil VIX, Gold VIX, Chicago Fed Financial Conditions, Dollar Index, KRW
    SERIES_IDS_RISK = ['SP500', 'VXVCLS', 'BAMLH0A0HYM2', 'DCOILWTICO', 'OVXCLS', 'GVZCLS', 'NFCI', 'DTWEXBGS', 'DEXKOUS']
    def fetch_series(series_id, api_key, start_date, end_date):
        """Fetch FRED series observations as a DataFrame."""
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
            f"&observation_start={start_date}&observation_end={end_date}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        obs = resp.json()['observations']
        df = pd.DataFrame(obs)[['date', 'value']]
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna()
        df = df.set_index('date')
        df = df.rename(columns={'value': series_id})
        return df

    # Fetch all series and join on date
    frames = []
    for series_id in SERIES_IDS_MACRO:
        print(f"Fetching {series_id}...")
        df = fetch_series(series_id, API_KEY, START_DATE, END_DATE)
        frames.append(df)

    data_macro = pd.concat(frames, axis=1)
    data_macro=data_macro.ffill().bfill()

    frames = []
    for series_id in SERIES_IDS_RISK:
        print(f"Fetching {series_id}...")
        df = fetch_series(series_id, API_KEY, START_DATE, END_DATE)
        frames.append(df)

    data_risk = pd.concat(frames, axis=1)
    data_risk=data_risk.ffill().bfill()

    # --- Align to weekly Wed ---
    # Create a range of Weds from start to end
    thursdays = pd.date_range(start=START_DATE, end=END_DATE, freq=dow)

    # For each Wed, get the most recent available value (including Wed)
    def get_most_recent(row, df):
        # Find the last available date <= row.name
        prior_dates = df[df.index <= row.name]
        if prior_dates.empty:
            return pd.Series([pd.NA] * len(SERIES_IDS), index=SERIES_IDS)
        return prior_dates.iloc[-1]

    weekly_data_macro = pd.DataFrame(index=thursdays, columns=SERIES_IDS_MACRO)
    weekly_data_risk = pd.DataFrame(index=thursdays, columns=SERIES_IDS_RISK)


    for series_id in SERIES_IDS_MACRO:
        s = data_macro[series_id].dropna()
        weekly_data_macro[series_id] = [
            s.loc[:date].iloc[-1] if not s.loc[:date].empty else pd.NA
            for date in weekly_data_macro.index
        ]

    for series_id in SERIES_IDS_RISK:
        s = data_risk   [series_id].dropna()
        weekly_data_risk[series_id] = [
            s.loc[:date].iloc[-1] if not s.loc[:date].empty else pd.NA
            for date in weekly_data_risk.index
        ]

    weekly_data_risk.iloc[0] = weekly_data_risk.iloc[0].fillna(data_risk.iloc[0])
    weekly_data_risk=weekly_data_risk.ffill().bfill()

    weekly_data_risk['SP500_diff'] = np.log(weekly_data_risk['SP500'] / weekly_data_risk['SP500'].shift(1))

    weekly_data_risk['Crude_diff'] = np.log(weekly_data_risk['DCOILWTICO'] / weekly_data_risk['DCOILWTICO'].shift(1))
    weekly_data_risk = weekly_data_risk.drop(columns=['SP500', 'DCOILWTICO'])

    # Optional: drop rows with all NAs

    weekly_data_macro = weekly_data_macro.dropna(how='all')
    weekly_data_risk = weekly_data_risk.dropna(how='all')
    # weekly_data_risk.index = weekly_data_risk.index + pd.Timedelta(days=1)
    # weekly_data_macro.index = weekly_data_macro.index + pd.Timedelta(days=1)

    weekly_data_macro.index= weekly_data_macro.index.date
    weekly_data_risk.index = weekly_data_risk.index.date

    return weekly_data_macro, weekly_data_risk
