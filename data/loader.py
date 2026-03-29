import yfinance as yf
import pandas as pd


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données OHLCV ajustées pour un ticker donné.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Correction yfinance récent : aplatir le MultiIndex des colonnes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Nettoyage : supprimer les lignes sans prix de clôture
    df = df.dropna(subset=["Close"])

    # Garder uniquement les colonnes utiles
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'returns' = rendement journalier."""
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    return df


if __name__ == "__main__":
    data = download_data("AAPL", "2020-01-01", "2024-01-01")
    data = compute_returns(data)
    print(data.tail(10))