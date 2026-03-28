import yfinance as yf
import pandas as pd


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données OHLCV ajustées pour un ticker donné.

    Args:
        ticker : symbole boursier, ex. 'AAPL', 'BNP.PA'
        start  : date de début au format 'YYYY-MM-DD'
        end    : date de fin au format 'YYYY-MM-DD'

    Returns:
        DataFrame avec colonnes Open, High, Low, Close, Volume
        et un index DatetimeIndex.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Nettoyage : supprimer les lignes sans prix de clôture
    df = df.dropna(subset=["Close"])

    # Garder uniquement les colonnes utiles
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'returns' = rendement journalier en %."""
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    return df


if __name__ == "__main__":
    # Test rapide
    data = download_data("AAPL", "2020-01-01", "2024-01-01")
    data = compute_returns(data)
    print(data.tail(10))
    print(f"\nShape : {data.shape}")
    print(f"Colonnes : {list(data.columns)}")