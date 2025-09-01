# Fichier: data_handler.py (Version Corrigée)
# Description: Gère le téléchargement et le formatage des données boursières.

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def get_stock_data(ticker):
    """
    Récupère l'historique des deux dernières années pour une action donnée
    et simplifie les colonnes.
    """
    print(f"📥 Téléchargement des données pour {ticker}...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365)
        
        # yfinance peut créer des colonnes multi-niveaux, surtout avec plusieurs tickers
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, group_by='ticker')
       
        if data.empty:
            print(f"❌ Aucune donnée trouvée pour le symbole '{ticker}'.")
            return None
            
        # NOUVEAU : On supprime le niveau "Ticker" si on a un seul ticker
        if len(data.columns.levels) > 1:
            data.columns = data.columns.droplevel(0)

        print("✅ Données téléchargées et formatées avec succès.")
        return data
        
    except Exception as e:
        print(f"❌ Une erreur imprévue est survenue lors du téléchargement : {e}")
        return None
if __name__ == "__main__":
    ticker = "AAPL"  # Exemple : Apple
    df = get_stock_data(ticker, years=2, interval="1d")

    if not df.empty:
        print(f"\n📊 Aperçu des données pour {ticker} :")
        print(df.head())   # 5 premières lignes
        print("\n✅ Nombre total de lignes :", len(df))
    else:
        print("⚠️ Pas de données téléchargées.")