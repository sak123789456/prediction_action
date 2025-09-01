# Fichier: main_app.py (Version avec Métriques Complètes)
# Description: Interface graphique affichant les métriques de classification et de régression.

import sys
import subprocess
import pkg_resources
import tkinter as tk
from tkinter import ttk, font, messagebox
import threading
import queue

from data_handler import get_stock_data
from model_trainer import train_evaluate_and_predict

# ... (SECTION 1: VÉRIFICATION DES DÉPENDANCES - reste identique)
# ... (Le code de cette section ne change pas)
REQUIRED_PACKAGES = {
    'pandas': 'pandas', 'yfinance': 'yfinance', 'scikit-learn': 'sklearn',
    'xgboost': 'xgboost', 'ta': 'ta', 'numpy': 'numpy'
}
def check_and_install_requirements():
    print("⚙️  Vérification des dépendances...")
    missing_packages = []
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.get_distribution(import_name)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package_name)

    if missing_packages:
        print(f"⚠️  {', '.join(missing_packages)} manquant(s). Installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("\n👍 Dépendances installées.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur d'installation : {e}")
            sys.exit(1)
    else:
        print("👍 Toutes les dépendances sont satisfaites.")

# ... (SECTION 2: LOGIQUE D'APPLICATION AVEC THREADING - reste identique)
# ... (Le code de cette section ne change pas)
def run_prediction_task(ticker, q):
    try:
        data = get_stock_data(ticker)
        if data is None:
            raise ValueError(f"Impossible de récupérer les données pour {ticker}.")
        direction, predicted_price, metrics = train_evaluate_and_predict(data)
        q.put(("OK", (ticker, direction, predicted_price, data['Close'].iloc[-1], metrics)))
    except Exception as e:
        q.put(("ERROR", str(e)))

def handle_prediction():
    ticker = entry_ticker.get().upper()
    if not ticker:
        messagebox.showwarning("Entrée invalide", "Veuillez entrer un symbole d'action.")
        return
    predict_button.config(state=tk.DISABLED)
    clear_button.config(state=tk.DISABLED)
    progress_bar.start()
    clear_results(clear_entry=False)
    prediction_queue = queue.Queue()
    threading.Thread(target=run_prediction_task, args=(ticker, prediction_queue)).start()
    root.after(100, check_queue, prediction_queue)

def check_queue(q):
    try:
        status, result = q.get(block=False)
        progress_bar.stop()
        predict_button.config(state=tk.NORMAL)
        clear_button.config(state=tk.NORMAL)
        if status == "OK":
            display_results(*result)
        else:
            messagebox.showerror("Erreur", result)
    except queue.Empty:
        root.after(100, check_queue, q)
        
def clear_results(clear_entry=True):
    if clear_entry:
        entry_ticker.delete(0, tk.END)
    for widget in result_frame.winfo_children():
        widget.destroy()
    entry_ticker.focus()

# MODIFIÉ : La fonction d'affichage inclut maintenant les métriques R2 et RMSE
def display_results(ticker, direction, price, last_close, metrics):
    for widget in result_frame.winfo_children():
        widget.destroy()

    # --- Section Prédiction ---
    ttk.Label(result_frame, text=f"Prédiction pour l'action {ticker}", font=bold_font, style="Header.TLabel").pack(pady=(0, 10))
    color = "#2ECC71" if direction == "HAUSSE" else "#E74C3C"
    ttk.Label(result_frame, text="Tendance prédite :", style="Sub.TLabel").pack(anchor="w")
    ttk.Label(result_frame, text=direction, font=bold_font, foreground=color).pack(anchor="w", pady=(0, 10))
    ttk.Label(result_frame, text="Prix de clôture estimé :", style="Sub.TLabel").pack(anchor="w")
    ttk.Label(result_frame, text=f"${price:.2f}", font=bold_font, style="Header.TLabel").pack(anchor="w", pady=(0, 10))
    ttk.Label(result_frame, text=f"(Dernier prix : ${last_close:.2f})", font=italic_font, style="Italic.TLabel").pack(anchor="w")
    
    ttk.Separator(result_frame, orient='horizontal').pack(fill='x', pady=10, padx=20)
    
    # --- Section Métriques de Classification ---
    ttk.Label(result_frame, text="Performance (Tendance)", font=bold_font, style="Header.TLabel").pack(pady=(0, 5))
    ttk.Label(result_frame, text=f"• Accuracy : {metrics['accuracy']:.2%}", style="Sub.TLabel").pack(anchor="w")
    ttk.Label(result_frame, text=f"• Precision : {metrics['precision']:.2%}", style="Sub.TLabel").pack(anchor="w")
    ttk.Label(result_frame, text=f"• Recall : {metrics['recall']:.2%}", style="Sub.TLabel").pack(anchor="w")
    ttk.Label(result_frame, text=f"• ROC AUC : {metrics['roc_auc']:.3f}", style="Sub.TLabel").pack(anchor="w")

    ttk.Separator(result_frame, orient='horizontal').pack(fill='x', pady=10, padx=20)

    # --- NOUVEAU : Section Métriques de Régression ---
    ttk.Label(result_frame, text="Performance (Prix)", font=bold_font, style="Header.TLabel").pack(pady=(0, 5))
    ttk.Label(result_frame, text=f"• Score R² : {metrics['r2']:.3f}", style="Sub.TLabel").pack(anchor="w")
    ttk.Label(result_frame, text=f"• Erreur Moyenne (RMSE) : ${metrics['rmse']:.4f}", style="Sub.TLabel").pack(anchor="w")

# ... (SECTION 3: INTERFACE GRAPHIQUE - reste identique, j'ai juste augmenté la hauteur)
if __name__ == "__main__":
    check_and_install_requirements()
    root = tk.Tk()
    root.title("Prédiction Boursière v6.0")
    root.geometry("450x720") # Fenêtre plus haute pour les nouvelles métriques
    root.configure(bg="#2E2E2E")
    bold_font = font.Font(family="Segoe UI", size=12, weight="bold")
    italic_font = font.Font(family="Segoe UI", size=9, slant="italic")
    BG_COLOR = "#2E2E2E"
    FG_COLOR = "#EAEAEA"
    FRAME_COLOR = "#3C3C3C"
    ACCENT_COLOR = "#4A90E2"
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 10))
    style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=FG_COLOR, background=FRAME_COLOR)
    style.configure("Sub.TLabel", background=FRAME_COLOR, foreground=FG_COLOR)
    style.configure("Italic.TLabel", background=FRAME_COLOR, foreground="#B0B0B0")
    style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background=ACCENT_COLOR, foreground="white")
    style.map("TButton", background=[('active', '#63a3e6')])
    style.configure("TEntry", fieldbackground="#555555", foreground=FG_COLOR, bordercolor="#666666", insertcolor=FG_COLOR)
    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)
    input_frame = ttk.Frame(main_frame)
    input_frame.pack(fill=tk.X, pady=10)
    ttk.Label(input_frame, text="Symbole de l'action (ex: AAPL, MSFT):").pack(pady=5)
    entry_ticker = ttk.Entry(input_frame, width=30, font=("Segoe UI", 11))
    entry_ticker.pack(pady=5)
    entry_ticker.focus()
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10)
    predict_button = ttk.Button(button_frame, text="🚀 Lancer la prédiction", command=handle_prediction)
    predict_button.pack(side=tk.LEFT, padx=5)
    clear_button = ttk.Button(button_frame, text="Effacer", command=clear_results)
    clear_button.pack(side=tk.LEFT, padx=5)
    progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
    progress_bar.pack(fill=tk.X, padx=10, pady=5)
    result_frame = ttk.Frame(main_frame, padding="15", relief=tk.SOLID, borderwidth=1, style="TFrame")
    result_frame.configure(style="TFrame")
    result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    root.mainloop()