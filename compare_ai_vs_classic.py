import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def compare_algorithms():
    print("Ładowanie danych do ostatecznego starcia...")
    df = pd.read_csv('sionna_realistic_dataset.csv')
    avg_df = df.groupby(['SNR_dB', 'Channel', 'Speed_kmph', 'Chosen_Modulation_Bits'])['Actual_Throughput'].mean().reset_index()

    # --- 1. SZYBKI TRENING AI ---
    idx_opt = avg_df.groupby(['SNR_dB', 'Channel', 'Speed_kmph'])['Actual_Throughput'].idxmax()
    opt_df = avg_df.loc[idx_opt].reset_index(drop=True)

    le = LabelEncoder()
    opt_df['Channel_Code'] = le.fit_transform(opt_df['Channel'])
    avg_df['Channel_Code'] = le.transform(avg_df['Channel'])

    X = opt_df[['SNR_dB', 'Channel_Code', 'Speed_kmph']]
    y = opt_df['Chosen_Modulation_Bits']
    ai_model = DecisionTreeClassifier(max_depth=4)
    ai_model.fit(X, y)

    # --- 2. KLASYCZNY ALGORYTM (Tabela LUT oparta tylko na SNR) ---
    # Symulujemy sztywne progi zoptymalizowane dla "idealnych" warunków (TDL-A, 3km/h)
    def classic_lut_algorithm(snr):
        if snr < 4: return 2       # Poniżej 4dB -> 4-QAM
        elif snr < 12: return 4    # 4dB - 12dB -> 16-QAM
        else: return 6             # Powyżej 12dB -> 64-QAM

    # --- 3. WALKA NA NAJTRUDNIEJSZYM KANALE (TDL-C, 120 km/h) ---
    hard_channel = 'TDL-C'
    hard_speed = 120.0
    ch_code = le.transform([hard_channel])[0]

    test_snrs = np.arange(-5, 25, 1)

    ai_throughput = []
    classic_throughput = []

    # Odpytujemy oba algorytmy dla każdego poziomu SNR
    for snr in test_snrs:
        # Decyzja Klasycznego Algorytmu
        classic_choice = classic_lut_algorithm(snr)
        # Decyzja AI
        ai_choice = ai_model.predict([[snr, ch_code, hard_speed]])[0]

        # Sprawdzamy w danych, jaki był RZECZYWISTY throughput dla tych wyborów
        # (Czyli jaka była kara za złą decyzję)
        row_classic = avg_df[(avg_df['SNR_dB'] == snr) & (avg_df['Channel'] == hard_channel) &
                             (avg_df['Speed_kmph'] == hard_speed) & (avg_df['Chosen_Modulation_Bits'] == classic_choice)]
        row_ai = avg_df[(avg_df['SNR_dB'] == snr) & (avg_df['Channel'] == hard_channel) &
                        (avg_df['Speed_kmph'] == hard_speed) & (avg_df['Chosen_Modulation_Bits'] == ai_choice)]

        # Jeśli jakiegoś wyboru nie ma w danych, przepustowość = 0
        classic_tp = row_classic['Actual_Throughput'].values[0] if not row_classic.empty else 0
        ai_tp = row_ai['Actual_Throughput'].values[0] if not row_ai.empty else 0

        classic_throughput.append(classic_tp)
        ai_throughput.append(ai_tp)

    # --- 4. WYWALAMY WYKRESY ---
    plt.figure(figsize=(12, 7))
    plt.plot(test_snrs, classic_throughput, label='Klasyczny Algorytm (LUT)', color='red', linewidth=3, marker='x')
    plt.plot(test_snrs, ai_throughput, label='Sztuczna Inteligencja (AI)', color='green', linewidth=3, marker='o')

    # Obszary gdzie AI zdeklasowało klasyczny kod
    plt.fill_between(test_snrs, classic_throughput, ai_throughput,
                     where=(np.array(ai_throughput) > np.array(classic_throughput)),
                     color='lightgreen', alpha=0.3, label='Zysk z AI (Więcej Mbps!)')

    # Obszary gdzie Klasyczny algorytm "umarł" przez chciwość
    plt.fill_between(test_snrs, classic_throughput, ai_throughput,
                     where=(np.array(classic_throughput) == 0) & (np.array(ai_throughput) > 0),
                     color='red', alpha=0.1, label='Śmierć pakietów (BLER 100%) - Algorytm Klasyczny')

    plt.title(f'Starcie: AI vs Klasyczny Algorytm L1/MAC\nEkstremalne Warunki: {hard_channel}, {hard_speed} km/h', fontsize=16)
    plt.xlabel('Poziom Sygnału (SNR) [dB]', fontsize=14)
    plt.ylabel('Rzeczywista Przepustowość (Throughput)', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('AI_vs_Classic_Showdown.png')
    print("BOOM! Wykres zapisany jako 'AI_vs_Classic_Showdown.png'. Odpal go!")

if __name__ == '__main__':
    # Wyłączamy ostrzeżenia scikit-learn o nazwach cech (dla czystości konsoli)
    import warnings
    warnings.filterwarnings("ignore")
    compare_algorithms()
