import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate():
    print("1. Wczytywanie surowych logów L1...")
    df = pd.read_csv('sionna_professional_dataset.csv')

    # 2. Przygotowanie etykiet (Labeling) - szukamy optymalnego wyboru
    # Dla każdego poziomu SNR znajdujemy wiersz, w którym Throughput był najwyższy
    idx_optimal = df.groupby(['SNR_dB'])['Throughput'].idxmax()
    optimal_df = df.loc[idx_optimal].reset_index(drop=True)

    print("\nOto próbka optymalnych decyzji (nasz Ground Truth dla AI):")
    print(optimal_df[['SNR_dB', 'Modulation', 'Throughput']].head(10))

    # 3. Definicja wejścia (X) i wyjścia (y) dla modelu AI
    X = optimal_df[['SNR_dB']] # Cecha: Mierzony poziom szumu
    y = optimal_df['Bits_Per_Symbol'] # Target: Decyzja o modulacji (2, 4, lub 6)

    # 4. Trening Modelu AI (Drzewo Decyzyjne)
    print("\n2. Trenowanie modelu ML...")
    clf = DecisionTreeClassifier(max_depth=3) # Płytkie drzewo - łatwe do zaimplementowania w C++ (5gmax)
    clf.fit(X, y)

    # Ocena modelu
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Dokładność modelu AI: {accuracy * 100:.2f}%\n")

    # 5. Rysowanie profesjonalnego wykresu na studia
    print("3. Generowanie wykresu...")
    plt.figure(figsize=(10, 6))

    # Rysujemy surowe krzywe dla każdej modulacji z oryginalnego datasetu
    modulations = df['Bits_Per_Symbol'].unique()
    colors = {2: 'blue', 4: 'orange', 6: 'green'}
    names = {2: '4-QAM', 4: '16-QAM', 6: '64-QAM'}

    for mod in modulations:
        subset = df[df['Bits_Per_Symbol'] == mod]
        plt.plot(subset['SNR_dB'], subset['Throughput'],
                 label=names[mod], color=colors[mod], alpha=0.6, linestyle='--')

    # Rysujemy optymalną obwiednię, którą wybrało nasze AI
    plt.plot(optimal_df['SNR_dB'], optimal_df['Throughput'],
             label='Wybor Optymalny (Ground Truth)', color='black', linewidth=3)

    # Zaznaczamy tło kolorami, żeby pokazać jak AI podzieliło przestrzeń decyzyjną
    snr_test = np.linspace(df['SNR_dB'].min(), df['SNR_dB'].max(), 500).reshape(-1, 1)
    predictions = clf.predict(pd.DataFrame(snr_test, columns=['SNR_dB']))

    for mod in modulations:
        plt.fill_between(snr_test.flatten(), 0, 7,
                         where=(predictions == mod),
                         color=colors[mod], alpha=0.1)

    plt.title('Zarządzanie Zasobami Radiowymi (Link Adaptation) z użyciem AI')
    plt.xlabel('Stosunek sygnału do szumu (SNR) [dB]')
    plt.ylabel('Przepływność (Throughput) [bits/symbol]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('link_adaptation_ai_plot.png')
    print("Gotowe! Otwórz plik 'link_adaptation_ai_plot.png', aby zobaczyć efekty.")

if __name__ == '__main__':
    train_and_evaluate()
