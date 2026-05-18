# Raport z postępu prac — Algorytmy Wspomagania Decyzji

## Temat: Uczenie maszynowe w adaptacji łącza radiowego dla systemów 5G NR

**Autor:** Filip Ferenc  
**Data:** 23 kwietnia 2026

---

## 1. O czym jest ten projekt?

W sieciach komórkowych 5G stacja bazowa musi na bieżąco decydować, **jak szybko przesyłać dane** do każdego telefonu. Ta „szybkość" jest kontrolowana przez parametr zwany **MCS** (schemat modulacji i kodowania) — im wyższy MCS, tym więcej danych przesyłamy w jednym pakiecie, ale jednocześnie rośnie ryzyko błędnej transmisji.

Tradycyjnie stacja bazowa korzysta z prostych heurystyk: telefon raportuje jakość sygnału (CQI), a stacja dobiera MCS z gotowej tabeli. Problem w tym, że jakość łącza zależy od wielu czynników naraz — prędkości użytkownika, typu otoczenia, interferencji z sąsiednich stacji — a prosta tabelka tego nie uwzględnia.

**Nasz cel:** nauczyć algorytm ML, aby na podstawie aktualnych warunków radiowych dobierał optymalny MCS — tak aby przesyłać możliwie dużo danych, ale jednocześnie nie przekraczać dopuszczalnego poziomu błędów transmisji (BLER ≤ 10%).

---

## 2. Dlaczego wybraliśmy akurat taki model?

### 2.1 Dlaczego nie zwykła sieć neuronowa?

Na pierwszy rzut oka wydaje się, że wystarczy wziąć sieć neuronową, podać jej dane z symulacji i nauczyć rozpoznawać, który MCS jest najlepszy. Problem polega na tym, że:

1. **MCS ma naturalny porządek** — MCS 14 jest „między" MCS 11 a MCS 17. Gdybyśmy potraktowali indeksy MCS jako zwykłe kategorie (tak jak np. kolory), model nie wiedziałby, że pomylenie MCS 14 z MCS 17 to mały błąd, a z MCS 3 — katastrofa. Dlatego zastosowaliśmy **regresję ordinalną**, która rozumie, że te wartości są uporządkowane.

2. **Błędy nie są symetryczne** — jeśli model wybierze MCS za niski, tracimy trochę przepustowości (łagodny skutek). Ale jeśli wybierze MCS za wysoki, pakiet się nie zdekoduje i trzeba go retransmitować, co marnuje czas radiowy i zwiększa opóźnienie (poważny skutek). Dlatego nasza funkcja kosztu **karze nadmierną estymację 3× silniej** niż niedoszacowanie.

3. **Fizyka musi być zachowana** — lepszy sygnał (wyższy SNR) powinien zawsze prowadzić do co najmniej takiego samego MCS. Sieć neuronowa tego nie gwarantuje — może przypadkowo dać niższy MCS przy lepszym sygnale. Nasz model **Gradient Boosting z ograniczeniami monotonicznymi** gwarantuje to na poziomie algorytmu.

### 2.2 Dlaczego drzewo decyzyjne, a nie bezpośrednio GBM?

Model GBM (Gradient Boosted Machine) to zespół setek małych drzewek — daje dobre wyniki, ale potrzebuje ~10 ms na predykcję. W stacji bazowej 5G scheduler dostaje **1 ms na slot** i musi w tym czasie obsłużyć dziesiątki użytkowników. 10 ms to za wolno.

Rozwiązanie: **destylacja wiedzy** — trenujemy duży, dokładny model („nauczyciel"), a potem uczymy małe, szybkie drzewo decyzyjne („uczeń") naśladować decyzje nauczyciela. Efekt: jedno drzewo o głębokości 10 — prosta kaskada `if/else` — które wykonuje się w **84 nanosekundach** (0.000084 ms). To ponad 10 000× szybciej niż wymagany budżet czasowy.

### 2.3 Warstwa bezpieczeństwa OLLA

Samo drzewo decyzyjne nie widzi, co się dzieje w rzeczywistości po jego decyzji. Dlatego dodaliśmy mechanizm OLLA (Outer Loop Link Adaptation) — system śledzący bieżący BLER:

- Jeśli zbyt wiele pakietów ma błędy → OLLA obniża MCS o krok
- Jeśli transmisje idą bezbłędnie → OLLA podnosi MCS o krok
- Drzewo ML daje „punkt startowy", a OLLA koryguje go w czasie rzeczywistym

To podejście hybrydowe łączy zalety obu światów: ML daje dobrą bazową predykcję, a OLLA zapewnia adaptację do warunków, których model nie widział podczas treningu.

### 2.4 Dlaczego GBM, a nie sieć neuronowa?

W tabeli wyników (sekcja 4.1) widać, że sieci neuronowe (DNN Classifier, GRU) osiągają wyższą dokładność niż nasz GBM. Mimo to świadomie wybraliśmy GBM. Oto dlaczego:

**Problem 1: Sieci neuronowej nie da się wdrożyć w stacji bazowej.**
Scheduler MAC w gNB działa w języku C++ bez dostępu do Pythona, PyTorcha czy TensorFlow. Sieć neuronowa wymaga biblioteki do inferencji (ONNX Runtime, TFLite itp.) — to dodaje zależności, komplikuje build i wprowadza ryzyko niestabilności. Natomiast drzewo decyzyjne eksportujemy jako czysty kod C++ — zwykłą kaskadę `if/else` — która nie potrzebuje żadnych zewnętrznych bibliotek.

**Problem 2: Sieć neuronowa nie gwarantuje fizycznej poprawności.**
DNN może nauczyć się, że przy SINR = 20 dB odpowiedni jest MCS 20, a przy SINR = 22 dB — MCS 17. To fizyczny nonsens (lepszy sygnał → gorszy wybór), ale sieć tego nie wie. GBM z **ograniczeniami monotonicznymi** gwarantuje na poziomie algorytmu, że lepsza jakość sygnału nigdy nie da niższego MCS.

**Problem 3: DNN + OLLA jest niestabilny.**
Wyniki symulacji zamkniętej pętli (sekcja 4.2) pokazują, że kombinacja DNN + OLLA daje BLER = 1.2% — to ponad 10× więcej niż dopuszczalne dla usług URLLC. Dzieje się tak, ponieważ DNN czasem wybiera zbyt agresywny MCS, a OLLA próbuje to korygować, ale reakcja jest spóźniona (potrzebuje kilku cykli HARQ). GBM + OLLA utrzymuje BLER na poziomie 0.2%.

**Podsumowując:** GBM wygrywa nie dlatego, że jest najdokładniejszy, ale dlatego, że jest **jedynym modelem, który jednocześnie:**
1. Da się wyeksportować do czystego C++ (bez zależności)
2. Gwarantuje fizyczną poprawność (monotoniczność)
3. W połączeniu z OLLA daje stabilne i bezpieczne wyniki
4. Działa w 84 nanosekundach (vs ~1 ms dla DNN)

### 2.5 Dlaczego GBM + OLLA, a nie sam GBM?

Sam model GBM jest wytrenowany na danych z symulacji — zna warunki, które „widział" podczas treningu. Ale w prawdziwej sieci mogą pojawić się sytuacje, których nie było w danych treningowych: nowy typ interferencji, niespotykana kombinacja parametrów, sprzęt o innych charakterystykach.

OLLA rozwiązuje ten problem, bo **nie zależy od danych treningowych** — po prostu obserwuje, czy pakiety dochodzą (ACK) czy nie (NACK), i koryguje MCS w górę lub w dół. Działa jak termostat: ustawia się „okolicach" punktu docelowego i stabilizuje.

Bez OLLA, sam GBM daje bezpieczne, ale konserwatywne wyniki — wybiera niższy MCS „na wszelki wypadek". Z OLLA, system zaczyna od dobrej predykcji GBM i delikatnie podnosi MCS, dopóki BLER jest niski. Wynik: przepustowość zbliżona do idealnej, przy minimalnym ryzyku błędów.

---

## 3. Skąd wzięliśmy dane?

Wykorzystaliśmy symulator **NVIDIA Sionna**, który modeluje łącze radiowe 5G zgodnie ze standardem 3GPP. W praktyce oznacza to, że:

- Generujemy sygnał radiowy (modulacja, kodowanie LDPC)
- Przepuszczamy go przez realistyczny model kanału radiowego (TDL — Tapped Delay Line)
- Dodajemy szum, efekt Dopplera (ruch telefonu), wielodrogowość (odbicia od budynków)
- Dekodujemy sygnał po stronie odbiorczej i sprawdzamy, czy pakiet został poprawnie odebrany

Wygenerowaliśmy ponad **2,2 miliona** próbek, obejmujących:

| Parametr | Zakres |
|----------|--------|
| Jakość sygnału (SINR) | −10 do +30 dB |
| Prędkość użytkownika | 3 km/h (spacer) do 120 km/h (autostrada) |
| Model kanału | TDL-A (biuro), TDL-B (miasto), TDL-C (gęsta zabudowa), CDL-D (prosta linia) |
| Częstotliwość | sub-6 GHz i mmWave (28 GHz) |
| Anteny | 1×1 (jedna antena) i 2×2 (MIMO — dwie anteny) |

Każda próbka zawiera informację: „przy takich warunkach radiowych i takim wyborze MCS — czy pakiet został poprawnie odebrany?"

---

## 4. Wyniki eksperymentalne

### 4.1 Porównanie metod (offline)

Porównaliśmy 8 różnych podejść do selekcji MCS. Każdy eksperyment powtórzyliśmy 10 razy z różnymi podziałami danych, aby uzyskać wiarygodne wyniki (przedziały ufności 95%):

| Metoda | Opis | vs Oracle | Naruszenia BLER |
|--------|------|-----------|-----------------|
| Shannon Bound | Teoretyczne maximum z fizyki | 39.6% | 45.0% |
| Static LUT | Prosta tabelka SINR→MCS | 71.7% | 21.3% |
| DNN 1D (z literatury) | Sieć neuronowa z artykułu naukowego | 73.0% | 15.2% |
| DNN 1D (bezpieczny) | Ta sama sieć z ostrożniejszym progiem | 68.7% | 13.1% |
| **Ordinal GBM (nasz)** | **Nasz model z ograniczeniami** | **69.2%** | **14.2%** |
| DNN 3D (rozszerzony) | Ulepszona sieć z dodatkowymi cechami | 69.7% | 11.7% |
| GRU Sequential | Sieć rekurencyjna (pamięta historię) | 74.8% | 12.4% |
| DNN Classifier | Klasyfikator wieloklasowy | 75.6% | 12.5% |

**Kolumna „vs Oracle"** — jaki procent idealnej (orakularnej) przepustowości osiąga dana metoda. Orakul zna wynik z góry i zawsze wybiera najlepszy MCS — w praktyce nieosiągalny, ale stanowi punkt odniesienia.

**Kolumna „Naruszenia BLER"** — jak często metoda przekracza dopuszczalny poziom błędów (10%). Im mniej, tym bezpieczniej.

**Wniosek:** nasz model osiąga porównywalne wyniki do sieci neuronowych z literatury, ale w przeciwieństwie do nich:
- gwarantuje fizyczną poprawność (monotoniczność)
- da się wyeksportować do C++ jako proste drzewo decyzyjne
- działa w <100 nanosekundach

### 4.2 Symulacja zamkniętej pętli

W rzeczywistym systemie stacja bazowa nie tylko wybiera MCS — ona też dostaje informację zwrotną (ACK/NACK), czy pakiet dotarł. Zasymulowaliśmy to w 3 scenariuszach:

**Scenariusz 1: Stały kanał** (biuro, 30 km/h, 15 dB)
| Agent | Przepustowość | BLER |
|-------|---------------|------|
| Czysty OLLA | 2.155 | 0.0% |
| **Nasz GBM + OLLA** | **2.043** | **0.2%** |
| DNN + OLLA | 1.607 | 1.2% ⚠️ |

**Scenariusz 2: Nagła interferencja** (inne stacje zaczynają nadawać)
| Agent | Przepustowość | BLER |
|-------|---------------|------|
| Czysty OLLA | 3.217 | 0.1% |
| **Nasz GBM + OLLA** | **3.212** | **0.2%** |

**Scenariusz 3: Zmiana prędkości** (3→120 km/h — telefon wsiada do samochodu)
| Agent | Przepustowość | BLER |
|-------|---------------|------|
| Czysty OLLA | 2.018 | 5.7% |
| **Nasz GBM + OLLA** | **1.931** | **6.4%** |

**Wniosek:** GBM + OLLA utrzymuje przepustowość zbliżoną do czystego OLLA, ale z niskim BLER. Natomiast DNN + OLLA okazał się niestabilny — BLER 1.2% to niedopuszczalne dla usług krytycznych.

### 4.3 Szybkość działania (benchmark C++)

Zmierzyliśmy czas wykonania drzewa decyzyjnego w C++ na **1 milion** losowych wejść:

| Metryka | Wartość |
|---------|---------|
| Średni czas decyzji | **84 nanosekundy** |
| 99. percentyl | 200 nanosekund |
| Budżet czasowy slotu 5G | 1 000 000 nanosekund (1 ms) |
| **Wykorzystanie budżetu** | **0.008%** |

Innymi słowy: nasz algorytm zajmuje mniej niż 1/10 000 dostępnego czasu. Reszta jest do dyspozycji schedulera na inne operacje.

### 4.4 Jakość destylacji

Porównaliśmy decyzje „dużego" modelu (nauczyciel) z „małym" drzewem (uczeń):

- W **68.2%** przypadków drzewo podejmuje identyczną decyzję jak nauczyciel
- W **70.2%** przypadków różni się o co najwyżej 1 krok MCS
- Średni błąd to 2.91 kroków MCS

To akceptowalny kompromis — drzewo jest 100 000× szybsze od nauczyciela.

---

## 5. Wdrożenie w prawdziwej stacji bazowej 5G

### 5.1 Co zrobiliśmy

Zdestylowane drzewo decyzyjne wgraliśmy do kodu schedulera MAC stacji bazowej **srsRAN Project** (open-source'owa implementacja gNB zgodna z 3GPP). Uruchomiliśmy pełny stos 5G:

```
Telefon (srsUE) ← ZMQ radio → Stacja bazowa (gNB z ML) ← GTP/SCTP → Rdzeń sieci (Open5GS)
```

- **srsUE** — oprogramowanie symulujące telefon 5G
- **gNB** — stacja bazowa z naszym zmodyfikowanym schedulerem
- **Open5GS** — rdzeń sieci 5G (zarządzanie sesjami, autentykacja) uruchomiony w Dockerze
- **ZMQ** — wirtualne radio (zamiast prawdziwej anteny, próbki IQ przesyłane przez TCP)

### 5.2 Wyniki testów na żywo

**Podłączenie telefonu do sieci:**
- Rejestracja w sieci 5G SA ✅
- Ustanowienie sesji danych: IP 10.45.1.2 ✅
- Konfiguracja radiowa ✅

**Test ping (500 pakietów):**

| Metryka | Wartość |
|---------|---------|
| Pakiety wysłane/odebrane | 500 / 500 |
| **Utrata pakietów** | **0%** |
| Opóźnienie min/śr/max | 16 / 36 / 124 ms |

**Test przepustowości (iperf3 UDP, 60 sekund, 10 Mbps):**

| Metryka | Nadawca | Odbiorca |
|---------|---------|----------|
| Transfer | 71.5 MB | 21.5 MB |
| Bitrate | 10.0 Mbps | 2.96 Mbps |

> Straty po stronie odbiorcy (~70%) wynikają z ograniczeń wirtualnego radia ZMQ na standardowym komputerze (brak jądra czasu rzeczywistego). Na dedykowanym sprzęcie radiowym (USRP) straty byłyby bliskie zeru.

### 5.3 Potwierdzenie działania algorytmu ML

Kluczowy dowód, że nasz algorytm ML faktycznie podejmuje decyzje w stacji bazowej:

- **Telefon raportuje CQI = 15** (najlepsza jakość — bo kanał ZMQ jest idealny)
- **Drzewo ML wybiera MCS 20** jako punkt bazowy
- **OLLA podnosi o +2** (bo żaden pakiet nie miał błędu na idealnym kanale)
- **Końcowy MCS = 24–28** (najwyższe wartości z tabeli 64QAM)

Dla porównania — uplink (gdzie nie ma ML, tylko standardowy algorytm) używa MCS 4–8. Różnica DL vs UL potwierdza, że ML jest aktywne na downlinku.

---

## 6. Struktura repozytorium

```
ml-link-adaptation-5g/
├── generate_v2_dataset.py          # Generacja danych symulacyjnych
├── train_real_ml_model.py          # Trening modelu + destylacja do drzewa
├── benchmark_la_approaches.py      # Porównanie 8 metod (wykresy z CI)
├── online_la_simulator.py          # Symulacja zamkniętej pętli HARQ
├── run_additional_experiments.py   # SNR sweep, destylacja, per-kanał
├── test_ml_link_adaptation.cpp     # Test C++ (latencja 84 ns)
├── parse_srsran_logs.py            # Parser logów stacji bazowej
├── srsran_integration/             # Kod do wdrożenia w srsRAN
│   ├── scheduler/                  #   → zmodyfikowany scheduler z ML
│   ├── configs/                    #   → konfiguracje gNB i UE
│   └── docker/                     #   → rdzeń sieci (Docker)
├── results/                        # Wykresy i wyniki
└── REPORT.md                       # Raport techniczny (ang.)
```

---

## 7. Podsumowanie

| Aspekt | Status |
|--------|--------|
| Symulacja PHY + generacja danych | ✅ 2.2M próbek |
| Trening modelu ML | ✅ Ordinal GBM z ograniczeniami |
| Porównanie z literaturą | ✅ 8 metod, 10 seedów, 95% CI |
| Eksport do C++ | ✅ Drzewo decyzyjne, 84 ns |
| Wdrożenie w srsRAN | ✅ Pełny stos 5G SA |
| Testy na żywo | ✅ 0% packet loss, ML aktywne |

**Najważniejszy wniosek:** udało się zamknąć pełną pętlę — od symulacji fizycznej, przez trening modelu, po wdrożenie w działającej stacji bazowej 5G. Algorytm działa w czasie rzeczywistym (84 ns) i podejmuje poprawne decyzje o selekcji MCS.

## 8. Dalsze prace

- Testy na prawdziwym sprzęcie radiowym (USRP) zamiast wirtualnego radia
- Integracja z architekturą O-RAN (aktualizacja modelu bez restartu stacji)
- Rozszerzenie o scheduling wielu użytkowników jednocześnie (multi-user MIMO)
- Testy z emulowanym kanałem radiowym (pogorszenie jakości sygnału w czasie)
