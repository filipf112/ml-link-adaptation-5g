import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import pandas as pd
import numpy as np

class SimpleLink5G(tf.keras.Model):
    def __init__(self, num_bits_per_symbol):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol

        # Prawdziwe Parametry 5G LDPC (Base Graph)
        self.k = 1056  # Ilość bitów informacji w bloku
        self.n = 2112  # Długość słowa kodowego z bitami parzystości (Rate 1/2)
        self.coderate = self.k / self.n

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.channel = AWGN()
        self.demapper = Demapper("app", "qam", num_bits_per_symbol)

        # Wbudowany dekoder, który sam radzi sobie ze znakami LLR!
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        # 1. Bity informacyjne
        bits = self.binary_source([batch_size, self.k])

        # 2. Kodowanie LDPC (Dodajemy tarcze ochronne na szum)
        codewords = self.encoder(bits)

        # 3. Szum i Modulacja
        no = ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        x = self.mapper(codewords)
        y = self.channel(x, no)

        # 4. Demapowanie i Inteligentne Dekodowanie LDPC
        llr = self.demapper(y, no)
        bits_rx = self.decoder(llr)

        return bits, bits_rx

def generate_ml_dataset():
    print("Rozpoczynam symulację 5G L1 (z LDPC) i generowanie dumpów...")
    batch_size = 1000
    snr_range = np.arange(-5, 20, 2)
    modulations = [2, 4, 6]
    dataset = []

    for bits_per_symbol in modulations:
        print(f"Konfiguracja modelu dla {2**bits_per_symbol}-QAM...")
        model = SimpleLink5G(bits_per_symbol)

        for snr in snr_range:
            bits_tx, bits_rx = model(
                batch_size=batch_size,
                ebno_db=tf.constant(snr, dtype=tf.float32)
            )

            # Obliczamy BLER (Block Error Rate) w warstwie MAC
            # Jeśli choć jeden bit w całym bloku jest błędny -> pakiet do kosza (NACK)
            errors_per_block = tf.reduce_sum(tf.abs(bits_tx - bits_rx), axis=1)
            bler = tf.reduce_mean(tf.cast(errors_per_block > 0, tf.float32)).numpy()

            # Faktyczna przepustowość spektralna (Throughput)
            throughput = (1.0 - bler) * bits_per_symbol * model.coderate

            dataset.append({
                'SNR_dB': snr,
                'Bits_Per_Symbol': bits_per_symbol,
                'Modulation': f"{2**bits_per_symbol}-QAM",
                'BLER': bler,
                'Throughput': throughput
            })

    df = pd.DataFrame(dataset)
    df.to_csv('sionna_link_adaptation_dataset.csv', index=False)
    print("Gotowe! Zapisano nowy plik.")

if __name__ == '__main__':
    generate_ml_dataset()
