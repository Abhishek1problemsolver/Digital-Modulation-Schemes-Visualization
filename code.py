import numpy as np
import matplotlib.pyplot as plt

# Parameters
bit_rate = 1e3
carrier_freq = 10e3
sampling_rate = 100e3
duration = 1
num_bits = int(bit_rate * duration)
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate random bit sequence
bit_sequence = np.random.randint(0, 2, num_bits)

# Modulation Schemes
def ask_modulation(bit_sequence, carrier_freq, sampling_rate, t):
    return np.array([(bit * np.cos(2 * np.pi * carrier_freq * t[i:i+int(sampling_rate/bit_rate)])) 
                     for i, bit in enumerate(bit_sequence)]).flatten()

def fsk_modulation(bit_sequence, carrier_freq, sampling_rate, t):
    fsk_signal = []
    for bit in bit_sequence:
        freq = carrier_freq + (bit * carrier_freq / 2)
        fsk_signal.append(np.cos(2 * np.pi * freq * t[:int(sampling_rate/bit_rate)]))
    return np.concatenate(fsk_signal)

def psk_modulation(bit_sequence, carrier_freq, sampling_rate, t):
    return np.array([(np.cos(2 * np.pi * carrier_freq * t[i:i+int(sampling_rate/bit_rate)] + 
                    np.pi * bit)) for i, bit in enumerate(bit_sequence)]).flatten()

# Generate Signals
ask_signal = ask_modulation(bit_sequence, carrier_freq, sampling_rate, t)
fsk_signal = fsk_modulation(bit_sequence, carrier_freq, sampling_rate, t)
psk_signal = psk_modulation(bit_sequence, carrier_freq, sampling_rate, t)

# Plot Signals
def plot_signal(signal, title, t, sampling_rate):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t[:len(signal)], signal)
    plt.title(f'{title} - Time Domain')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    fft_signal = np.fft.fft(signal)
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft_signal)))
    plt.title(f'{title} - Frequency Domain')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()

# Plot ASK
plot_signal(ask_signal, 'ASK Modulation', t, sampling_rate)

# Plot FSK
plot_signal(fsk_signal, 'FSK Modulation', t, sampling_rate)

# Plot PSK
plot_signal(psk_signal, 'PSK Modulation', t, sampling_rate)
