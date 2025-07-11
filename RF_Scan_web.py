import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from rtlsdr import RtlSdr
from collections import deque
import time
import pandas as pd
import socket

st.set_page_config(layout="wide")
FFT_SIZE = 4096
BUFFER_SIZE = 100
UDP_PORT = 8052

def receive_red_pitaya_data(sock):
    try:
        data, _ = sock.recvfrom(256 * 4)
        spectrum = np.frombuffer(data, dtype=np.int32)
        spectrum = np.fft.fftshift(spectrum)
        return spectrum.astype(float) / 25000.0
    except BlockingIOError:
        return None

def receive_udp_fft(recv_sock):
    try:
        data, _ = recv_sock.recvfrom(FFT_SIZE * 8)
        samples = np.frombuffer(data, dtype=np.complex64)
        spectrum = np.fft.fftshift(np.fft.fft(samples * np.hanning(FFT_SIZE)))
        power = 20 * np.log10(np.abs(spectrum) / 50)
        return power
    except BlockingIOError:
        return None

def receive_sdr_fft(sdr, center_freq):
    sdr.center_freq = center_freq
    samples = sdr.read_samples(FFT_SIZE)
    spectrum = np.fft.fftshift(np.fft.fft(samples * np.hanning(FFT_SIZE)))
    power = 20 * np.log10(np.abs(spectrum) / 50)
    return power, samples

st.sidebar.title("Menu")
udp_mode = st.sidebar.selectbox("UDP Mode", ["Unicast", "Broadcast"])
mode_select = st.sidebar.radio("Mode", ["Manual", "Sweep"])
udp_ip_input = "255.255.255.255" if udp_mode == "Broadcast" else st.sidebar.text_input("UDP Destination IP", value="172.16.129.180")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
if udp_mode == "Broadcast":
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

input_source = st.sidebar.selectbox("Input Source", ["SDR", "UDP Socket", "Red Pitaya"])
menu = st.sidebar.selectbox("Navigate", ["Home", "Settings", "About"])

spectrogram_buffer = deque(maxlen=BUFFER_SIZE)
power_buffer = deque(maxlen=BUFFER_SIZE)
mean_buffer = deque(maxlen=40)

plot = st.empty()
plot2 = st.empty()
plot3 = st.empty()
csv_button = st.button("Save Data to CSV")

if mode_select == "Manual":
    center_freq_mhz = st.slider("Center Frequency (MHz)", min_value=150.0, max_value=800.0, step=10.0, value=91.9)
    center_freq = center_freq_mhz * 1e6
else:
    sweep_delay = st.sidebar.slider("Sweep Interval (s)", 0.01, 1.0, 0.1, step=0.5)
    freq_list = np.arange(200, 800, 1) * 1e6
    if "current_freq_idx" not in st.session_state:
        st.session_state.current_freq_idx = 0
        st.session_state.last_sweep_time = time.time()
    center_freq = freq_list[st.session_state.current_freq_idx]

freq_list = np.arange(150, 800, 1) * 1e6

if input_source == "UDP Socket" or input_source == "Red Pitaya":
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.setblocking(False)
    recv_sock.bind(("0.0.0.0", UDP_PORT))

if input_source == "SDR":
    if "sdr" not in st.session_state:
        sdr = RtlSdr()
        sdr.sample_rate = 2.048e6
        sdr.gain = 1000.0
        st.session_state.sdr = sdr
    else:
        sdr = st.session_state.sdr

st.sidebar.markdown("Developed by SRIKANTH, CHANDANA, SHISHIR, MEENAKSHI")

while True:
    now = time.time()
    if mode_select == "Sweep" and now - st.session_state.last_sweep_time >= sweep_delay:
        st.session_state.current_freq_idx = (st.session_state.current_freq_idx + 1) % len(freq_list)
        center_freq = freq_list[st.session_state.current_freq_idx]
        st.session_state.last_sweep_time = now

    power = None
    samples = None

    if input_source == "SDR":
        power, samples = receive_sdr_fft(sdr, center_freq)
    elif input_source == "UDP Socket":
        power = receive_udp_fft(recv_sock)
    elif input_source == "Red Pitaya":
        power = receive_red_pitaya_data(recv_sock)
        # power = power/10000
        # power = power/(np.max(power) * 20)

    if power is None:
        time.sleep(0.01)
        continue

    spectrogram_buffer.appendleft(power)
    power_list = np.sum(power)
    mean_buffer.append(power_list)
    power_buffer.append(np.mean(mean_buffer))

    if input_source == "Red Pitaya":
        freqs_mhz = np.linspace(-62.5, 62.5, 256)
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(len(power), 1/2.048e6)) + center_freq
        freqs_mhz = freqs / 1e6

    fig_fft, ax_fft = plt.subplots(figsize=(5, 3))
    ax_fft.plot(freqs_mhz, power)
    ax_fft.set_title("Real-Time FFT")
    ax_fft.set_ylabel("Power (dB)")
    ax_fft.set_xlabel("Frequency (MHz)")
    if input_source == "Red Pitaya":
        ax_fft.set_xlim(0, 50)
    else:
        ax_fft.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax_fft.set_ylim(np.min(power), np.max(power))
    ax_fft.grid(True)
    plot.pyplot(fig_fft)
    plt.close(fig_fft)

    if udp_ip_input and input_source == "SDR" and samples is not None:
        sock.sendto(samples.astype(np.complex64).tobytes(), (udp_ip_input, UDP_PORT))

    fig_spec, (ax_spec, ax_pow) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3), width_ratios=[3, 1])
    ax_spec.imshow(np.array(spectrogram_buffer), origin='upper', aspect='auto', extent=[0 if input_source == "Red Pitaya" else freqs_mhz[0],50 if input_source == "Red Pitaya" else freqs_mhz[-1],0, len(spectrogram_buffer)],cmap='jet', vmin=np.min(power), vmax=np.max(power))
    ax_spec.set_title("Spectrogram")
    ax_pow.plot(power_buffer, np.arange(len(power_buffer)))
    ax_pow.set_title("Real-Time Power")
    ax_pow.set_xlabel("Time")
    ax_pow.set_ylabel("Total Power (dB)")
    plot2.pyplot(fig_spec)
    plt.close(fig_spec)

    if len(spectrogram_buffer) > 1 and input_source != "Red Pitaya":
        spec_array = np.array(spectrogram_buffer)
        avg_power_density = np.mean(spec_array, axis=0)
        avg_power_density -= np.min(avg_power_density)
        probability = avg_power_density / np.sum(avg_power_density)
        num_bins = 50
        bin_edges = np.linspace(freqs_mhz[0], freqs_mhz[-1], num_bins + 1)
        bin_indices = np.digitize(freqs_mhz, bin_edges) - 1
        binned_prob = np.zeros(num_bins)
        for i in range(num_bins):
            binned_prob[i] = np.sum(probability[bin_indices == i])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig_hist, ax_hist = plt.subplots(figsize=(6, 2.5))
        ax_hist.bar(bin_centers, binned_prob, width=(bin_edges[1] - bin_edges[0]), color='cyan')
        ax_hist.set_title("Frequency Distribution")
        ax_hist.set_xlabel("Frequency (MHz)")
        ax_hist.set_ylabel("Probability")
        ax_hist.grid(True)
        plot3.pyplot(fig_hist)
        plt.close(fig_hist)

    if csv_button and samples is not None:
        df = pd.DataFrame(samples, columns=['samples'])
        df.to_csv("data.csv", mode='a', index=False, header=False)

    time.sleep(0.01)
