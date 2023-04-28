import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import resample
from scipy.signal import stft, istft, gaussian, convolve
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly
import librosa
import soundfile as sf
import pywt

def apply_window(data, window_type):
    if window_type == 'hanning':
        window = np.hanning(len(data))
    elif window_type == 'hamming':
        window = np.hamming(len(data))
    elif window_type == 'blackman':
        window = np.blackman(len(data))
    elif window_type == 'kaiser':
        window = np.kaiser(len(data), beta=14)
    elif window_type == 'cos3':
        window = signal.windows.cosine(len(data))
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    return data * window

def find_peak(signal):
    return np.argmax(np.abs(signal))

def resample_sinc(signal, new_len):
    return signal.resample(signal, new_len, window='kaiser', num=None)


def wavelet_morph(wave1, wave2, alpha, wavelet_type, freq_ratio, window_type, beta=None):
    coeffs1 = pywt.wavedec(wave1, wavelet_type)
    coeffs2 = pywt.wavedec(wave2, wavelet_type)

    morphed_coeffs = []
    for c1, c2 in zip(coeffs1, coeffs2):
        if freq_ratio != 1.0:
            c1_len = len(c1)
            c2_len = len(c2)
            new_len = int(c1_len * freq_ratio)

            c1_windowed = apply_window_around_peak(c1, window_type, beta)
            c2_windowed = apply_window_around_peak(c2, window_type, beta)

            c1_resampled = resample(c1_windowed, new_len)
            c2_resampled = resample(c2_windowed, new_len)

            c1_resampled = resample(c1_resampled, c1_len) 
            c2_resampled = resample(c2_resampled, c2_len) 

            morphed_c = alpha * c1_resampled + (1 - alpha) * c2_resampled
        else:
            morphed_c = alpha * c1 + (1 - alpha) * c2

        morphed_coeffs.append(morphed_c)

    morphed_wave = pywt.waverec(morphed_coeffs, wavelet_type)

    return morphed_wave

def apply_window_around_peak(signal, window_type, beta=None, peak_ratio=0.1):
    signal_len = len(signal)
    peak_index = np.argmax(np.abs(signal))
    window_len = int(signal_len * peak_ratio)

    start_index = max(0, peak_index - window_len // 2)
    end_index = min(signal_len, peak_index + window_len // 2)

    if window_type == 'hann':
        window = np.hanning(window_len)
    elif window_type == 'hamming':
        window = np.hamming(window_len)
    elif window_type == 'blackman':
        window = np.blackman(window_len)
    elif window_type == 'kaiser':
        if beta is None:
            raise ValueError('beta parameter must be provided for kaiser window')
        window = np.kaiser(window_len, beta)
    elif window_type == 'cos3':
        window = np.cos(np.linspace(0, np.pi, window_len))**3
    else:
        raise ValueError(f'Unsupported window type: {window_type}')

    # Resize window to match signal[start_index:end_index] shape
    window_resized = window[:end_index - start_index]

    windowed_signal = np.copy(signal)
    windowed_signal[start_index:end_index] = signal[start_index:end_index] * window_resized

    return windowed_signal


def read_waveform(filename):
    _, data = wavfile.read(filename)
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / np.iinfo(np.int16).max
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / np.iinfo(np.int32).max

    return data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Morph between two waveforms from WAV files')
    parser.add_argument('file1', type=str, help='Path to the first WAV file')
    parser.add_argument('file2', type=str, help='Path to the second WAV file')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='Morphing factor (0 <= alpha <= 1)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name (WAV format)')
    parser.add_argument('-b', '--blur', type=float, default=0.0, help='Gaussian blur strength (0.0-50.0)')
    parser.add_argument('-n', '--normalize', type=float, default=0.0, help='Normalization level in dBFS (default: 0 dBFS)')
    parser.add_argument('--dc', '-d', action='store_true', help='Apply DC filter to remove DC offset (default: off)')
    parser.add_argument('--stretch', '-s', action='store_true', help='Apply time stretching instead of zero-padding (default: off)')
    parser.add_argument('--plot', nargs='?', const=-1, type=float, default=None, help='Plot waveforms and spectrograms. Specify the horizontal width in milliseconds (default: the length of the longer input waveform)')
    parser.add_argument('--wavelet', type=str, default='db4', help='Wavelet type: haar, dbN, symN, coifN, or biorNr.Nd (default: db4)')
    parser.add_argument('--freq', type=float, default=1.0, help='Frequency morphing ratio (0.0 <= freq <= 1.0)')
    parser.add_argument('--window', type=str, default='hann', help='Window type for resampling (hann, hamming, blackman, kaiser, cos3).')
    parser.add_argument('--beta', type=float, default=None, help='Beta value for the Kaiser window.')
    parser.add_argument('--fft_size', type=int, default=2048, help='Size of the FFT window')
    parser.add_argument('--resample', '-r', type=str, choices=['poly', 'sinc'], default='poly', help='Resampling method to be used (poly or sinc).')
    return parser.parse_args()

def normalize_waveform(waveform, target_dBFS):
    current_dBFS = 20 * np.log10(np.max(np.abs(waveform)))
    gain = 10 ** ((target_dBFS - current_dBFS) / 20)

    return waveform * gain

def time_stretch(waveform, target_length):
    current_length = waveform.shape[0]
    target_indices = np.round(np.linspace(0, current_length - 1, target_length)).astype(int)
    return waveform[target_indices]

def apply_gaussian_blur(waveform, sr, blur_strength):
    f, t, Zxx = stft(waveform, fs=sr, nperseg=1024, noverlap=512)
    blurred_Zxx = gaussian_filter(np.abs(Zxx), sigma=blur_strength * 50) * np.exp(1j * np.angle(Zxx))
    _, morphed_wave = istft(blurred_Zxx, fs=sr, nperseg=1024, noverlap=512)
    return morphed_wave


def remove_dc_offset(waveform):
    return waveform - np.mean(waveform)

def main():
    args = parse_arguments()

    wave1 = read_waveform(args.file1)
    wave2 = read_waveform(args.file2)
    sr1 = sr2 = librosa.get_samplerate(args.file1)

    if sr1 != sr2:
        print("Error: The sampling rates of the two files must be the same.")
        return

    if wave1.ndim == 2:
        wave1_left, wave1_right = wave1[:, 0], wave1[:, 1]
    else:
        wave1_left, wave1_right = wave1, wave1

    if wave2.ndim == 2:
        wave2_left, wave2_right = wave2[:, 0], wave2[:, 1]
    else:
        wave2_left, wave2_right = wave2, wave2

    max_length_left = max(wave1_left.shape[0], wave2_left.shape[0])
    max_length_right = max(wave1_right.shape[0], wave2_right.shape[0])

    if args.stretch:
        wave1_left = time_stretch(wave1_left, max_length_left)
        wave1_right = time_stretch(wave1_right, max_length_right)
        wave2_left = time_stretch(wave2_left, max_length_left)
        wave2_right = time_stretch(wave2_right, max_length_right)
    else:
        wave1_left = np.pad(wave1_left, (0, max_length_left - wave1_left.shape[0]), 'constant')
        wave1_right = np.pad(wave1_right, (0, max_length_right - wave1_right.shape[0]), 'constant')
        wave2_left = np.pad(wave2_left, (0, max_length_left - wave2_left.shape[0]), 'constant')
        wave2_right = np.pad(wave2_right, (0, max_length_right - wave2_right.shape[0]), 'constant')

    morphed_wave_left = wavelet_morph(wave1_left, wave2_left, args.alpha, args.wavelet, args.freq, args.window, beta=args.beta)
    morphed_wave_right = wavelet_morph(wave1_right, wave2_right, args.alpha, args.wavelet, args.freq, args.window, beta=args.beta)

    if 0 < args.blur <= 50.0:
        wave1_left = apply_gaussian_blur(wave1_left, sr1, args.blur)
        wave1_right = apply_gaussian_blur(wave1_right, sr1, args.blur)
        wave2_left = apply_gaussian_blur(wave2_left, sr2, args.blur)
        wave2_right = apply_gaussian_blur(wave2_right, sr2, args.blur)
        morphed_wave_left = apply_gaussian_blur(morphed_wave_left, sr1, args.blur)
        morphed_wave_right = apply_gaussian_blur(morphed_wave_right, sr1, args.blur)

    wave1_left_normalized = normalize_waveform(wave1_left, args.normalize)
    wave1_right_normalized = normalize_waveform(wave1_right, args.normalize)
    wave2_left_normalized = normalize_waveform(wave2_left, args.normalize)
    wave2_right_normalized = normalize_waveform(wave2_right, args.normalize)
    morphed_wave_left_normalized = normalize_waveform(morphed_wave_left, args.normalize)
    morphed_wave_right_normalized = normalize_waveform(morphed_wave_right, args.normalize)

    if args.dc:
        wave1_left = remove_dc_offset(wave1_left)
        wave1_right = remove_dc_offset(wave1_right)
        wave2_left = remove_dc_offset(wave2_left)
        wave2_right = remove_dc_offset(wave2_right)

    morphed_wave = np.column_stack((morphed_wave_left_normalized, morphed_wave_right_normalized)).astype(np.float32)

    if args.output:
        sf.write(args.output, morphed_wave, sr1, subtype='FLOAT')
        
    if args.plot:
        plt.figure(figsize=(20, 12))

        def plot_waveform_and_spectrogram(index, waveform, title):
            f, t, Zxx = stft(waveform, fs=sr1, nperseg=1024, noverlap=512)
            magnitude = np.abs(Zxx)
            log_magnitude = 20 * np.log10(np.maximum(magnitude, 1e-8)) 

            clipped_waveform = np.clip(waveform, -1, 1)

            plt.subplot(3, 4, index)
            plt.plot(np.clip(waveform, -1, 1))
            plt.ylim(-1.05, 1.05) 
            plt.title(f'{title} Waveform')

            if args.plot > 0:  
                plt.xlim(0, int(args.plot * sr1 / 1000)) 
            else:  
                plt.xlim(0, len(waveform))

            plt.subplot(3, 4, index + 1)
            plt.pcolormesh(t, f, log_magnitude, shading='gouraud', cmap='viridis')
            plt.title(f'{title} Spectrogram')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')

        plot_waveform_and_spectrogram(1, wave1_left, 'Waveform 1 Left Channel')
        plot_waveform_and_spectrogram(3, wave1_right, 'Waveform 1 Right Channel')
        plot_waveform_and_spectrogram(5, wave2_left, 'Waveform 2 Left Channel')
        plot_waveform_and_spectrogram(7, wave2_right, 'Waveform 2 Right Channel')
        plot_waveform_and_spectrogram(9, morphed_wave_left, 'Morphed Waveform Left Channel')
        plot_waveform_and_spectrogram(11, morphed_wave_right, 'Morphed Waveform Right Channel')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
