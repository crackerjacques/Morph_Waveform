# Morph_Waveform
Morph 2 wave file with wavelet.
![IMG](https://github.com/crackerjacques/Morph_Waveform/blob/main/images/moprh.png?raw=true)

# Requirements


Python as librosa enabled version.

```
 pip install argparse numpy scipy soundfile matplolib numpy librosa PyWavelets
```

When you got pip error, please use python 3.9 in conda or pyenv.

# Command and Option

___Example___

```
python morphwave.py sound01.wav sound02.wav -a 0.6 -n -6 --plot -o morped.wav
```



___With blur, time streach and frequency morph FFT without plotting.___

```
python morphwave.py sound01.wav sound02.wav -s -a 0.5 -n -6 -b 10 --fft_size 2048 --window blackman --wavelet coif4 --freq 0.75 --dc -o morped02.wav
```

  -a alpha mix ratio
  -o output filename.wav
  -b blur bluring waveform
  -n normalize dBFS
  --dc dc offset filtre
  -s streach shorter file to align longer file
  --plot plot spectrogram and waveform
  --freq frequency morphing 0.0 to 1.0
  --window window function hamming, hann, blackman, kaiser, cos3
  --fft_size fft window size
  --wavelet haar, dbN, symN, coifN, or biorNr.Nd
 
# Usage

I expected this to morph the impulse response and create two intermediate files. Generally successful, but it might also be useful for creating waveforms for a wavetable synthesizer. Therefore, the output file is fixed to stereo, but if you want to operate it as monaural, please use sox's remix command to output as monaural.



Bluring sample::
![IMG](https://github.com/crackerjacques/Morph_Waveform/blob/main/images/blur.png?raw=true)

IR+IR /w Freq FFT configured.
![IMG](https://github.com/crackerjacques/Morph_Waveform/blob/main/images/IR.png?raw=true)

