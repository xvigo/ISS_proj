#!/usr/bin/env python 

# ISS projekt - Vilem Gottwald [Xgottw07]

from scipy.io import wavfile
from scipy.signal import spectrogram, butter, buttord, sosfilt, sosfreqz, sos2tf, sos2zpk, lfilter
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs

# Vytvoreni slozek pro grafy a audio
makedirs("./audio", exist_ok=True)
makedirs("./grafy", exist_ok=True)


### ---- 1. Ukol ----
print("První úkol:")

# nacteni vstupniho signalu
fs, signal = wavfile.read('xgottw07.wav')
signal = signal / 2**15

# delka signalu [vzorky]
len_samp = signal.size
print("  Délka signálu ve vzorcích", len_samp)

# delka signalu [sekundy]
len_sec = len_samp / fs
print("  Délka signálu v sekundách:" , len_sec, "s")

# max a min hodnota
print ("  Maximální hodnota:", signal.max())
print ("  Minimální hodnota:", signal.min())

# zobrazeni signalu
plt.figure(figsize=(10, 5))
plt.locator_params(axis='y', min_n_ticks=8)
plt.locator_params(axis='x', min_n_ticks=10)
plt.plot(np.linspace(0, len_sec, num=len_samp), signal)
plt.title('Vstupní signál')
plt.xlabel('$t[s]$')
plt.savefig("./grafy/1_Signal.svg", bbox_inches='tight')
# plt.show()
plt.close(1)

print ("  -- Zobrazen vstupní signál -- ")

### ---- 2. Ukol ----
print("\nDruhý úkol:")

# Ustredneni signalu
sig_mean = signal.mean()
signal = signal - sig_mean

# Normalizace signalu
sig_max = abs(signal).max()
signal = signal / sig_max

# Rozdeleni signalu na useky - vytvoreni matice ramcu
frames = []
s_start = 0 
sec_values = np.linspace(0, 1024 / fs, num=1024)

for i in range(int(len_samp/512) - 1):
	frames.append(signal[s_start:s_start + 1024])
	s_start += 512
	
	## zobrazeni grafu vsech ramcu pro rucni prohledani
	# plt.figure(figsize=(15, 5))
	# plt.plot(sec_values, frames[i])
	# plt.title(i)
	# plt.xlabel('$t[s]$')
	# g_name = "./ramce/" + str(i) + "ramec.jpg"
	# plt.savefig(g_name)
	# plt.close(1)

	## vytvoreni nahravek ramcu pro otestovani znelosti
	# a_name = "./audia/" + str(i) + "audio.wav"
	# wavfile.write(a_name, fs, frames[i])

# Zobrazeni zneleho ramce s periodickym charakterem - ramec 43
plt.figure(figsize=(15, 5))
plt.plot(sec_values, frames[43])
plt.title('Znělý rámec s periodickým charakterem')
plt.xlabel('$t[s]$')
plt.locator_params(axis='y', min_n_ticks=8)
plt.locator_params(axis='x', min_n_ticks=8)
plt.savefig("./grafy/2_Znely_periodicky_ramec.svg", bbox_inches='tight')
#plt.show()
plt.close(1)

print("  -- Zobrazen znělý rámec -- ")


### ---- 3. Ukol ----
print("\nTřetí úkol:")

# výpočet DFT za použítí matic
# zdroj: https://www.slideshare.net/sarang1689/computing-dft-using-matrix-method

# Vytvoreni komlexniho koeficientu (twiddle factor)
N = 1024
W_N = np.e**(-2j*np.pi/N)

# Vytvoreni matice bazi
TF_matrix = []
for n in range(N):
	col = []
	for k in range(N):
		col.append(W_N**(n*k))
	TF_matrix.append(col)

# DFT jako skalarni soucin matice bazi s vektorem ramce 43 
DFT = np.array(TF_matrix).dot(np.array(frames[43]))

# Knihovni implementace fft - ramec 43
FFT =  np.fft.fft(frames[43])

if np.allclose(DFT,FFT):
	print("  np.allclose - Výsledky vlastní a knihovní implementace DFT jsou shodné.")
else:
	print("  np.allclose - Výsledky vlastní a knihovní implementace DFT NEjsou shodné.")

# Zobrazeni vlastní DFT
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, fs/2, num=512), np.absolute(DFT[0:512]))
plt.title('DFT - rámec 43')
plt.locator_params(axis='y', min_n_ticks=10)
plt.locator_params(axis='x', min_n_ticks=10)
plt.xlabel('$F[Hz]$')
plt.ylabel('$|X[k]|$')
plt.savefig("./grafy/3_DFT.svg", bbox_inches='tight')
#plt.show()
plt.close(1)



print("  -- Zobrazen graf vastní implementace DFT -- ")

# Zobrazeni numpy FFT
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, fs/2, num=512), np.absolute(FFT[0:512]))
plt.locator_params(axis='y', min_n_ticks=10)
plt.locator_params(axis='x', min_n_ticks=10)
plt.title('Numpy FFT - rámec 43')
plt.xlabel('$F[Hz]$')
plt.ylabel('$|X[k]|$')
plt.savefig("./grafy/3_FFT.svg", bbox_inches='tight')
#plt.show()
plt.close(1)


print("  -- Zobrazen graf knihovní implementace FFT -- ")


### --- 4. úkol ----
print("\nČtvrtý úkol:")

# Vytvoreni spektrogram
f, t, sgr = spectrogram(signal, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 

# Zobrazeni spektrogramu
plt.figure(figsize=(10,5))
plt.pcolormesh(t,f,sgr_log)
plt.title('Logaritmický výkonový spektrogram')
plt.locator_params(axis='x', min_n_ticks=8)
plt.locator_params(axis='y', min_n_ticks=10)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.savefig("./grafy/4_Spektrogram_pred_filtraci.svg", bbox_inches='tight')
# plt.show()
plt.close(1)


print("  -- Zobrazen logaritmický výkonový spektrogram --")


### ---- 5. úkol ----
print("\nPátý úkol:")

# Nalezeni frekvenci se spektralni hustotou vykonu vetsi nez -100 dB na zacatku nahravky (bez hlasu)
print("  Adepti na rušivé frekvence:")
spectre0 = [i[0] for i in sgr_log]
freqs = []
values = []

# Indexy rusivych frekvenci  manualne vybranych z adeptu
manual_freq_index = [41, 82, 123, 164]

for i in range(len(spectre0)):
	if spectre0[i] > -50: 
		print("    frekvence: " +  "{:>8.3f}".format(f[i]) + " Hz, hodnota: " +  "{:>6.6f}".format(spectre0[i]) + " dB, index: " + "{:>3d}".format(i) )

		# Rusive frekvence - manualne vybrany z adeptu
		if i in manual_freq_index:
			freqs.append(f[i])
			values.append(spectre0[i])

# Vypis rusivych frekvenci
print("\n  Rušivé frekvence:")
for i in range(4):
	print("    Frekvence:", "{:>8.3f}".format(freqs[i]), "Hz s hodnotou:", "{:>6.6f}".format(values[i]), "dB")

# Kontrola zda jsou frekvence harmonikcy vztazene
if np.isclose(freqs[1] % freqs[0], 0) and np.isclose(freqs[2] % freqs[0], 0) and np.isclose(freqs[3] % freqs[0], 0):
	print("\n  Rušivé frekvence jsou harmonicky vztažené.")
else:
	print("\n  Rušivé frekvence NEjsou harmonicky vztažené.")



### 6. úkol
print("\nŠestý úkol:")

# vytvoreni signalu 4 cosinusovek
t = np.linspace(0, len_sec, num = len_samp) 
y = 1/4 * np.cos(2 * np.pi * freqs[0] * t) + 1/4 * np.cos(2 * np.pi * freqs[1] * t) + 1/4 * np.cos(2 * np.pi * freqs[2] * t) + 1/4 * np.cos(2 * np.pi * freqs[3] * t)
y *= 0.024 #snizeni amplitudy - vycteno ze zacatku grafu puvodniho signalu (lze pozorovat cosinusovky)
wavfile.write('./audio/4cos.wav', fs, y.astype(np.float32))

# vytvoreni spektrogramu 4 cosinusovek
f2, t2, sgr2 = spectrogram(y, fs, nperseg=1024, noverlap=512)
sgr_log2 = 10 * np.log10(sgr2+1e-20) 

# zobrazeni spektrogramu 4 cosunusovek
plt.figure(figsize=(10,5))
plt.pcolormesh(t2,f2,sgr_log2)
plt.title('Logaritmický výkonový spektrogram cosinusovek')
plt.locator_params(axis='x', min_n_ticks=8)
plt.locator_params(axis='y', min_n_ticks=10)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.savefig("./grafy/6_Spektrogram_4cos.svg", bbox_inches='tight')
# plt.show()
plt.close(1)


print("  -- Zobrazen spektrogram 4 cosinusovek --")


### 7. úkol
print("\nSedmý úkol:")

# Funkce pro zobrazeni impulsni odezvy filtru
def plot_imp_resp(n_of_samp, h, index):

	plt.figure(figsize=(12,5))
	plt.plot(np.arange(n_of_samp), h)
	plt.gca().set_xlabel('$n$')
	plt.locator_params(axis='y', min_n_ticks=10)
	plt.locator_params(axis='x', min_n_ticks=10)
	plt.gca().set_title(("Impulsní odezva " + str(index + 1) + ". filtru (" + str(freqs[i]) + " Hz) $h[n]$"))

	plt.grid(alpha=0.5, linestyle='--')

	plt.savefig(("./grafy/7_Imp_odezva_filtr" + str(index + 1) + ".svg"), bbox_inches='tight')
	# plt.show()
	plt.close(1)
	

# Nyquistova frekvence
nyq = fs/2 
sos_l = [] 

for i in range(len(freqs)):
	# Vytvoreni hranic zaverneho a propustneho pasma
	pass_min = (freqs[i]  - 15 - 50) / nyq
	pass_max = (freqs[i]  + 15 + 50) / nyq
	stop_min = (freqs[i] - 15) / nyq
	stop_max = (freqs[i] + 15) /nyq

	# Vytvoreni filtru
	N, Wn = buttord([pass_min, pass_max], [stop_min, stop_max], 3, 40)
	sos_l.append(butter(N, Wn, 'bandstop', False, 'sos'))

	# Zobrazeni impulsni odezvy filtru
	N = 500
	imp = [1, *np.zeros(N-1)]
	imp_resp = sosfilt(sos_l[i], imp)
	plot_imp_resp(N, imp_resp, i)

	# Vypis koeficientu filtru
	b,a = sos2tf(sos_l[i])
	print("\n  Koeficienty "+ str(i+1) +". filtru (" + str(freqs[i]) + " Hz)")
	print("    a:", a)
	print("    b:", b)

print("\n  -- Zobrazeny impulsní odezvy filtrů --")


### 8. úkol
print("\nOsmý úkol:")

# Funkce pro zobrazeni nul a polu filtru
def plot_zeros_poles(z, p, index):
	plt.figure(figsize=(6,6))

	# jednotkova kruznice
	ang = np.linspace(0, 2*np.pi,100)
	plt.plot(np.cos(ang), np.sin(ang))

	# nuly, poly
	plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
	plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
	plt.legend(loc='upper left')
	
	plt.title(("Nuly a póly " + str(index + 1) + ". filtru (" + str(freqs[i]) + " Hz)"))
	plt.xlabel('Realná složka $\mathbb{R}\{$z$\}$')
	plt.ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

	plt.grid(alpha=0.5, linestyle='--')
	plt.savefig(("./grafy/8_Nuly_poly_filtr" + str(index + 1) + ".svg"), bbox_inches='tight')
	#plt.show()
	plt.close(1)

# Zobrazeni nul a polu filtru
for i in range(len(sos_l)):
	z, p ,k = sos2zpk(sos_l[i])
	plot_zeros_poles(z, p, i)

print("  -- Zobrazeny nuly a póly filtrů --")


### 9. úkol
print("\nDevátý úkol:")

# Funkce pro zobrazeni frekvencni charakteristiky filtru
def plot_freq_char(index):
	w, H = sosfreqz(sos_l[index], fs)
	figure, ax = plt.subplots(1, 2, figsize=(15,6))
	ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
	ax[0].set_xlabel('Frekvence [Hz]')
	ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
	ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
	ax[1].set_xlabel('Frekvence [Hz]')
	ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
	figure.suptitle((("Frekvenční charakteristika " + str(index + 1) + ". filtru (" + str(freqs[i]) + " Hz)")), fontsize=14)
	for ax1 in ax:
		ax1.grid(alpha=0.5, linestyle='--')
	plt.savefig("./grafy/9_Frekv_charakt_filtr" + str(index + 1) + ".svg", bbox_inches='tight')
	# plt.show()
	plt.close(1)

# Zobrazeni frekv. charakteristik filtru
for i in range(len(sos_l)):
	plot_freq_char(i)

print("  -- Zobrazeny frekvenční charakteristiky filtrů --")


### 10. úkol
print("\nDesátý úkol:")

# Provedeni filtrace
for i in range(len(sos_l)):
		signal = sosfilt(sos_l[i], signal)

# Kontrola dynamickeho rozsahu -1 - 1
out_max = abs(signal).max()
if out_max > 1:
	signal *= out_max

# Uvedeni siganlu do puvodniho stavu - odnormalizovani, odustredneni
signal = signal * sig_max
signal = signal + sig_mean

# Vytvoreni wav souboru 
wavfile.write("./audio/clean_bandstop.wav", fs, signal.astype(np.float32))
print("  -- Provedena filtrace a uložení audio souboru -- ")
