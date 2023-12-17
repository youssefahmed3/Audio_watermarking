from scipy.io import wavfile
import matplotlib.pyplot as plt 
import numpy as np 
import pywt 
import cv2

#Retreiving the audio File as Wav and gett only the samplerate and the data
# wavfile.read() outputs (samplerate, data, dtype)
samplerate, data = wavfile.read('sample.wav')
t = np.arange(len(data)) / float(samplerate);  # Retrieving Time
data = data / max(data)

#-------------- Watermark pre-processing
# Load the image
img = cv2.imread('small-watermark.jpg')

# Flatten the image
flat_img = img.ravel()

# Normalize the vector
norm_img = (flat_img - np.min(flat_img)) / (np.max(flat_img) - np.min(flat_img))


print(f'Original image size: {img.shape}')
print(f'Normalized image size: {norm_img.shape}')
""" print(f'Reduced image size: {reduced_img.shape}') """
#-------------- Multi Decomposition 

coeffs = pywt.wavedec(data=data, wavelet='bior6.8', mode="sym", level=2) # DWT Multilevel Decomposition

cA2, cD2, cD1 = coeffs

# Ensure the image data and cD2 are of the same length
if len(norm_img) > len(cD2):
    norm_img = norm_img[:len(cD2)]
elif len(norm_img) < len(cD2):
    norm_img = np.pad(norm_img, (0, len(cD2) - len(norm_img)))

# Scale down the image data
scaled_img = norm_img * 0.01  # Adjust the constant as needed

# Add the scaled image data to cD2 to create the watermark
watermarked_cD2 = cD2 + scaled_img

# Replace the original cD2 with the watermarked cD2
coeffs[1] = watermarked_cD2

# Perform inverse DWT to get the watermarked audio
watermarked_audio = pywt.waverec(coeffs=coeffs, wavelet='bior6.8', mode='sym')

# Write the watermarked audio to a file
wavfile.write('watermarked_audio_lena.wav', samplerate, watermarked_audio)
print('watermarked_audio')

# Formatting for figure
L = len(data)
watermarked_audio = watermarked_audio[0:L];  # Matching length with input for plotting

plt.figure(figsize=(30, 20));

plt.subplot(5, 1, 1)
plt.plot(t, data, color='k');
plt.xlabel('Time');
plt.ylabel('S');
plt.title('Original Signal');

plt.subplot(5, 1, 2)
plt.plot(cA2, color='r');
plt.xlabel('Samples');
plt.ylabel('cA2');
plt.title('Approximation Coeff. (cA2)');

plt.subplot(5, 1, 3)
plt.plot(cD2, color='g');
plt.xlabel('Samples');
plt.ylabel('cD2');
plt.title('Detailed Coeff. (cD2)');

plt.subplot(5, 1, 3)
plt.plot(watermarked_cD2, color='g');
plt.xlabel('Samples');
plt.ylabel('cD2 watermarked');
plt.title('Detailed Coeff. (watermarked cD2)');

plt.subplot(5, 1, 4)
plt.plot(cD1, color='g');
plt.xlabel('Samples');
plt.ylabel('cD2');
plt.title('Detailed Coeff. (cD1)')

plt.subplot(5, 1, 5)
plt.plot(t, watermarked_audio, color='b');
plt.xlabel('Time');
plt.ylabel('Value');
plt.title('Reconstructed Signal');

plt.savefig('plot.png', dpi=100)  # Saving plot as PNG image
plt.show()



