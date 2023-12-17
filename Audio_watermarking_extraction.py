from scipy.io import wavfile
import matplotlib.pyplot as plt 
import numpy as np 
import pywt 
import cv2

def calculate_snr(original_signal, noise_signal):
    # Calculate the power of the original signal and the noise signal
    original_power = np.sum(original_signal ** 2) / len(original_signal)
    noise_power = np.sum(noise_signal ** 2) / len(noise_signal)
    
    # Calculate SNR in dB
    snr = 10 * np.log10(original_power / noise_power)
    return round(snr, 3)

def calculate_psnr(original_signal, noise_signal):
    # Calculate the maximum possible power of a signal
    max_power = np.max(original_signal) ** 2
    
    # Calculate the power of the noise signal
    noise_power = np.sum(noise_signal ** 2) / len(noise_signal)
    
    # Calculate PSNR in dB
    psnr = 10 * np.log10(max_power / noise_power)
    return round(psnr,3)

def calculate_nc(original_watermark, extracted_watermark):
    # Flatten the images and normalize them
    original_watermark = original_watermark.flatten() / np.max(original_watermark)
    extracted_watermark = extracted_watermark.flatten() / np.max(extracted_watermark)
    
    # Calculate the NC
    nc = np.sum(original_watermark * extracted_watermark) / np.sqrt(np.sum(original_watermark ** 2) * np.sum(extracted_watermark ** 2))
    return round(nc, 3)

#Load Orignial watermark
original_watermark = cv2.imread('small-watermark.jpg')

#Load The Original Audio Sample
samplerate, original_data = wavfile.read('sample.wav')
t = np.arange(len(original_data)) / float(samplerate);  # Retrieving Time
original_data = original_data / max(original_data)

coeffs = pywt.wavedec(data=original_data, wavelet='bior6.8', mode="sym", level=2) # DWT Multilevel Decomposition

cA2, cD2, cD1 = coeffs

# Load the watermarked audio
samplerate, watermarked_data = wavfile.read('watermarked_audio_lena.wav')
watermarked_data = watermarked_data / max(watermarked_data)

# Perform 2-level DWT on the watermarked audio
watermarked_coeffs = pywt.wavedec(data=watermarked_data, wavelet='bior6.8', mode="sym", level=2)
watermarked_cA2, watermarked_cD2, watermarked_cD1 = watermarked_coeffs

# Subtract the original cD2 from the watermarked cD2 to get the scaled image data
scaled_img_data = watermarked_cD2 - cD2

# Scale up the image data
extracted_img_data = scaled_img_data / 0.01  # Adjust the constant as needed

# Calculate the total number of elements in original_watermark
num_elements = np.prod(original_watermark.shape)

# Ensure extracted_img_data has the same number of elements
if extracted_img_data.size > num_elements:
    extracted_img_data = extracted_img_data[:num_elements]
elif extracted_img_data.size < num_elements:
    extracted_img_data = np.pad(extracted_img_data, (0, num_elements - extracted_img_data.size))


# Reshape the image data back to its original shape
extracted_img = extracted_img_data.reshape(original_watermark.shape)

# Calculate the difference between the original and watermarked audio To know the noise 
difference = watermarked_data - original_data


# Calculate the Evaluation Metrics
snr = calculate_snr(original_data, difference)
psnr = calculate_psnr(original_data, difference)

nc = calculate_nc(original_watermark, extracted_img)

print(f"SNR: {snr} dB")
print(f"PSNR: {psnr} dB")
print(f"NC: {nc}")




#-------------- Plotting -----------------#
plt.figure(figsize=(30, 20))

# Subplot for Original Signal
plt.subplot(3, 2, 1)
plt.plot(t, original_data, color='k')
plt.xlabel('Time')
plt.ylabel('S')
plt.title('Original Signal')

# Subplot for Watermarked Signal
plt.subplot(3, 2, 3)
plt.plot(t, watermarked_data, color='g')
plt.xlabel('Time')
plt.ylabel('S')
plt.title('Watermarked Signal')

# Subplot for Difference
plt.subplot(3, 2, 5)
plt.plot(t, difference, color='r')
plt.xlabel('Time')
plt.ylabel('S')
plt.title('Difference')

# Subplot for Original Watermark
plt.subplot(2, 2, 2)
plt.imshow(original_watermark, cmap='gray')
plt.title('Original Watermark')

# Subplot for Extracted Watermark
plt.subplot(2, 2, 4)
plt.imshow(extracted_img, cmap='gray')
plt.title('Extracted Watermark')

# Subplot for Eval Metrics
plt.figtext(0.55, 0.1, f"SNR: {snr} dB \n PSNR: {psnr} dB \n NC: {nc}", horizontalalignment='center', verticalalignment='bottom',fontsize=14)

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the padding between subplots
plt.savefig('plot.png', dpi=100)  # Saving plot as PNG image

plt.show()