# DIP Image Enhancement System
# Author: Maiza Fatima
# Reg ID: 235205

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('hawkes_bay_in.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Task 6.1 - Image Acquisition
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Task 6.2 - Sampling & Quantization
img_05 = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
img_025 = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)
img_15 = cv2.resize(gray, (0,0), fx=1.5, fy=1.5)
img_2 = cv2.resize(gray, (0,0), fx=2, fy=2)
img_4bit = (gray & 0xF0)
img_2bit = (gray & 0xC0)

# Task 6.3 - Geometric Transformations
angles = [30, 45, 60, 90, 120, 150, 180]
M = np.float32([[1,0,50],[0,1,30]])
translated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
M_shear = np.float32([[1, 0.3, 0],[0.2, 1, 0]])
sheared = cv2.warpAffine(gray, M_shear, (gray.shape[1], gray.shape[0]))

# Task 6.4 - Intensity Transformations
img_double = gray / 255.0
negative = 255 - gray
c = 2
log_img = c * np.log1p(img_double)
log_img = log_img / log_img.max()
gamma_bright = np.power(img_double, 0.5)
gamma_dark = np.power(img_double, 1.5)

# Task 6.5 - Histogram Processing
hist_orig = cv2.calcHist([gray], [0], None, [256], [0,256])
pdf = hist_orig / hist_orig.sum()
cdf = np.cumsum(pdf)
new_values = np.uint8((255) * cdf)
eq_manual = new_values[gray]
eq_histeq = cv2.equalizeHist(gray)

# Task 6.6 - Final Pipeline
log_uint8 = np.uint8(log_img * 255)
hist = cv2.calcHist([log_uint8], [0], None, [256], [0,256])
pdf = hist / hist.sum()
cdf = np.cumsum(pdf)
new_values = np.uint8((255) * cdf)
enhanced_image = new_values[log_uint8]

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray');           plt.title('Original')
plt.subplot(1,3,2); plt.imshow(log_uint8, cmap='gray');      plt.title('Log Enhanced')
plt.subplot(1,3,3); plt.imshow(enhanced_image, cmap='gray'); plt.title('Final Enhanced')
plt.show()
