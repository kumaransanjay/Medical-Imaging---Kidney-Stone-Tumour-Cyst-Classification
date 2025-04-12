# Medical Imaging - Kidney Disease Classification Using Deep Learning


## Objective

Develop an automated deep learning pipeline that classifies kidney conditions from medical images using advanced preprocessing techniques (enhancement and degradation) to improve diagnostic accuracy.

---

## Introduction

Medical image quality inconsistencies (noise, low contrast, blur) impact accurate kidney disease diagnosis. This project tackles such issues with:

- Enhancement techniques (CLAHE, sharpening)
- Degradation simulations (blur, noise)
- Custom CNN and ResNet50 models
- Performance evaluations under various conditions

---

## Dataset & Preprocessing

**Classes:** Normal, Stone, Cyst, Tumor  
**Versions Created:** Enhanced, Degraded

### Preprocessing Steps:

1. **Noise Reduction**  
   - Gaussian Blur  
   - Median Filtering  

2. **Contrast Enhancement**  
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)  

3. **Sharpening and Edge Enhancement**  
   - Unsharp Masking  
   - Laplacian Filters  

4. **Image Degradation (Controlled Distortion)**  
   - Blur  
   - Contrast Reduction  
   - Noise Addition  

5. **Denoising and Restoration**  
   - Adaptive Median Filtering  
   - Non-Local Means Denoising  

6. **Normalization**  
   - Rescaling (0–1 range for FastAI)  
   - `preprocess_input` for TensorFlow

---

## Methodology

### 1. Dataset Organization
- Structured into `train/`, `val/`, `test/` with class-wise folders.

### 2. Image Enhancement
- Applied CLAHE, Gaussian blur, sharpening, and Laplacian filters.

### 3. Image Degradation
- Simulated low-quality images with noise, blur, and contrast reduction.

### 4. Deep Learning Models

#### A. Custom CNN (FastAI)
- 2 Conv layers + Pooling + Dense + Dropout + ReLU  
- Trained for 5 epochs (LR = 3e-3)

#### B. ResNet50 (TensorFlow + Transfer Learning)
- `include_top=False` with custom classification layers  
- Frozen base, augmented images using `ImageDataGenerator`

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, Confusion Matrix  
- Visual performance plots generated

---

## Tools and Libraries
- OpenCV, NumPy, FastAI, TensorFlow/Keras  
- Matplotlib, Seaborn

---

## Applications
- Early diagnosis of kidney conditions
- Robust deep learning under image variability
- Integration into clinical/telemedicine systems

---

## Results

| Model          | Dataset Type | Accuracy | Precision | Recall |
|----------------|--------------|----------|-----------|--------|
| Custom CNN     | Enhanced     | 92.5%    | -         | -      |
| ResNet50       | Enhanced     | 93.6%    | 93.8%     | 93.4%  |
| Custom CNN     | Degraded     | 89.7%    | -         | -      |
| ResNet50       | Degraded     | 90.2%    | -         | -      |

---

## 🔗 Code

🔗 [Google Colab Project Notebook](https://colab.research.google.com/drive/1p8pkOGkCAZiUbz5apxx_SckHxSbteLh5?usp=sharing)

🔗 [Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

---

## Conclusion

Image enhancement significantly improves the accuracy of kidney condition classification models. ResNet50 with enhanced images outperformed other combinations, highlighting the importance of preprocessing. The approach is reliable, scalable, and ready for real-world medical use.

