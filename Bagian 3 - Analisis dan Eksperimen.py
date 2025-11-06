import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

# Eksperimen dengan citra nyata
def experiment_real_image():
    """Eksperimen dengan citra daun untuk ekstraksi urat daun"""
    
    # Jika tidak ada citra nyata, buat simulasi citra daun
    leaf_img = np.zeros((300, 300), dtype=np.uint8)
    
    # Buat bentuk daun oval
    cv2.ellipse(leaf_img, (150, 150), (120, 80), 0, 0, 360, 255, -1)
    
    # Tambahkan urat daun (garis-garis)
    # Urat utama
    cv2.line(leaf_img, (150, 70), (150, 230), 200, 3)
    # Urat sekunder
    for i in range(5):
        y = 90 + i * 30
        cv2.line(leaf_img, (150, y), (80, y-20), 200, 2)
        cv2.line(leaf_img, (150, y), (220, y-20), 200, 2)
    
    # Tambahkan noise
    noise_mask = np.random.random((300, 300)) > 0.98
    leaf_img[noise_mask] = 255
    
    # Tambahkan lubang kecil
    cv2.circle(leaf_img, (100, 120), 3, 0, -1)
    cv2.circle(leaf_img, (200, 180), 2, 0, -1)
    
    # Proses ekstraksi urat daun
    # 1. Closing untuk menutup lubang kecil
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(leaf_img, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Opening untuk menghilangkan noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Top-Hat untuk menonjolkan urat daun
    kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    tophat = cv2.morphologyEx(opened, cv2.MORPH_TOPHAT, kernel_th)
    
    # 4. Thresholding untuk binerisasi urat daun
    _, veins_binary = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
    
    # 5. Skeletonization untuk mendapatkan struktur urat
    veins_skeleton = morphology.skeletonize(veins_binary > 0)
    veins_skeleton = veins_skeleton.astype(np.uint8) * 255
    
    # Plot hasil
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    images = [
        (leaf_img, "Citra Daun Asli"),
        (closed, "Setelah Closing (5x5)"),
        (opened, "Setelah Opening (3x3)"),
        (tophat, "Top-Hat Transform"),
        (veins_binary, "Urat Daun Biner"),
        (veins_skeleton, "Skeleton Urat Daun")
    ]
    
    for i, (img, title) in enumerate(images):
        ax = axes[i//3, i%3]
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ekstraksi_urat_daun.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return leaf_img, veins_skeleton

# Jalankan eksperimen
leaf_original, veins_result = experiment_real_image()

# Mini Report
print("\n" + "="*60)
print("MINI REPORT: EKSTRAKSI URAT DAUN")
print("="*60)
print("Nama Citra: simulated_leaf.png")
print("Tujuan Pemrosesan: Mengekstrak struktur urat daun")
print("\nRangkaian Operasi Morfologi:")
print("1. Closing (SE: ellipse 5x5) - Menutup lubang kecil pada daun")
print("2. Opening (SE: ellipse 3x3) - Menghilangkan noise kecil")
print("3. Top-Hat (SE: rectangle 15x15) - Menonjolkan urat daun")
print("4. Thresholding - Binerisasi hasil Top-Hat")
print("5. Skeletonization - Mendapatkan struktur garis tengah urat")
print("\nAlasan Efektivitas Kombinasi:")
print("- Closing dan Opening membersihkan noise tanpa merusak struktur utama")
print("- Top-Hat secara spesifik dirancang untuk mengekstrak fitur kecil")
print("- Skeletonization menyederhanakan struktur menjadi kerangka")
print("="*60)