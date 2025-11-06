import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, io, color, filters
import matplotlib

# Setup untuk menampilkan gambar
def setup_plot():
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

def create_sample_images():
    """Membuat dua citra biner sederhana"""
    # Citra 1: Huruf A
    img1 = np.zeros((200, 200), dtype=np.uint8)
    points = np.array([[50, 150], [100, 50], [150, 150], [120, 120], [80, 120]], np.int32)
    cv2.fillPoly(img1, [points], 255)
    
    # Citra 2: Bentuk geometris kompleks
    img2 = np.zeros((200, 200), dtype=np.uint8)
    # Persegi panjang
    cv2.rectangle(img2, (30, 30), (100, 100), 255, -1)
    # Lingkaran
    cv2.circle(img2, (150, 80), 40, 255, -1)
    # Segitiga
    triangle_pts = np.array([[120, 150], [80, 120], [160, 120]], np.int32)
    cv2.fillPoly(img2, [triangle_pts], 255)
    
    # Tambahkan noise untuk demonstrasi
    noise_mask1 = np.random.random((200, 200)) > 0.95
    img1[noise_mask1] = 255
    
    noise_mask2 = np.random.random((200, 200)) > 0.97
    img2[noise_mask2] = 255
    
    return img1, img2

def apply_morphology_operations(image, title):
    """Menerapkan berbagai operasi morfologi pada citra"""
    results = {}
    results['title'] = title
    results['original'] = image.copy()
    
    # Normalisasi ke range 0-1 untuk operasi morfologi
    img_binary = (image > 128).astype(np.uint8)
    
    # Structuring elements
    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_line_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))
    kernel_line_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1))
    
    # Dilasi
    results['dilasi_disk'] = cv2.dilate(img_binary, kernel_disk, iterations=1) * 255
    results['dilasi_line_vert'] = cv2.dilate(img_binary, kernel_line_vert, iterations=1) * 255
    results['dilasi_line_horiz'] = cv2.dilate(img_binary, kernel_line_horiz, iterations=1) * 255
    
    # Erosi
    results['erosi_disk'] = cv2.erode(img_binary, kernel_disk, iterations=1) * 255
    results['erosi_line_vert'] = cv2.erode(img_binary, kernel_line_vert, iterations=1) * 255
    results['erosi_line_horiz'] = cv2.erode(img_binary, kernel_line_horiz, iterations=1) * 255
    
    # Opening dengan ukuran berbeda
    for size in [3, 5, 9]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        results[f'opening_{size}'] = opening * 255
        results[f'closing_{size}'] = closing * 255
    
    # Skeletonization
    skeleton = morphology.skeletonize(img_binary > 0)
    results['skeleton'] = skeleton.astype(np.uint8) * 255
    
    # Top-Hat and Bottom-Hat
    kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel_th)
    bottomhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_th)
    results['tophat'] = tophat
    results['bottomhat'] = bottomhat
    
    return results

def plot_results(results1, results2):
    """Plot hasil operasi morfologi dalam grid"""
    fig, axes = plt.subplots(11, 5, figsize=(20, 30))
    
    operations = [
        ('Original', 'original'),
        ('Dilasi Disk', 'dilasi_disk'),
        ('Dilasi Line Vert', 'dilasi_line_vert'),
        ('Dilasi Line Horiz', 'dilasi_line_horiz'),
        ('Erosi Disk', 'erosi_disk'),
        ('Erosi Line Vert', 'erosi_line_vert'),
        ('Erosi Line Horiz', 'erosi_line_horiz'),
        ('Opening 3x3', 'opening_3'),
        ('Opening 5x5', 'opening_5'),
        ('Opening 9x9', 'opening_9'),
        ('Closing 3x3', 'closing_3'),
        ('Closing 5x5', 'closing_5'),
        ('Closing 9x9', 'closing_9'),
        ('Skeleton', 'skeleton'),
        ('Top-Hat', 'tophat'),
        ('Bottom-Hat', 'bottomhat')
    ]
    
    for i, (op_name, op_key) in enumerate(operations):
        if i < 11:  # Pastikan tidak melebihi jumlah baris
            # Citra 1
            axes[i, 0].imshow(results1[op_key], cmap='gray')
            axes[i, 0].set_title(f'Citra 1: {op_name}')
            axes[i, 0].axis('off')
            
            # Citra 2
            axes[i, 1].imshow(results2[op_key], cmap='gray')
            axes[i, 1].set_title(f'Citra 2: {op_name}')
            axes[i, 1].axis('off')
    
    # Tambahkan analisis tekstual
    analysis_text = """
    ANALISIS PERUBAHAN UTAMA:
    
    1. DILASI: 
       - Disk: Memperbesar objek secara merata, mengisi celah kecil
       - Line Vert/Horiz: Memperbesar objek secara selektif sesuai arah
    
    2. EROSI:
       - Disk: Memperkecil objek, menghilangkan noise kecil
       - Line Vert/Horiz: Memperkecil objek secara selektif
    
    3. OPENING:
       - Size 3: Menghilangkan noise sangat kecil
       - Size 5: Menghilangkan noise sedang, mempertahankan bentuk
       - Size 9: Menghilangkan fitur besar, menyederhanakan bentuk
    
    4. CLOSING:
       - Size 3: Menutup lubang sangat kecil
       - Size 5: Menutup celah sedang, menyambung bagian terputus
       - Size 9: Menyambung objek yang berdekatan
    
    5. SKELETON: Mengekstrak kerangka/tulang punggung objek
    
    6. TOP-HAT: Menonjolkan fitur terang yang lebih kecil dari SE
    7. BOTTOM-HAT: Menonjolkan fitur gelap yang lebih kecil dari SE
    """
    
    axes[10, 2].text(0.1, 0.5, analysis_text, transform=axes[10, 2].transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[10, 2].axis('off')
    axes[10, 3].axis('off')
    axes[10, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil_morfologi.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    setup_plot()
    
    # Buat citra sample
    img1, img2 = create_sample_images()
    
    # Terapkan operasi morfologi
    results1 = apply_morphology_operations(img1, "Citra 1: Huruf A")
    results2 = apply_morphology_operations(img2, "Citra 2: Bentuk Geometris")
    
    # Plot hasil
    plot_results(results1, results2)
    
    # Tampilkan analisis per operasi
    print("ANALISIS SINGKAT PER OPERASI:")
    print("1. DILASI: Memperbesar objek, menyambung bagian terputus, mengisi lubang kecil")
    print("2. EROSI: Memperkecil objek, menghilangkan noise, memisahkan objek menempel")
    print("3. OPENING: Menghilangkan noise di luar objek, menghaluskan kontur")
    print("4. CLOSING: Menutup lubang dalam objek, menyambung celah")
    print("5. SKELETON: Mengekstrak struktur garis tengah objek")
    print("6. TOP-HAT: Mendeteksi fitur terang kecil pada background gelap")
    print("7. BOTTOM-HAT: Mendeteksi fitur gelap kecil pada background terang")