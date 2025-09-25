import cv2
import numpy as np
import datetime
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self):
        """Inisialisasi YOLO model"""
        self.model = YOLO('yolov8n.pt')  # menggunakan model nano untuk performa lebih cepat
        
    def detect_objects(self, frame):
        """Deteksi objek dalam frame"""
        results = self.model(frame, verbose=False)
        result = results[0]
        
        detections = []
        if hasattr(result, 'boxes'):
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                confidence = float(box.conf)
                
                # Get class name
                class_name = result.names[int(box.cls)]
                
                if confidence > 0.5:  # Filter deteksi dengan confidence > 0.5
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2]
                    })
        
        return detections

class MotionDetector:
    def __init__(self, threshold=25, min_area=5000):
        """
        Inisialisasi motion detector
        
        Args:
            threshold: Sensitivitas deteksi gerakan (semakin rendah semakin sensitif)
            min_area: Area minimum untuk mendeteksi gerakan
        """
        self.threshold = threshold
        self.min_area = min_area
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect_motion(self, frame):
        """
        Mendeteksi gerakan dalam frame
        
        Args:
            frame: Frame dari kamera
            
        Returns:
            motion_detected: True jika terdeteksi gerakan
            processed_frame: Frame yang sudah diproses dengan bounding box
        """
        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Background subtraction
        mask = self.background_subtractor.apply(gray)
        
        # Morphological operations untuk menghilangkan noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Cari kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        processed_frame = frame.copy()
        
        # Proses setiap kontur yang ditemukan
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Hanya proses kontur yang cukup besar
            if area > self.min_area:
                motion_detected = True
                
                # Gambar bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Tambahkan label
                cv2.putText(processed_frame, "MOTION DETECTED", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return motion_detected, processed_frame, mask

def get_color_for_class(class_name):
    """Mendapatkan warna berbeda untuk setiap kelas objek"""
    color_map = {
        # Manusia dan hewan
        'person': (255, 0, 0),     # Biru
        'dog': (0, 255, 0),        # Hijau
        'cat': (0, 0, 255),        # Merah
        
        # Perangkat elektronik
        'cell phone': (255, 255, 0),    # Cyan
        'laptop': (255, 0, 255),        # Magenta
        'keyboard': (0, 255, 255),      # Kuning
        'tv': (128, 0, 0),              # Biru tua
        'mouse': (0, 128, 0),           # Hijau tua
        
        # Perabotan
        'chair': (0, 0, 128),           # Merah tua
        'couch': (128, 128, 0),         # Olive
        'bed': (128, 0, 128),           # Ungu
        'dining table': (0, 128, 128),  # Teal
        
        # Peralatan dapur
        'cup': (192, 192, 0),           # Kuning tua
        'bottle': (192, 0, 192),        # Pink
        'wine glass': (0, 192, 192),    # Turquoise
        'bowl': (128, 128, 128),        # Abu-abu
        
        # Transportasi
        'car': (255, 128, 0),           # Oranye
        'bicycle': (128, 255, 0),       # Lime
        'motorcycle': (0, 255, 128),    # Spring green
        
        # Lainnya
        'book': (255, 0, 128),          # Pink tua
        'clock': (128, 0, 255),         # Ungu terang
        'vase': (0, 128, 255),          # Biru muda
    }
    # Warna default untuk objek lain dengan warna random tapi konsisten
    if class_name not in color_map:
        # Generate warna pseudo-random tapi konsisten berdasarkan nama kelas
        hash_value = sum(ord(c) for c in class_name)
        r = (hash_value * 123) % 255
        g = (hash_value * 456) % 255
        b = (hash_value * 789) % 255
        color_map[class_name] = (r, g, b)
    
    return color_map[class_name]

def draw_results(frame, motion_detected, detections):
    """Menggambar hasil deteksi pada frame"""
    # Dictionary untuk menghitung objek per kelas
    class_counts = {}
    processed_frame = frame.copy()
    
    # Gambar hasil deteksi objek
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Update counter
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Gambar bounding box dengan warna sesuai kelas
        color = get_color_for_class(class_name)
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        
        # Calculate size of object relative to frame
        obj_width = x2 - x1
        obj_height = y2 - y1
        frame_area = frame.shape[0] * frame.shape[1]
        obj_area = obj_width * obj_height
        size_percent = (obj_area / frame_area) * 100
        
        # Add size information to label
        size_info = "kecil" if size_percent < 10 else "sedang" if size_percent < 30 else "besar"
        
        # Tambah label dengan nama kelas, ukuran dan confidence score
        label = f"{class_name} ({size_info}) {confidence:.2f}"
        
        # Add background to text for better visibility
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(processed_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(processed_frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Status gerakan
    status_text = "MOTION DETECTED" if motion_detected else "NO MOTION"
    status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Info jumlah objek yang terdeteksi
    y_offset = 60
    for class_name, count in class_counts.items():
        count_text = f"{class_name}s detected: {count}"
        color = get_color_for_class(class_name)
        cv2.putText(frame, count_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
    
    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)  # 0 untuk kamera default
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera")
        return
    
    # Set resolusi kamera (opsional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Inisialisasi detectors
    motion_detector = MotionDetector(threshold=25, min_area=3000)
    object_detector = ObjectDetector()
    
    print("Intelligent Camera System dimulai...")
    print("Tekan 'q' untuk keluar")
    print("Tekan 's' untuk mengambil screenshot")
    print("\nObjek yang bisa dideteksi dengan warna:")
    print("- Orang (Biru)")
    print("- Gelas/Cup (Hijau)")
    print("- Handphone (Merah)")
    print("- Botol (Cyan)")
    print("- Laptop (Magenta)")
    print("- Keyboard (Kuning)")
    print("- Objek lain (Abu-abu)")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera")
            break
        
        # Deteksi gerakan
        motion_detected, motion_frame, mask = motion_detector.detect_motion(frame)
        
        # Deteksi objek dengan YOLO
        detections = object_detector.detect_objects(frame)
        
        # Proses frame dengan deteksi objek dan informasi gerakan
        processed_frame = draw_results(frame, motion_detected, detections)
        
        # Tambahkan informasi tambahan ke frame
        # - Status gerakan dengan highlight
        status_text = "MOTION DETECTED" if motion_detected else "NO MOTION"
        status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
        cv2.rectangle(processed_frame, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(processed_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # - Timestamp dengan background
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(processed_frame, 
                     (5, processed_frame.shape[0]-35),
                     (200, processed_frame.shape[0]-5),
                     (0, 0, 0), -1)
        cv2.putText(processed_frame, timestamp,
                   (10, processed_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # - Statistik deteksi
        stats_y = 70
        cv2.rectangle(processed_frame, (5, 45), (200, 45 + 30 * len(set([d['class'] for d in detections]))), (0, 0, 0), -1)
        
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        for class_name, count in class_counts.items():
            color = get_color_for_class(class_name)
            text = f"{class_name}: {count}"
            cv2.putText(processed_frame, text, (10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            stats_y += 30
        
        # Tampilkan frame
        cv2.imshow('Intelligent Camera System', processed_frame)
        cv2.imshow('Motion Mask', mask)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Simpan screenshot
            filename = f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Screenshot disimpan: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")

if __name__ == "__main__":
    main()