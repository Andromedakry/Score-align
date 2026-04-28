import cv2
import numpy as np
import sys
import os
import glob

def get_horizontal_projection_variance(img_binary):
    proj = np.sum(img_binary, axis=1)
    return np.var(proj)

def fine_tune_angle(img_binary, rough_angle, search_range=1.0, step=0.05):
    best_angle = rough_angle
    best_score = -1
    h, w = img_binary.shape
    center = (w // 2, h // 2)
    for delta in np.arange(-search_range, search_range + step, step):
        angle = rough_angle + delta
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(img_binary, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        score = get_horizontal_projection_variance(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle

def detect_skew_projection_precise(image, angle_range=5, step=0.2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binary.shape
    center = (w // 2, h // 2)
    best_angle = 0
    best_score = -1
    
    for angle in np.arange(-angle_range, angle_range + step, step):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(binary, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        score = get_horizontal_projection_variance(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
    
    if step > 0.05:
        best_angle = fine_tune_angle(binary, best_angle, search_range=step, step=0.05)
    
    return best_angle

def auto_deskew_sheetmusic(image_path, output_path, min_angle=0.02):
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    angle = detect_skew_projection_precise(img, angle_range=6, step=0.2)
    print(f"Detected angle: {angle:.4f}°")
    
    if abs(angle) < min_angle:
        cv2.imwrite(output_path, img)
        print(f"Angle too small (< {min_angle}°), no correction.")
        return True
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    deskewed = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    cv2.imwrite(output_path, deskewed)
    return True

def batch_process(input_pattern, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = glob.glob(input_pattern)
    if not files:
        print(f"No files found: {input_pattern}")
        return
    for idx, f in enumerate(files):
        base = os.path.basename(f)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(output_folder, f"{name}_corrected{ext}")
        success = auto_deskew_sheetmusic(f, out_path)
        status = "OK" if success else "FAIL"
        print(f"[{idx+1}/{len(files)}] {base} -> {status}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python deskew.py input.jpg output.jpg")
        print("  Batch: python deskew.py --batch \"*.jpg\" output_folder")
        sys.exit(1)
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Batch usage: python deskew.py --batch \"pattern\" output_folder")
            sys.exit(1)
        pattern = sys.argv[2]
        out_dir = sys.argv[3]
        batch_process(pattern, out_dir)
    else:
        if len(sys.argv) < 3:
            print("Single file usage: python deskew.py input.jpg output.jpg")
            sys.exit(1)
        inp = sys.argv[1]
        outp = sys.argv[2]
        ok = auto_deskew_sheetmusic(inp, outp)
        if ok:
            print(f"Success: {outp}")
        else:
            print("Failed to process image.")