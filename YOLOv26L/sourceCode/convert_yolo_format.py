import os
import json
import shutil
import glob
from tqdm import tqdm

# ================= 설정 영역 =================
BASE_PATH = r"C:\gitnconda\Swin-Transformer\원본"
DATASET_REAL = os.path.join(BASE_PATH, "군 경계 작전 환경 내 인식 데이터")
DATASET_SYN = os.path.join(BASE_PATH, "군 경계 작전 환경 합성 데이터")

OUTPUT_DIR = r"C:\gitnconda\Swin-Transformer\yolo_format\Merged_Dataset"

TARGET_CLASSES = [
    "Fishing_Boat", "Merchant_Ship", "Warship", "Person", "Bird", 
    "Fixed_Wing", "Rotary_Wing", "UAV", "Leaflet", "Trash_Bomb"
]

MAP_REAL = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
MAP_SYN = {11: 0, 13: 1, 12: 2, 31: 4, 21: 5, 22: 6, 23: 7, 41: 8, 42: 9}
# ============================================

def convert_to_yolo_format(size, box, format_type):
    """
    좌표 포맷을 YOLO 표준 (center_x, center_y, w, h) 정규화 좌표로 변환
    """
    img_w, img_h = size[0], size[1]
    x_min, y_min, w_box, h_box = 0, 0, 0, 0

    # 1. 포맷 해석
    if format_type == 'xywh':
        # [x, y, w, h] -> REAL 데이터 (I1, I2)
        x_min = box[0]
        y_min = box[1]
        w_box = box[2]
        h_box = box[3]
        
    elif format_type == 'whxy':
        # [w, h, x, y] -> SYN 데이터 (EO, IR)
        # 사용자 발견 규칙: 뒤의 두 값이 좌표(x,y), 앞의 두 값이 크기(w,h)
        w_box = box[0]
        h_box = box[1]
        x_min = box[2]
        y_min = box[3]

    # 2. 좌표 계산 (x_max, y_max)
    x_max = x_min + w_box
    y_max = y_min + h_box
    
    # 3. 이미지 범위 벗어남 방지 (Clipping)
    x_min = max(0, min(x_min, img_w))
    y_min = max(0, min(y_min, img_h))
    x_max = max(0, min(x_max, img_w))
    y_max = max(0, min(y_max, img_h))
    
    # 보정된 너비/높이 재계산
    w_final = x_max - x_min
    h_final = y_max - y_min
    
    # 유효성 검사 (너무 작은 박스 제거)
    if w_final <= 1 or h_final <= 1:
        return None

    # 4. YOLO 정규화 (Center X, Center Y, W, H) -> 0.0 ~ 1.0
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    
    return (
        min(max(x_center / img_w, 0.0), 1.0),
        min(max(y_center / img_h, 0.0), 1.0),
        min(max(w_final / img_w, 0.0), 1.0),
        min(max(h_final / img_h, 0.0), 1.0)
    )

def get_image_path(json_path, img_filename):
    base_dir = os.path.dirname(json_path)
    img_dir_candidate = base_dir.replace("02.라벨링데이터", "01.원천데이터")
    
    if "TL_" in img_dir_candidate:
        img_dir_candidate = img_dir_candidate.replace("TL_", "TS_")
    elif "VL_" in img_dir_candidate:
        img_dir_candidate = img_dir_candidate.replace("VL_", "VS_")
        
    full_path = os.path.join(img_dir_candidate, img_filename)
    
    if os.path.exists(full_path):
        return full_path
    
    base_name, _ = os.path.splitext(full_path)
    for check_ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        temp_path = base_name + check_ext
        if os.path.exists(temp_path):
            return temp_path
    return None

def determine_format(filename):
    """
    파일명에 따른 포맷 결정 로직
    - I1, I2 포함: REAL 데이터 -> 'xywh'
    - 그 외 (EO, IR 등): SYN 데이터 -> 'whxy'
    """
    if "I1_" in filename or "I2_" in filename:
        return 'xywh'
    else:
        return 'whxy'

def process_file(json_path, dataset_type, split_name, error_log):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_filename = ""
        img_width = 0
        img_height = 0
        annotations = []

        # --- REAL 데이터 처리 ---
        if dataset_type == 'REAL':
            if not data.get("annotations"): return
            ann_info = data["annotations"][0]
            img_filename = ann_info["filename"]
            img_width = ann_info["width"]
            img_height = ann_info["height"]
            
            # 포맷 결정
            current_format = determine_format(img_filename)

            for ann in data["annotations"]:
                if str(ann["class"]) in MAP_REAL:
                    target_id = MAP_REAL[str(ann["class"])]
                    bbox = convert_to_yolo_format((img_width, img_height), ann["bbox"], current_format)
                    if bbox: annotations.append((target_id, bbox))

        # --- SYN 데이터 처리 ---
        elif dataset_type == 'SYN':
            img_info = data["image"]
            img_filename = img_info["filename"]
            img_width = img_info["width"]
            img_height = img_info["height"]
            
            # 포맷 결정 (EO, IR 등은 whxy로 처리됨)
            current_format = determine_format(img_filename)
            
            if "annotations" in data:
                for ann in data["annotations"]:
                    sub_cls = ann.get("sub_class")
                    if sub_cls in MAP_SYN:
                        target_id = MAP_SYN[sub_cls]
                        bbox = convert_to_yolo_format(
                            (img_width, img_height), 
                            ann["bounding_box"], 
                            current_format
                        )
                        if bbox: annotations.append((target_id, bbox))
        
        if not annotations: return 

        # --- 파일 복사 및 저장 ---
        src_img_path = get_image_path(json_path, img_filename)
        if not src_img_path:
            if len(error_log) < 10: 
                error_log.append(f"MISSING: {img_filename}")
            return

        original_filename = os.path.basename(src_img_path)
        dst_img_path = os.path.join(OUTPUT_DIR, "images", split_name, original_filename)
        dst_label_path = os.path.join(OUTPUT_DIR, "labels", split_name, os.path.splitext(original_filename)[0] + ".txt")
        
        shutil.copy2(src_img_path, dst_img_path)
        
        with open(dst_label_path, 'w') as lf:
            for cls_id, (cx, cy, w, h) in annotations:
                lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    except Exception as e:
        print(f"Error processing {json_path}: {e}")

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"기존 폴더 확인됨: {OUTPUT_DIR}")
        print("덮어쓰기 진행 중...")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    error_log = []

    # 1. REAL 데이터
    print("--- REAL 데이터 처리 ---")
    for split_raw, split_yolo in [("Training", "train"), ("Validation", "val")]:
        files = glob.glob(os.path.join(DATASET_REAL, split_raw, "**", "*.json"), recursive=True)
        for f in tqdm(files, desc=f"REAL {split_raw}"):
            process_file(f, 'REAL', split_yolo, error_log)

    # 2. SYN 데이터
    print("\n--- SYN 데이터 처리 ---")
    for split_raw, split_yolo in [("Training", "train"), ("Validation", "val")]:
        files = glob.glob(os.path.join(DATASET_SYN, split_raw, "**", "*.json"), recursive=True)
        for f in tqdm(files, desc=f"SYN {split_raw}"):
            process_file(f, 'SYN', split_yolo, error_log)

    # 3. data.yaml 생성
    yaml_content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}
    """
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), 'w') as f:
        f.write(yaml_content)

    print("\n=== 변환 완료 ===")
    if error_log:
        print(f"⚠ 이미지 누락: {len(error_log)}건")

if __name__ == "__main__":
    main()