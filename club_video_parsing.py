from collections import Counter
import cv2
import os
import numpy as np
from cv2.typing import MatLike
import utils.opencv as opencv

# constants
TRUE_RATIO = 3.89
MIN_RATIO = TRUE_RATIO - 0.2
MAX_RATIO = TRUE_RATIO + 0.2
# ===
CLUB_HEADER_COLOR = "#7fcc0b"
# ===
ROW_HEADER_COLOR = "#e4ddd2"
ROW_BACKGROUND_COLOR = "#ffffff"
ROW_SELF_BACKGROUND_COLOR = "#fff4c6"
ROW_KEY_BACKGROUND = "#ece7e4"
# == 
ICON_I_GRADIENT_TOP_COLOR = "#ffffff"
ICON_I_GRADIENT_BOTTOM_COLOR = "#fafafa"
# ===
LEADER_FLAG_COLOR = "#ef3c39"
OFFICER_FLAG_COLOR = "#267fe9"
MEMBER_FLAG_COLOR = "#5dca10"

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16) 
    b = int(hex_color[4:6], 16)
    return np.array([b, g, r])

def replace_color(image: MatLike, from_color: str, to_color: str, tolerance: int = 0) -> MatLike:
    from_bgr = hex_to_bgr(from_color)
    to_bgr = hex_to_bgr(to_color)
    diff = np.abs(image.astype(np.int16) - from_bgr.astype(np.int16))
    mask = np.all(diff <= tolerance, axis=2)
    result = image.copy()
    result[mask] = to_bgr
    return result

def create_binary_mask(image: MatLike, target_colors: list[str], tolerance: int = 0) -> MatLike:
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    for color in target_colors:
        target_bgr = hex_to_bgr(color)
        diff = np.abs(image.astype(np.int16) - target_bgr.astype(np.int16))
        color_mask = np.all(diff <= tolerance, axis=2)
        combined_mask = combined_mask | color_mask
    binary_image = np.zeros_like(image)
    binary_image[combined_mask] = [255, 255, 255]
    binary_image[~combined_mask] = [0, 0, 0]
    return binary_image

def remove_noise(image: MatLike, min_area: int = 50) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_gray = gray.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            cv2.fillPoly(result_gray, [contour], 0)
    result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
    return result

def find_white_regions(image: MatLike, threshold: float = 0.5, min_width: int = 50, min_height: int = 20) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or h < min_height:
            continue
        roi = gray[y:y+h, x:x+w]
        white_pixels = np.sum(roi == 255)
        total_pixels = roi.size
        white_ratio = white_pixels / total_pixels
        if white_ratio >= threshold:
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def expand_white_areas(image: MatLike, radius: int) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(gray, kernel, iterations=1)
    result = cv2.cvtColor(expanded, cv2.COLOR_GRAY2BGR)
    return result

def shrink_white_areas(image: MatLike, radius: int) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    shrunk = cv2.erode(gray, kernel, iterations=1)
    result = cv2.cvtColor(shrunk, cv2.COLOR_GRAY2BGR)
    return result

def find_contours_containing_boxes(image: MatLike, target_boxes: list[tuple[int, int, int, int]], min_ratio: float =  MIN_RATIO, max_ratio: float = MAX_RATIO) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    matching_contour_boxes = []
    for contour in contours:
        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
        if contour_h == 0:
            continue
        aspect_ratio = contour_w / contour_h
        if not (min_ratio <= aspect_ratio <= max_ratio):
            continue
        for target_x, target_y, target_w, target_h in target_boxes:
            target_x2 = target_x + target_w
            target_y2 = target_y + target_h
            contour_x2 = contour_x + contour_w
            contour_y2 = contour_y + contour_h
            if (contour_x <= target_x and contour_y <= target_y and 
                contour_x2 >= target_x2 and contour_y2 >= target_y2 and
                target_h / contour_h < 0.275):
                matching_contour_boxes.append((contour_x, contour_y, contour_w, contour_h))
                break
    return matching_contour_boxes

def detect_player_rows(image: MatLike):
    # Guard in case previous step failed
    if image is None:
        return []
    # replace self indicator background color from yellow to white, which is any other member background color
    p_image = replace_color(image, ROW_SELF_BACKGROUND_COLOR, ROW_BACKGROUND_COLOR, 10)
    headers = remove_noise(create_binary_mask(p_image, [ROW_HEADER_COLOR], 5), 2)
    headers = shrink_white_areas(headers, 1)
    headers = find_white_regions(headers, 0.5, 10, 10)
    p_image = create_binary_mask(p_image, [ROW_HEADER_COLOR, ROW_BACKGROUND_COLOR, ROW_KEY_BACKGROUND], 5)
    p_image = remove_noise(p_image, 5)
    p_image = expand_white_areas(p_image, 2)
    boxes = find_contours_containing_boxes(p_image, headers)
    return boxes

# def detect_player_rows(image: MatLike, frame_idx: int = None, crop_y: int = None):
#     if image is None:
#         return []

#     debug_root = "debug_stages"
#     os.makedirs(debug_root, exist_ok=True)

#     # Make unique subfolder per frame/crop
#     tag = f"frame{frame_idx}_y{crop_y}" if frame_idx is not None and crop_y is not None else "unknown"
#     debug_dir = os.path.join(debug_root, tag)
#     os.makedirs(debug_dir, exist_ok=True)

#     def save_debug(img, name):
#         cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)

#     save_debug(image, "0_original")

#     # Replace self indicator background
#     p_image = replace_color(image, ROW_SELF_BACKGROUND_COLOR, ROW_BACKGROUND_COLOR, 10)
#     save_debug(p_image, "1_replaced")

#     headers = create_binary_mask(p_image, [ROW_HEADER_COLOR], 5)
#     save_debug(headers, "2_mask_headers")

#     headers = remove_noise(headers, 2)
#     save_debug(headers, "3_noise_removed")

#     headers = shrink_white_areas(headers, 1)
#     save_debug(headers, "4_shrunk")

#     headers_boxes = find_white_regions(headers, 0.5, 10, 10)
#     print(f"[DEBUG][{tag}] Header boxes found: {headers_boxes}")

#     p_image = create_binary_mask(p_image, [ROW_HEADER_COLOR, ROW_BACKGROUND_COLOR, ROW_KEY_BACKGROUND], 5)
#     save_debug(p_image, "5_combined_mask")

#     p_image = remove_noise(p_image, 5)
#     save_debug(p_image, "6_cleaned")

#     p_image = expand_white_areas(p_image, 2)
#     save_debug(p_image, "7_expanded")

#     boxes = find_contours_containing_boxes(p_image, headers_boxes)
#     print(f"[DEBUG][{tag}] Contour boxes found: {boxes}")

#     return boxes

def resize_image(image: MatLike, height: int) -> MatLike:
    # if the image is screenshoted from mobile, return the original image
    if image.shape[0] / 2 > image.shape[1]:
        return image
    return cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))

def fill_area(image: MatLike, area: tuple[int, int, int, int], color: str) -> MatLike:
    x, y, w, h = area
    image[y:y+h, x:x+w] = hex_to_bgr(color)
    return image

def crop_image(image: MatLike, box: tuple[int, int, int, int]):
    x, y, w, h = box
    return image[y:y+h, x:x+w]

def cleanup_image_before_ocr(image: MatLike) -> MatLike:
    w = int(image.shape[1] * 0.23)
    h = int(image.shape[0] * 0.25)
    image = fill_area(image, (image.shape[1] - w, 0, w, h), ROW_HEADER_COLOR)
    header_row = crop_image(image, (int(image.shape[1] * 0.215), int(h * 0.1), int(image.shape[1] * 0.8), int(h * 0.8)))
    mask = create_binary_mask(header_row, [ICON_I_GRADIENT_TOP_COLOR, ICON_I_GRADIENT_BOTTOM_COLOR], 10)
    mask = expand_white_areas(mask, int(h / 7.5))
    areas = find_white_regions(mask, 0.5, int(h * 0.5), int(h * 0.5))
    for x, _, w2, _ in areas:
        image = fill_area(image, (int(image.shape[1] * 0.215 + x), 0, int(image.shape[1] * 0.1 + h), h), ROW_HEADER_COLOR)
    return image

def ocr_image(image: MatLike) -> list[str]:
    """Run PaddleOCR and normalize output to list[str] (compatible with v2–v5)."""
    if not opencv.is_paddleocr_initialized():
        opencv.init_paddleocr()

    result = opencv.ocr.ocr(image)
    texts = []

    # ✅ Handle new PaddleOCR v5 format (dict-based)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        for page in result:
            if "rec_texts" in page:
                texts.extend(page["rec_texts"])
        return texts

    # ✅ Handle legacy PaddleOCR v2 format (list-based)
    for line in result or []:
        if not line:
            continue
        for item in line:
            try:
                texts.append(item[1][0])
            except Exception:
                pass

    return texts

def get_optimization_info(image: MatLike):
    step1 = create_binary_mask(image, [CLUB_HEADER_COLOR], 50)
    step2 = remove_noise(step1, 4000)
    boxes = find_white_regions(step2, 0.5, 10, 10)
    if len(boxes) == 0:
        return None
    # Be robust: choose the widest box instead of failing when multiple are found
    max_box = max(boxes, key=lambda b: b[2])
    return max_box

def optimize(image: MatLike):
    resized_image = resize_image(image, 960)
    info = get_optimization_info(resized_image)
    if info is None:
        return None
    return crop_image(resized_image, (
        max(info[0] - 10, 0),
        0,
        min(info[2] + 20, resized_image.shape[1]),
        resized_image.shape[0],
    ))

def to_fps(capture: cv2.VideoCapture, fps: int):
    current_fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        if current_fps is None or current_fps <= 0 or np.isnan(current_fps):
            current_fps = float(fps)
    except Exception:
        current_fps = float(fps)
    step = max(1, int(round(current_fps / float(fps))))
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if frame_count % step == 0:
            yield frame
        frame_count += 1

def parse_only_numbers(text: str) -> int:
    ret = 0
    for ch in text:
        if ch.isdigit():
            ret = ret * 10 + int(ch)
    return ret

def parse_last_login(text: str) -> int:
    num = parse_only_numbers(text)
    if 's' in text:
        return num
    elif 'm' in text:
        return num * 60
    elif 'h' in text:
        return num * 60 * 60
    elif 'd' in text:
        return num * 60 * 60 * 24
    return num

def reconstruct_paths(edges):
    adj = {}
    indegree = {}
    nodes = set()
    for u, v in edges:
        adj[u] = v
        indegree[v] = indegree.get(v, 0) + 1
        indegree[u] = indegree.get(u, 0)
        nodes.add(u); nodes.add(v)
    starts = [n for n in nodes if indegree[n] == 0]
    paths = []; visited = set()
    for start in starts:
        path = []; cur = start
        while cur in adj and cur not in visited:
            path.append(cur); visited.add(cur); cur = adj[cur]
        if cur not in visited:
            path.append(cur); visited.add(cur)
        paths.append(path)
    return paths

def get_captured_player_info_images(iterator):
    ret = []
    for frame_idx, frame in enumerate(iterator):
        if frame_idx % 10 == 0:
            print(f"[INFO] Processing frame {frame_idx}...")
        optimized_frame = optimize(frame)
        if optimized_frame is None:
            # Could not detect header in this frame; skip it
            continue
        boxes = detect_player_rows(optimized_frame)
        if not boxes:
            continue
        id = 0
        for box in boxes:
            image = crop_image(optimized_frame, box)
            _, y, _, _ = box
            ret.append((image, frame_idx, y))
            id = id + 1
    return ret

def extract_from_ocr_results(texts: list[str]):
    # Normalize all OCR text: lowercase, trim, and collapse multiple spaces
    normalized_texts = [' '.join(e.lower().strip().split(' ')) for e in texts]

    # Merge related tokens like "total" + "fans" or "last" + "login"
    merged_texts = []
    skip_next = False
    for i in range(len(normalized_texts)):
        if skip_next:
            skip_next = False
            continue
        token = normalized_texts[i]
        next_token = normalized_texts[i + 1] if i + 1 < len(normalized_texts) else ''
        if token == "total" and next_token == "fans":
            merged_texts.append("total fans")
            skip_next = True
        elif token == "last" and next_token == "login":
            merged_texts.append("last login")
            skip_next = True
        else:
            merged_texts.append(token)
    normalized_texts = merged_texts

    try:
        # The first token is assumed to be the role
        role = normalized_texts[0]

        # Extract the name portion (everything before "total fans")
        name = ' '.join(texts[1:normalized_texts.index("total fans")]).strip()

        # Keep only the first word (the player name, drop titles)
        name = name.split(" ")[0]

        # Extract numerical info
        total_fans = parse_only_numbers(normalized_texts[normalized_texts.index("total fans") + 1])
        last_login = parse_last_login(normalized_texts[normalized_texts.index("last login") + 1])

        # --- Sanity checks and normalization ---
        role = role.lower().strip()
        name = name.lower().strip()

        # If OCR swapped role and name (e.g., name == "members"), swap them back
        if name in ["members", "officer", "leader"]:
            role, name = name, role

        # Reject empty names
        if name == '':
            return False, None

        return True, (role, name, total_fans, last_login)
    except Exception:
        return False, None

def vote_by_majority(records: dict[str, list[dict[str, int]]]):
    ret = {}
    for name, record_by_frame in records.items():
        role_counter = Counter(e["role"] for e in record_by_frame)
        total_fans_counter = Counter(e["total_fans"] for e in record_by_frame)
        last_login_counter = Counter(e["last_login"] for e in record_by_frame)
        ret[name] = (role_counter.most_common(1)[0][0], total_fans_counter.most_common(1)[0][0], last_login_counter.most_common(1)[0][0])
    return ret

def get_order_relationship(records: dict[str, list[dict[str, int]]]):
    ret = set()
    for k1, first in records.items():
        for k2, second in records.items():
            if k1 == k2:
                break
            if (k1, k2) in ret or (k2, k1) in ret:
                break
            for first_frame in first:
                for second_frame in second:
                    if first_frame["frame_idx"] == second_frame["frame_idx"]:
                        if first_frame["frame_box_y"] < second_frame["frame_box_y"]:
                            ret.add((k1, k2))
                        else:
                            ret.add((k2, k1))
                        break
                else:
                    continue
                break
    return ret

def merge_group_with_same_groundtruth_inplace(records: dict[str, list[dict[str, int]]], groundtruths: dict[str, tuple[str, int, int]]):
    # Merges groups with identical majority-voted ground truth
    names = list(groundtruths.keys())
    for i, name in enumerate(names):
        if name not in records:
            continue
        for j in range(i+1, len(names)):
            name2 = names[j]
            if name2 not in records or name == name2:
                continue
            if groundtruths[name] == groundtruths[name2]:
                if len(records[name]) >= len(records[name2]):
                    records[name].extend(records[name2])
                    del records[name2]
                else:
                    records[name2].extend(records[name])
                    del records[name]
                    break

def extract_player_info(images):
    ret: dict[str, list[dict[str, int]]] = {}
    for image, frame_idx, y in images:
        texts = ocr_image(image)
        success, data = extract_from_ocr_results(texts)
        if not success:
            print(f"[INFO] → OCR parse failed (didn't match expected format). Data: {texts}")
            continue

        role, name, total_fans, last_login = data
        print(f"[INFO] → Parsed: role={role}, name={name}, fans={total_fans}, login={last_login}")

        if name not in ret:
            ret[name] = []
        ret[name].append({
            "role": role,
            "total_fans": total_fans,
            "last_login": last_login,
            "frame_idx": frame_idx,
            "frame_box_y": y,
        })
    return ret

def extract_video(path: str, fps: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video. Check path and codecs.")
    print("[INFO] Starting frame extraction and player detection...")
    images = get_captured_player_info_images(to_fps(cap, fps))
    print(f"[INFO] Finished: {len(images)} player info images captured.")
    if not images:
        cap.release()
        raise RuntimeError("No usable frames detected (header not found). Try a clearer video or different resolution.")
    player_data_group_by_name = extract_player_info(images)
    groundtruth_by_group = vote_by_majority(player_data_group_by_name)
    merge_group_with_same_groundtruth_inplace(player_data_group_by_name, groundtruth_by_group)
    order_relationship = get_order_relationship(player_data_group_by_name)
    reconstructed_paths = reconstruct_paths(order_relationship)
    cap.release()
    if len(reconstructed_paths) == 0:
        print("\n[DEBUG] --- DATA COLLECTION SUMMARY ---")
        print(f"Frames with captured player info images: {len(images)}")
        print(f"Players detected by OCR: {len(player_data_group_by_name)}")
        if player_data_group_by_name:
            for name, records in player_data_group_by_name.items():
                print(f"  - {name}: {len(records)} records")
                if records:
                    sample = records[0]
                    print(f"    Sample: role={sample['role']}, fans={sample['total_fans']}, login={sample['last_login']}")
        else:
            print("  [!] No OCR-detected players.")

        print(f"Order relationships found: {len(order_relationship)}")
        print(f"Groundtruth groups: {len(groundtruth_by_group)}")

        print("[DEBUG] No reconstructed paths could be formed.")
        print("[DEBUG] This usually means ordering (Y positions) or OCR failed to match between frames.\n")

        raise RuntimeError("No reconstructed paths found.")
    return [
        {
            "name": name,
            "role": groundtruth_by_group[name][0],
            "total_fans": groundtruth_by_group[name][1],
            "last_login": groundtruth_by_group[name][2],
        }
        for names in reconstructed_paths
        for name in names
    ]
