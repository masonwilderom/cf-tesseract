#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# AI-deck / Webcam viewer  ·  Robust single-letter OCR  ·  UDP command sender
#
# Commands: S U D F B R L G
#
# Usage examples
# --------------
# Default (AI-deck):
#   python opencv-viewer.py -n 192.168.4.1 -p 5000
# Laptop webcam:
#   python opencv-viewer.py --mode webcam
# External cam index 2, forward UDP to another host:
#   python opencv-viewer.py --mode webcam --cam-index 2 --dest-ip 192.168.4.42
#

import argparse, socket, struct, time, cv2, numpy as np, pytesseract
from collections import deque
from pathlib import Path

# ───────── parameters ───────────────────────────────────────────────────── #
COMMANDS            = {"S","U","D","F","B","R","L","G"}
FRAME_SKIP          = 3                      # OCR every N frames
GAMMA               = 0.5
CLAHE_CLIP          = 4.0
CLAHE_TILE          = (8,8)
MASK_THRESH         = 200                    # dark-pixel mask threshold
SAVE_DIR            = Path("stream_out")
# ─────────────────────────────────────────────────────────────────────────── #

# ───────── CLI ──────────────────────────────────────────────────────────── #
parser = argparse.ArgumentParser(description="OCR viewer for AI-deck or webcam")
parser.add_argument("--mode", choices=["aideck","webcam"], default="aideck",
                    help="aideck (default) or webcam")
parser.add_argument("-n", default="192.168.4.1", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, help="AI-deck port")
parser.add_argument("--cam-index", type=int, default=0,
                    help="OpenCV camera index for webcam mode")
parser.add_argument("--dest-ip", default="127.0.0.1",
                    help="UDP destination IP (controller host)")
parser.add_argument("--dest-port", type=int, default=9000,
                    help="UDP destination port (controller)")
parser.add_argument("--save", action="store_true", help="Save frames to disk")
args = parser.parse_args()

DEST_IP, DEST_PORT = args.dest_ip, args.dest_port

# ───────── UDP socket for sending commands ──────────────────────────────── #
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ───────── image-processing helpers ─────────────────────────────────────── #
gamma_lut = np.array([((i/255.)**GAMMA)*255 for i in range(256)], dtype="uint8")
clahe     = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
kernel    = np.ones((5,5), np.uint8)
psm_cfg   = "--psm 10 --oem 3 -c tessedit_char_whitelist=SUDFBRLG"

# ---------- build Hu-moment templates (single uppercase letters) ---------- #
def build_templates(font_scale=5, thickness=12, img_size=200):
    templates={}
    for ch in COMMANDS:
        img = np.zeros((img_size, img_size), np.uint8)
        cv2.putText(img, ch, (10, img_size-10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness,
                    lineType=cv2.LINE_AA)
        _, bin_img = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt=max(cnts,key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)  # log-scale invariants
        templates[ch] = hu
    return templates

HU_TEMPLATES = build_templates()

# ---------- core recogniser ---------------------------------------------- #
def extract_letter(gray: np.ndarray) -> str:
    """
    Locate largest dark glyph on bright background and return the
    recognised command letter using Tesseract, falling back to Hu-moments.
    """
    # 1. contrast enhancement
    proc = cv2.LUT(gray, gamma_lut)
    proc = clahe.apply(proc)

    # 2. Sauvola local threshold to isolate ink
    win, R, k = 35, 128, 0.25
    f = proc.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (win, win))
    sqmean = cv2.boxFilter(f*f, -1, (win, win))
    std = np.sqrt(np.maximum(sqmean - mean*mean, 0))
    sauvola_mask = (f < mean*(1 + k*((std/R)-1))).astype(np.uint8)*255

    # 3. morphology
    mask = cv2.morphologyEx(sauvola_mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.dilate(mask, kernel, 1)

    # 4. largest component = glyph
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return ""
    cnt = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    if w*h < 8000: return ""           # too small / noise

    roi = proc[y:y+h, x:x+w]
    roi = cv2.resize(roi, (140,140), interpolation=cv2.INTER_CUBIC)
    _, roi_bin = cv2.threshold(roi,0,255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Stage 1 – Tesseract
    t_txt = pytesseract.image_to_string(roi_bin, config=psm_cfg).strip().upper()
    if t_txt and len(t_txt)==1 and t_txt in COMMANDS:
        return t_txt

    # Stage 2 – Hu-moment template match
    cnts2,_ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts2: return ""
    glyph = max(cnts2,key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(glyph)).flatten()
    hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)

    best,ch_best = 1e9,""
    for ch, tpl in HU_TEMPLATES.items():
        d = np.linalg.norm(hu - tpl)
        if d < best:
            best, ch_best = d, ch
    return ch_best if best < 2.5 else ""

# ---------- shared per-frame processing ---------------------------------- #
def process_frame(frame: np.ndarray, recent: deque):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    letter = extract_letter(gray)
    matched = letter in COMMANDS
    print(f"[OCR raw] '{letter}' ({'OK' if matched else 'no match'})")
    if matched:
        now = time.time()
        if not any(letter == c and now - ts < 1 for c, ts in recent):
            udp_sock.sendto(letter.encode(), (DEST_IP, DEST_PORT))
            print(f"[OCR] >>> {letter}")
            recent.append((letter, now))

# ───────── AI-deck capture loop ─────────────────────────────────────────── #
def run_aideck():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to AI-deck {args.n}:{args.p} …")
    tcp.connect((args.n, args.p)); print("Socket connected ✓")

    def rx_bytes(n:int)->bytes:
        buf=bytearray()
        while len(buf)<n:
            part=tcp.recv(n-len(buf))
            if not part: raise ConnectionError("AI-deck stream closed")
            buf.extend(part)
        return bytes(buf)

    frame_cnt, recent = 0, deque(maxlen=3)
    if args.save: SAVE_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        length,_,_=struct.unpack('<HBB', rx_bytes(4))
        hdr = rx_bytes(length-2)
        try: magic,w,h,_,fmt,size = struct.unpack('<BHHBBI',hdr)
        except struct.error: continue
        if magic != 0xBC: continue

        payload=bytearray()
        while len(payload)<size:
            blk_len,_,_=struct.unpack('<HBB', rx_bytes(4))
            payload.extend(rx_bytes(blk_len-2))

        if fmt==0:  # Bayer
            frame=cv2.cvtColor(np.frombuffer(payload,np.uint8).reshape(h,w),
                               cv2.COLOR_BayerBG2BGR)
        else:       # JPEG
            frame=cv2.imdecode(np.frombuffer(payload,np.uint8),
                               cv2.IMREAD_COLOR)
            if frame is None: continue

        frame_cnt += 1
        if frame_cnt % FRAME_SKIP == 0:
            process_frame(frame, recent)

        cv2.imshow("AI-deck Feed  (ESC quits)", frame)
        if args.save:
            cv2.imwrite(str(SAVE_DIR/f"aideck_{frame_cnt:06d}.jpg"), frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    tcp.close()

# ───────── Webcam capture loop ──────────────────────────────────────────── #
def run_webcam():
    cam = cv2.VideoCapture(args.cam_index)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam_index}")
    print(f"Webcam {args.cam_index} opened ✓")

    frame_cnt, recent = 0, deque(maxlen=3)
    if args.save: SAVE_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️  Frame capture failed"); break
        frame_cnt += 1
        if frame_cnt % FRAME_SKIP == 0:
            process_frame(frame, recent)

        cv2.imshow("Webcam Feed  (ESC quits)", frame)
        if args.save:
            cv2.imwrite(str(SAVE_DIR/f"webcam_{frame_cnt:06d}.jpg"), frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cam.release()

# ───────── dispatch & cleanup ───────────────────────────────────────────── #
try:
    if args.mode == "aideck":
        run_aideck()
    else:
        run_webcam()
finally:
    udp_sock.close()
    cv2.destroyAllWindows()
