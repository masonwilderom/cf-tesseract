import argparse, socket, struct, time
from collections import deque
from pathlib import Path

import cv2, numpy as np, pytesseract

COMMANDS            = {"S", "U", "D", "F", "B", "R", "L", "G"}
DEST_IP, DEST_PORT  = "127.0.0.1", 9000
FRAME_SKIP          = 4
GAMMA               = 0.6
CLAHE_CLIP          = 3.0
CLAHE_TILE          = (8, 8)
ADAPT_BLOCK, ADAPT_C = 31, 7
SAVE_DIR            = Path("stream_out")

parser = argparse.ArgumentParser(description="AI‑deck viewer with letter‑OCR")
parser.add_argument("-n", default="192.168.4.1", help="AI‑deck IP")
parser.add_argument("-p", type=int, default=5000, help="AI‑deck port")
parser.add_argument("--save", action="store_true", help="Save frames to disk")
args = parser.parse_args()

# sockets
print(f"Connecting to {args.n}:{args.p} …")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.connect((args.n, args.p))
print("Socket connected ✓")
sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def rx_bytes(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part: raise ConnectionError("stream closed")
        buf.extend(part)
    return bytes(buf)

# helpers
GAMMA_LUT = np.array([((i / 255.) ** GAMMA) * 255 for i in range(256)], dtype="uint8")
CLAHE  = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
KERNEL = np.ones((3, 3), np.uint8)
_recent, frame_counter, t0 = deque(maxlen=3), 0, time.time()
if args.save: SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ───────── main loop ─────────────────────────────────────────────────────── #
while True:
    length, _, _ = struct.unpack('<HBB', rx_bytes(4))
    hdr = rx_bytes(length - 2)
    try:   magic, w, h, depth, fmt, size = struct.unpack('<BHHBBI', hdr)
    except struct.error: continue
    if magic != 0xBC: continue

    img = bytearray()
    while len(img) < size:
        blk_len, _, _ = struct.unpack('<HBB', rx_bytes(4))
        img.extend(rx_bytes(blk_len - 2))

    if fmt == 0:  # Bayer
        bayer = np.frombuffer(img, np.uint8).reshape(h, w)
        frame = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
    else:         # JPEG
        frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: continue

    frame_counter += 1
    if frame_counter % FRAME_SKIP == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.LUT(gray, GAMMA_LUT)
        enhanced = CLAHE.apply(gray)
        scale = 1000 / max(enhanced.shape)
        resized = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        binary = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,ADAPT_BLOCK,ADAPT_C)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, KERNEL, 1)
        text = pytesseract.image_to_string(
            binary, config="--psm 7 --oem 3 -c tessedit_char_whitelist=SUD FBRLG").strip().upper()
        print(f"[OCR raw] '{text}'")

        for cmd in COMMANDS:
            if text == cmd:
                now = time.time()
                if not any(cmd == c and now - ts < 1 for c, ts in _recent):
                    sock_cmd.sendto(cmd.encode(), (DEST_IP, DEST_PORT))
                    print(f"[OCR] >>> {cmd}")
                    _recent.append((cmd, now))
                break

    cv2.imshow("AI‑deck Feed  (ESC quits)", frame)
    if args.save: cv2.imwrite(str(SAVE_DIR / f"frame_{frame_counter:06d}.jpg"), frame)
    if cv2.waitKey(1) & 0xFF == 27: break

sock.close(); sock_cmd.close(); cv2.destroyAllWindows()
print(f"{frame_counter} frames processed in {time.time()-t0:.1f}s")
