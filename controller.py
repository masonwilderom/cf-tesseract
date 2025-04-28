"""
Crazyflie UDP command listener (single‑letter protocol)

Letters:
  S  → take‑off (start)
  U  → up 1 m      | D → down 1 m
  F  → forward 1 m | B → back 1 m
  R  → turn right 45° | L → turn left 45°
  G  → ground / land
"""

import logging, socket, time
from threading import Event

import cflib.crtp
from cflib.crazyflie           import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log       import LogConfig

# ───────── configuration ─────────────────────────────────────────────────── #
URI          = 'radio://0/80/2M/E7E7E7E701'
HOST, PORT   = "0.0.0.0", 9000
DIST, ANGLE  = 1.0, 45.0
DEFAULT_H    = 0.5
# ─────────────────────────────────────────────────────────────────────────── #

def _listen_udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((HOST, PORT)); s.setblocking(False)
    print(f"Listening on UDP {HOST}:{PORT}")
    return s

def _recv(s:socket.socket):
    try: data, _ = s.recvfrom(32); return data.decode().strip().upper()
    except BlockingIOError: return None

def _handle(cmd:str, mc:MotionCommander):
    if   cmd == "U": mc.up(DIST)
    elif cmd == "D": mc.down(DIST)
    elif cmd == "F": mc.forward(DIST)
    elif cmd == "B": mc.back(DIST)
    elif cmd == "L": mc.turn_left(ANGLE)
    elif cmd == "R": mc.turn_right(ANGLE)

# ───────── main ──────────────────────────────────────────────────────────── #
logging.basicConfig(level=logging.ERROR)
cflib.crtp.init_drivers()

lg = LogConfig(name="Stab", period_in_ms=100)
for v in ("stabilizer.roll","stabilizer.pitch","stabilizer.yaw"): lg.add_variable(v,"float")

sock = _listen_udp()
print("Waiting for S (start) …")

while (cmd:=_recv(sock)) != "S": time.sleep(0.05)

print("S received – taking off")
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    with MotionCommander(scf, default_height=DEFAULT_H) as mc:
        flying = True
        while flying:
            cmd = _recv(sock)
            if   cmd in {"U","D","F","B","L","R"}: _handle(cmd, mc)
            elif cmd == "G": print("G received – landing"); flying=False
            time.sleep(0.05)

sock.close()
print("Flight session ended.")
