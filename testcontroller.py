import logging
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.power_switch import PowerSwitch

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

from cflib.positioning.motion_commander import MotionCommander

# URI to the Crazyflie to connect to
uri = 'radio://0/80/2M/E7E7E7E701'  # TODO: change the uri based on your crazyflie number
'''
1: radio://0/80/2M/E7E7E7E701
2: radio://0/80/2M/E7E7E7E702
3: radio://0/80/2M/E7E7E7E703
4: radio://0/90/2M/E7E7E7E704
5: radio://0/90/2M/E7E7E7E705
6: radio://0/90/2M/E7E7E7E706
7: radio://0/100/2M/E7E7E7E707
8: radio://0/100/2M/E7E7E7E708
9: radio://0/100/2M/E7E7E7E709
'''

deck_attached_event = Event()


def simple_connect():
    # Test connection. You can use this function to check whether you can connect to crazyflie
    print("Yeah, I'm connected! :D")
    time.sleep(3)
    print("Now I will disconnect :'(")


def read_parameter(scf, logconf):
    """
    Q1. Command #1: "i" - read and print roll, pitch, yaw from the Crazyflie.
    """
    print("Reading roll, pitch, yaw...")
    # Create a logger instance that will get one log entry and then exit
    try:
        with SyncLogger(scf, logconf) as logger:
            # Get one data sample from the logger
            data = next(logger)
            # data[1] is a dictionary with logged variables
            roll = data[1]['stabilizer.roll']
            pitch = data[1]['stabilizer.pitch']
            yaw = data[1]['stabilizer.yaw']
            print("Roll: {:.2f}, Pitch: {:.2f}, Yaw: {:.2f}".format(roll, pitch, yaw))
    except Exception as e:
        print("Error reading parameters:", e)


def moving_up(mc, dis):
    """
    Q3. Command #3: "u x" - move up x meters.
    """
    print("Moving up {:.2f} m".format(dis))
    mc.up(dis)
    time.sleep(0.5)


def moving_down(mc, dis):
    """
    Q4. Command #4: "d x" - move down x meters.
    """
    print("Moving down {:.2f} m".format(dis))
    mc.down(dis)
    time.sleep(0.5)


def forwarding(mc, dis):
    """
    Q5. Command #5: "f x" - move forward x meters.
    """
    print("Moving forward {:.2f} m".format(dis))
    mc.forward(dis)
    time.sleep(0.5)


def backwarding(mc, dis):
    """
    Q6. Command #6: "b x" - move backward x meters.
    """
    print("Moving backward {:.2f} m".format(dis))
    mc.back(dis)
    time.sleep(0.5)


def turning_left(mc, deg):
    """
    Q7. Command #7: "l x" - turn left x degrees.
    """
    print("Turning left {:.2f}°".format(deg))
    mc.turn_left(deg)
    time.sleep(0.5)


def turning_right(mc, deg):
    """
    Q8. Command #8: "r x" - turn right x degrees.
    """
    print("Turning right {:.2f}°".format(deg))
    mc.turn_right(deg)
    time.sleep(0.5)


def landing(mc):
    """
    Q9. Command #9: "n" - land the crazyflie.
    """
    print("Landing ...")
    mc.stop()  # The MotionCommander stops further motion.
    # When the MotionCommander context exits, landing is automatically performed.


def param_deck_flow(_, value_str):
    """
    Callback to check whether positioning deck is attached.
    """
    value = int(value_str)
    print("Deck parameter value:", value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')


def fly_commander(scf, lg_stab, mc, SLEEP_TIME=0.3):
    """
    After take-off, process commands for flying. Commands:
      i: read parameters
      u/d/f/b/l/r: movement commands
      n: land (exits fly_commander)
      e: exit immediately
    """
    print("=== Fly Commander ===")
    command = ""
    while command != 'e':
        command = input("Enter fly command (i/u/d/f/b/l/r/n/e): ").strip()
        if command == 'e':
            break
        elif command == 'i':
            read_parameter(scf, lg_stab)
        elif command[0] == 'u':  # up
            try:
                dis = float(command.split()[1])
                moving_up(mc, dis)
            except Exception as e:
                print("Error parsing 'u' command:", e)
        elif command[0] == 'd':  # down
            try:
                dis = float(command.split()[1])
                moving_down(mc, dis)
            except Exception as e:
                print("Error parsing 'd' command:", e)
        elif command[0] == 'f':  # forward
            try:
                dis = float(command.split()[1])
                forwarding(mc, dis)
            except Exception as e:
                print("Error parsing 'f' command:", e)
        elif command[0] == 'b':  # backward
            try:
                dis = float(command.split()[1])
                backwarding(mc, dis)
            except Exception as e:
                print("Error parsing 'b' command:", e)
        elif command[0] == 'l':  # turn left
            try:
                deg = float(command.split()[1])
                turning_left(mc, deg)
            except Exception as e:
                print("Error parsing 'l' command:", e)
        elif command[0] == 'r':  # turn right
            try:
                deg = float(command.split()[1])
                turning_right(mc, deg)
            except Exception as e:
                print("Error parsing 'r' command:", e)
        elif command == 'n':
            landing(mc)
            return  # Exit fly_commander


def base_commander(scf, lg_stab, DEFAULT_HEIGHT=0.5, SLEEP_TIME=0.3):
    """
    The main command processor. Here the 's' command causes the Crazyflie
    to take off and then enters fly_commander to process further flight commands.
    """
    print("=== Base Commander ===")
    command = ""
    while command != 'e':
        command = input("Enter base command (i/s/e): ").strip()
        if command == 'e':
            print("Exiting base commander.")
            break
        elif command == 'i':
            read_parameter(scf, lg_stab)
        elif command == 's':
            print("Crazyflie taking off ...")
            # When using MotionCommander in a context manager, take-off is performed
            # automatically to the DEFAULT_HEIGHT and landing is executed on exit.
            with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
                print("Take-off successful. Now you can fly!")
                fly_commander(scf, lg_stab, mc)
            # After fly_commander returns, the context manager will ensure landing.
        else:
            print("Unknown command. Please use 'i' (info), 's' (take off), or 'e' (exit).")


if __name__ == '__main__':
    # Initialize low-level drivers
    cflib.crtp.init_drivers()

    # Setup logging for the stabilizer data (roll, pitch, yaw)
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    # Connect to Crazyflie using a synchronous interface.
    logging.basicConfig(level=logging.ERROR)
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        # Add a parameter update callback to detect whether the positioning deck is attached.
        scf.cf.param.add_update_callback(group="deck", name="bcLighthouse4",
                                         cb=param_deck_flow)
        time.sleep(1)  # Wait a moment for parameter update

        # Start the base commander: this will wait for your input commands.
        base_commander(scf, lg_stab)
