import serial
import time

# Configuration
PORT = '/dev/ttyUSB0'
BAUD = 921600
TIMEOUT = 0.05
CONVEYOR_ID = 0  # Change this to your conveyor motor's ID


def initialize_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        return ser
    except Exception as e:
        print(f"Error: {e}")
        return None


def set_conveyor_speed(ser, motor_id, speed_dps):
    """
    Sets the conveyor speed in degrees per second (DPS).
    Positive = Clockwise, Negative = Counter-Clockwise.
    Example: 360 = 1 rotation per second.
    """
    # Format: #<ID>WD<Speed in tenths of deg/s>\r
    # Note: Some LSS firmware versions use WD for degrees per second directly 
    # or in tenths. We'll use the standard 'WD' command.
    speed_val = int(speed_dps * 10)
    cmd = f"#{motor_id}WD{speed_val}\r"
    
    ser.write(cmd.encode('ascii'))
    print(f"Conveyor speed set to: {speed_dps}°/s")


def stop_conveyor(ser, motor_id):
    """Stops the motor (Halt)."""
    cmd = f"#{motor_id}H\r"
    ser.write(cmd.encode('ascii'))
    print("Conveyor halted.")


def main():
    ser = initialize_serial()
    if not ser:
        return

    try:
        # 1. Start the conveyor at 180 degrees per second (0.5 RPM)
        set_conveyor_speed(ser, CONVEYOR_ID, 180)
        time.sleep(2)

        # 2. Start the conveyor at 180 degrees per second (0.5 RPM)
        set_conveyor_speed(ser, CONVEYOR_ID, -180)
        time.sleep(2)

        # 3. Stop
        stop_conveyor(ser, CONVEYOR_ID)

    except KeyboardInterrupt:
        stop_conveyor(ser, CONVEYOR_ID)
    finally:
        ser.close()

if __name__ == "__main__":
    main()