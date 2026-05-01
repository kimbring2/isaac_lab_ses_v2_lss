import serial
import time

# --- Configuration ---
PORT = '/dev/ttyUSB0'
BAUD = 921600
TIMEOUT = 0.05
SERVO_IDS = range(1, 7)  # IDs 1, 2, 3, 4, 5, 6


def initialize_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        return ser
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_position(ser, servo_id):
    """Queries a single servo for its current position."""
    ser.reset_input_buffer()
    cmd = f"#{servo_id}QD\r"
    ser.write(cmd.encode('ascii'))
    
    # Wait for response like *1QD<value>\r
    response = ser.read_until(b'\r').decode('ascii').strip()
    print(f"response from servo {servo_id}: {response}")

    if response.startswith(f"*{servo_id}QD"):
        try:
            # Extract numerical value (in tenths of a degree)
            raw_val = int(response.split('QD')[-1])
            return raw_val / 10.0  # Convert to degrees
        except ValueError:
            return None

    return None



def move_joints(ser, j1, j2, j3, j4, j5, j6, duration_ms=1000):
    """
    Moves all 6 joints to specific positions.
    Positions are in degrees. 
    duration_ms: How long the move should take (e.g., 1000 = 1 second).
    """
    joints = [j1, j2, j3, j4, j5, j6]
    
    # We build one large string to send all commands at once.
    # The LSS can handle multiple commands in one go if they are formatted correctly.
    full_command = ""
    
    for i, pos in enumerate(joints):
        servo_id = i + 1
        # Convert degrees to tenths of a degree (e.g., 45.5 -> 455)
        pos_tenths = int(pos * 10)
        
        # Format: #<ID>D<Value>T<Duration>
        # Adding 'T' ensures all motors reach the destination together.
        full_command += f"#{servo_id}D{pos_tenths}T{duration_ms}"
    
    # End the multi-command string with a carriage return
    full_command += "\r"
    
    ser.write(full_command.encode('ascii'))
    print(f"Moving to: {joints} over {duration_ms}ms")


def main():
    ser = initialize_serial()
    if not ser:
        return

    try:
        # Example 1: Move all to 0 quickly (500ms)
        move_joints(ser, -176, 0, 0, 0, 0, 0, 500)
        
        #move_joints(ser, -175, 50, -45, -80, 45, 10, 500)
        time.sleep(1)

        # 2. Verify and print positions for each motor
        print("\n--- Verifying Positions ---")
        for s_id in SERVO_IDS:
            pos = get_position(ser, s_id)
            if pos is not None:
                print(f"Motor {s_id}: {pos}°")
            else:
                print(f"Motor {s_id}: No response (Check ID/Wiring)")
    except KeyboardInterrupt:
        print("\nStopped.")



if __name__ == "__main__":
    main()