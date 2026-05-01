import serial
import time


class LSSArmController:
    def __init__(self, port='/dev/ttyUSB0', baud=921600, timeout=0.05):
        """Initializes the serial connection to the LSS arm."""
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.connect()

    def connect(self):
        """Establishes the serial connection."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            print(f"Connected to LSS Arm on {self.port}")
        except Exception as e:
            print(f"Connection Error: {e}")
            self.ser = None

    def get_position(self, servo_id):
        """Queries a single servo for its current position in degrees."""
        if not self.ser:
            return None
            
        self.ser.reset_input_buffer()
        cmd = f"#{servo_id}QD\r"
        self.ser.write(cmd.encode('ascii'))
        
        response = self.ser.read_until(b'\r').decode('ascii').strip()
        
        if response.startswith(f"*{servo_id}QD"):
            try:
                # Extract numerical value (in tenths of a degree)
                raw_val = int(response.split('QD')[-1])
                return raw_val / 10.0
            except ValueError:
                return None
        return None

    def move_joints(self, joint_angles, duration_ms=1000):
        """
        Moves joints to specific positions.
        :param joint_angles: List of angles in degrees [j1, j2, j3, j4, j5, j6]
        :param duration_ms: Duration of movement in milliseconds
        """
        if not self.ser:
            print("Serial connection not active.")
            return

        full_command = ""
        for i, pos in enumerate(joint_angles):
            servo_id = i + 1
            pos_tenths = int(pos * 10)
            # Format: #<ID>D<Value>T<Duration>
            full_command += f"#{servo_id}D{pos_tenths}T{duration_ms}"
        
        full_command += "\r"
        self.ser.write(full_command.encode('ascii'))

    def close(self):
        """Closes the serial port."""
        if self.ser:
            self.ser.close()
            print("Connection closed.")