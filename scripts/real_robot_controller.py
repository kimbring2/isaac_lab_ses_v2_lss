from lss_controller import LSSArmController
import time

# 1. Initialize the arm
arm = LSSArmController(port='/dev/ttyUSB0')

# 2. Move joints (Pass as a list)
#target_positions = [-176, 0, 0, 0, 45, 0]
target_positions = [-176, 35, -25, -80, 0, 0]
arm.move_joints(target_positions, duration_ms=500)

# Give it time to move
time.sleep(1)

# 3. Get positions
for i in range(1, 7):
    pos = arm.get_position(i)
    print(f"Servo {i} is at: {pos} degrees")

# 4. Always good practice to close
arm.close()