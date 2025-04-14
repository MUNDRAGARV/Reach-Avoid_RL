import numpy as np
import cv2
import time

# Dubins car class
class DubinsCar:
    def __init__(self, x, y, theta, v=0.5, omega=0.833):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega

    def update(self, u):
        self.x += self.v * np.cos(self.theta)
        self.y += self.v * np.sin(self.theta)
        self.theta += u

# PID controller for angle control
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute_control(self, x, y, theta):
        target_angle = np.arctan2(-y, -x)  # since goal is at (0,0)
        error = self._angle_diff(target_angle, theta)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Discretize to {-omega, 0, omega}
        if u > 0.1:
            return car.omega
        elif u < -0.1:
            return -car.omega
        else:
            return 0

    def _angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

# Parameters
v = 0.5
omega = 0.833
R = 1.0  # Outer constraint radius
r = 0.5  # Goal radius

# Initialize car and controller
car = DubinsCar(x=0.8, y=0.0, theta=np.pi/2, v=v, omega=omega)
pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.5)

# Visualization setup
scale = 0.01  # meters to pixels
width = 400
height = 400
origin = (width // 2, height // 2)

max_steps = 500
reached_goal = False

for step in range(max_steps):
    if not reached_goal:
        control = pid.compute_control(car.x, car.y, car.theta)
        car.update(control)

        # Constraint check
        distance_from_origin = np.sqrt(car.x**2 + car.y**2)
        if distance_from_origin >= R:
            print("❌ Left constraint region")
            break

        # Goal check
        if np.sqrt(car.x**2 + car.y**2) <= r:
            print("✅ Reached goal!")
            reached_goal = True

    # Render
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, origin, int(R / scale), (255, 0, 255), 2)  # Outer constraint
    cv2.circle(img, origin, int(r / scale), (0, 255, 255), -1)  # Goal
    car_pos = (int(origin[0] + car.x / scale), int(origin[1] - car.y / scale))
    heading = (
        int(car_pos[0] + 10 * np.cos(car.theta)),
        int(car_pos[1] - 10 * np.sin(car.theta))
    )
    cv2.arrowedLine(img, car_pos, heading, (0, 255, 0), 2)
    cv2.circle(img, car_pos, 4, (0, 255, 0), -1)

    cv2.imshow("Dubins Car - Hybrid Controller", img)

    key = cv2.waitKey(0)  # Slow down
    if key == ord('q'):
        break

print("Simulation ended. Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()