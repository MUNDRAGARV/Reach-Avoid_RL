import numpy as np
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import time
import pandas as pd

# --- PID Gain Action Space ---
Kp_vals = [1.0, 2.0, 3.0]
Ki_vals = [0.0, 0.1]
Kd_vals = [0.0, 0.5, 1.0]
pid_actions = [(kp, ki, kd) for kp in Kp_vals for ki in Ki_vals for kd in Kd_vals]  # 18 combos

# --- DDQN Model ---
def build_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)),  # State: x, y
        Dense(32, activation='relu'),
        Dense(len(pid_actions), activation='linear')  # Q-values for each PID combo
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

q_network = build_model()
target_network = build_model()
target_network.set_weights(q_network.get_weights())

# --- Dubins Car Environment ---
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

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute_control(self, x, y, theta):
        target_angle = np.arctan2(-y, -x)
        error = self._angle_diff(target_angle, theta)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        return np.clip(u, -omega, omega)

    def _angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

# --- Hyperparameters ---
epochs = 100
batch_size = 32
memory = deque(maxlen=2000)
discount = 0.95
epsilon = 1.0
epsilon_min = 0.01
eps_decay = 0.995
target_update_freq = 10

# --- Simulation Settings ---
R = 1.0
r = 0.5
v = 0.2 # reduced the speed for better visualisation
omega = 0.833
scale = 0.01
width, height = 400, 400
origin = (width // 2, height // 2)

# --- Training Loop ---
successful_pids = []

for episode in range(epochs):
    while True:
        spawn_x = random.uniform(-R, R)
        spawn_y = random.uniform(-R, R)
        if r < np.sqrt(spawn_x**2 + spawn_y**2) < R:
            break
    spawn_theta = random.uniform(0, 2 * np.pi)
    car = DubinsCar(spawn_x, spawn_y, spawn_theta, v=v, omega=omega)
    state = np.array([car.x, car.y])

    if np.random.rand() < epsilon:
        action_idx = np.random.randint(len(pid_actions))
    else:
        q_values = q_network.predict(state.reshape(1, -1), verbose=0)
        action_idx = np.argmax(q_values)

    kp, ki, kd = pid_actions[action_idx]
    pid = PIDController(kp, ki, kd)

    total_reward = 0
    max_steps = 500
    reached_goal = False

    for step in range(max_steps):
        control = pid.compute_control(car.x, car.y, car.theta)
        car.update(control)

        dist = np.sqrt(car.x**2 + car.y**2)

        if dist <= r:
            reward = 100
            done = True
            reached_goal = True
            print(f"✅ Reached goal with PID: Kp={kp}, Ki={ki}, Kd={kd}")
            successful_pids.append((kp, ki, kd))
        elif dist >= R:
            reward = -100
            done = True
            print(f"❌ Left constraint region with PID: Kp={kp}, Ki={ki}, Kd={kd}")
        else:
            reward = -1
            done = False

        next_state = np.array([car.x, car.y])
        memory.append((state, action_idx, reward, next_state, done))
        total_reward += reward
        state = next_state

        if step % 5 == 0 or done:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(img, origin, int(R / scale), (255, 0, 255), 2)
            cv2.circle(img, origin, int(r / scale), (0, 255, 255), -1)
            car_pos = (int(origin[0] + car.x / scale), int(origin[1] - car.y / scale))
            heading = (
                int(car_pos[0] + 10 * np.cos(car.theta)),
                int(car_pos[1] - 10 * np.sin(car.theta))
            )
            cv2.arrowedLine(img, car_pos, heading, (0, 255, 0), 2)
            cv2.circle(img, car_pos, 4, (0, 255, 0), -1)

            cv2.imshow("Dubins Car - RL PID", img)
            time.sleep(0.05) # added a time delay
            key = cv2.waitKey(50)
            if key == ord('q'):
                break

        if done:
            break

    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        states_mb = np.array([m[0] for m in minibatch])
        actions_mb = np.array([m[1] for m in minibatch])
        rewards_mb = np.array([m[2] for m in minibatch])
        next_states_mb = np.array([m[3] for m in minibatch])
        dones_mb = np.array([m[4] for m in minibatch])

        q_current = q_network.predict(states_mb, verbose=0)
        q_next_main = q_network.predict(next_states_mb, verbose=0)
        q_next_target = target_network.predict(next_states_mb, verbose=0)

        for i in range(batch_size):
            if dones_mb[i]:
                q_current[i][actions_mb[i]] = rewards_mb[i]
            else:
                next_act = np.argmax(q_next_main[i])
                q_current[i][actions_mb[i]] = rewards_mb[i] + discount * q_next_target[i][next_act]

        q_network.fit(states_mb, q_current, epochs=1, verbose=0)

    if episode % target_update_freq == 0:
        target_network.set_weights(q_network.get_weights())

    if epsilon > epsilon_min:
        epsilon *= eps_decay

    print(f"Episode {episode} - Total Reward: {total_reward:.2f} - Reached Goal: {reached_goal}")

print("Training complete.")
cv2.destroyAllWindows()

# --- Display Optimal PID Gains ---
if successful_pids:
    df = pd.DataFrame(successful_pids, columns=["Kp", "Ki", "Kd"])
    best_gains = df.value_counts().reset_index(name='Count').sort_values(by='Count', ascending=False)
    print("\nMost Successful PID Gains:")
    print(best_gains.head())
else:
    print("No successful PID parameters found.")