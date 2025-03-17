import numpy as np 
import random 
import cv2 
import tensorflow as tf  
from collections import deque  
import pandas as pd

WIDTH, HEIGHT = 500, 500 

VY = 15  # y-direction velocity
VX = 15  # x-direction velocity

ACTIONS = [-1, 0, 1]  # (-1: Move Left, 0: Stay, 1: Move Right)

obstacles = [
    ((WIDTH/4, HEIGHT/2), (100, 50)),  # Obstacle at (125,250) with size 100x50
    ((WIDTH/2, HEIGHT/2), (80, 50)),   # Obstacle at (250,250) with size 80x50
    ((WIDTH*0.75, HEIGHT/2), (120, 60))   # Obstacle at (375,250) with size 120x60
]

target = ((250, 50), (80, 50))  # Target is at (250, 50) with size 80x50

class ParticleEnv:
    def __init__(self):
        self.bad_positions = set()  # previous bad decision
        self.reset()  

    def reset(self):
        while True:
            self.state = [np.random.randint(0, WIDTH), HEIGHT]  # Start at a random position at the bottom
            if tuple(self.state) not in self.bad_positions and not self.is_in_obstacle_or_target(self.state):  #valid start position
                break
        return np.array(self.state) / np.array([WIDTH, HEIGHT])  # Normalize the state

    def is_in_obstacle_or_target(self, state):
        x, y = state  # Extract x and y coordinates
        for (px, py), (lx, ly) in obstacles:  
            if abs(x - px) < lx / 2 and abs(y - py) < ly / 2:  # Check if in obstacle 
                return True  # Collision detected
        (tx, ty), (tlx, tly) = target  # target position and size
        if abs(x - tx) < tlx / 2 and abs(y - ty) < tly / 2:  # Check if in target 
            return True  # Reached target
        return False  # No collision detected

    def step(self, action):
        prev_state = self.state.copy()  # Store previous state
        u = ACTIONS[action]  # movement direction
        self.state[0] += u * VX  # Update x position based on action
        self.state[1] -= VY  # Move upward

        for (px, py), (lx, ly) in obstacles:  # Check for collision with obstacles
            if abs(self.state[0] - px) < lx / 2 and abs(self.state[1] - py) < ly / 2:
                self.bad_positions.add(tuple(prev_state))  # Mark bad position
                print("Hit obstacle")
                return self.reset(), -100, True, {} 

        (tx, ty), (tlx, tly) = target  # Get target position
        if abs(self.state[0] - tx) < tlx / 2 and abs(self.state[1] - ty) < tly / 2:
            print("Reached target")
            return self.reset(), 200, True, {}  # Reward for reaching target

        if self.state[0] < 0 or self.state[0] > WIDTH or self.state[1] < 0:  # Check if out of bounds
            self.bad_positions.add(tuple(prev_state))  # Mark bad position
            print("Hit the wall")
            return self.reset(), -10, True, {}  

        return np.array(self.state) / np.array([WIDTH, HEIGHT]), -1, False, {}  # Return new state

    def render(self):
        img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255  

        for (px, py), (lx, ly) in obstacles:  # obstacles
            px, py, lx, ly = map(int, [px, py, lx, ly])
            cv2.rectangle(img, (px - lx // 2, py - ly // 2), (px + lx // 2, py + ly // 2), (0, 0, 255), -1)
        
        (tx, ty), (tlx, tly) = target  # target
        tx, ty, tlx, tly = map(int, [tx, ty, tlx, tly])
        cv2.rectangle(img, (tx - tlx // 2, ty - tly // 2), (tx + tlx // 2, ty + tly // 2), (0, 255, 0), -1)
        
        cv2.circle(img, (int(self.state[0]), int(self.state[1])), 5, (0, 255, 255), -1)  # Draw player
        cv2.imshow("Environment", img)
        cv2.waitKey(50)  


class DDQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)  # to store past experiences
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.98  
        self.learning_rate = 0.001  # Learning rate
        self.model = self.build_model()  # Build the main model
        self.target_model = self.build_model()  # Build the target model
        self.episode_data = [] # for storing in excel
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=2, activation="relu"), 
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(ACTIONS), activation="linear") 
        ])
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))  
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Store experiences
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Explore with probability epsilon
            return random.randrange(len(ACTIONS))
        q_values = self.model.predict(np.array([state]), verbose=0)  # Get Q-values
        return np.argmax(q_values[0])  # Choose the best action
    
    def replay(self):
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, 32)  # Sample batch from memory to use for training the player
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(np.array([next_state]), verbose=0))
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
env = ParticleEnv()
agent = DDQNAgent()

episodes = 2000
decay_interval = 50  # Decay epsilon every 50 episodes

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for time in range(500):
        env.render()
        action = agent.act(state)  # Let the agent decide the action
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)  # Store experience
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            break
    
    agent.replay()  # Train the agent
    agent.update_target_model()  # Update the target network

    # Decay epsilon every `decay_interval` episodes
    if (e + 1) % decay_interval == 0 and agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay  # Reduce exploration rate

    agent.episode_data.append([e+1, total_reward, agent.epsilon])

cv2.destroyAllWindows()
df = pd.DataFrame(agent.episode_data, columns=["Episode", "Reward", "Epsilon"])
df.to_excel("training_data.xlsx", index=False)
