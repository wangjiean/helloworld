import gym
import time
env = gym("CartPole-v1", render_mode="rgb_array")
env.reset()
env.render()
time.sleep(2)