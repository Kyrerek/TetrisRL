import gymnasium as gym
import pygame
from tetris_gymnasium.envs.tetris import Tetris
from InputHandler import InputHandler

if __name__ == "__main__":
    pygame.init()

    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)
    inputH = InputHandler()
    
    clock = pygame.time.Clock()
    terminated = False
    while not terminated:
        env.render()

        events = pygame.event.get()
    
        #random
        #action = env.action_space.sample()
        #human
        action = inputH.update(events)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        clock.tick(10)
    env.close()
    print("Game Over!")


