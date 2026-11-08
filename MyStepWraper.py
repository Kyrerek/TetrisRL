import gymnasium as gym

class MyStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.rewards = {
            "game_over": -100.0,  
            "cleared_lines": {
                1: 1.0, 
                2: 3.0,
                3: 5.0,
                4: 10.0 
            }
        }

    def step(self, action):
        
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        removed_lines = info.get("lines_cleared", 0)
        
        game_over = terminated 
        
        frame_reward = 0.0
        
        if game_over:
            frame_reward += self.rewards["game_over"]
        else:
            if removed_lines > 0:
                lines_max_4 = min(removed_lines, 4) 
                frame_reward += self.rewards["cleared_lines"][lines_max_4]

        return obs, frame_reward, terminated, truncated, info