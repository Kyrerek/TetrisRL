import gymnasium as gym
import numpy as np
from functools import wraps

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
            },
            "holes": 0.1 # one size hole 
        }

        self.current_holes_count = 0
        self._piece_placed = False
        self._hook_commit_method()


    def _hook_commit_method(self):
        original_commit = self.env.unwrapped.commit_active_tetromino

        @wraps(original_commit)
        def wrapped_commit(*args, **kwargs):
            self._piece_placed = True
            return original_commit(*args, **kwargs)

        self.env.unwrapped.commit_active_tetromino = wrapped_commit

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_holes_count = 0
        self._piece_placed = False
        return self._binarize_obs(obs), info

    def step(self, action):
        
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        removed_lines = info.get("lines_cleared", 0)
        
        game_over = terminated 
        
        frame_reward = 0.0

        if game_over:
            frame_reward += self.rewards["game_over"]
        else:
            if self._piece_placed:
                n_holes = self._calc_holes(obs)
                if self.current_holes_count != n_holes:
                    frame_reward += (self.current_holes_count - n_holes) * abs(self.rewards["holes"])
                self._piece_placed = False
                self.current_holes_count = n_holes

            if removed_lines > 0:
                lines_max_4 = min(removed_lines, 4) 
                frame_reward += self.rewards["cleared_lines"][lines_max_4]

        obs = self._binarize_obs(obs)
        return obs, frame_reward, terminated, truncated, info


    def _calc_holes(self,obs):
        board = obs["board"]
        mask = obs["active_tetromino_mask"]
        
        stable_board = np.pad(np.where(mask==1,0,board),pad_width=1,mode="edge")

        rows, cols = stable_board.shape
        start_point = None
        for row in range(rows):
            for col in range(cols):
                if stable_board[row,col] == 0:
                    start_point = (row,col)
                    break
            if start_point is not None:
                break

        free_spaces = self._find_all_conected_dfs(stable_board,start_point)
        
        zero_count = int(np.sum(stable_board == 0))
        return zero_count - len(free_spaces)

    def _find_all_conected_dfs(self,board,start_point):
        rows,cols = board.shape
        visited = set()
        visited.add(start_point)

        def _neighbours_idx(point):
            idx,idy = point
            neighbours_pos_idxs = [(idx-1,idy),(idx+1,idy),(idx,idy-1),(idx,idy+1)]
            neighbours_idx = []
            for n_idx,n_idy in neighbours_pos_idxs:
                if board[n_idx,n_idy] == board[idx,idy]:
                    neighbours_idx.append((n_idx,n_idy))
            return neighbours_idx

        def _find_neighbours(start_point):
            nghbrs = _neighbours_idx(start_point)
            for nghbr in nghbrs:
                if nghbr not in visited:
                    visited.add(nghbr)
                    _find_neighbours(nghbr)

        _find_neighbours(start_point)


        return visited


    def _binarize_obs(self,obs):
        obs_bin = dict(obs)
        obs_bin["board"] = (obs["board"] > 0).astype(obs["board"].dtype)
        obs_bin["holder"] = (obs["holder"] > 0).astype(obs["holder"].dtype)
        obs_bin["queue"] = (obs["queue"] > 0).astype(obs["queue"].dtype)
        return obs_bin