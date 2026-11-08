import pygame

class InputHandler:
    def __init__(self):
        self.active_keys = [] 

    def update(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key not in self.active_keys:
                    self.active_keys.append(event.key)
                    
            elif event.type == pygame.KEYUP:
                if event.key in self.active_keys:
                    self.active_keys.remove(event.key)

        if not self.active_keys:
            return self._get_action(None)
            
        newest_key = self.active_keys[-1]
        
        return self._get_action(newest_key)

    def _get_action(self, key_code):
        match key_code:        
            case pygame.K_LEFT: return 0 # left
            case pygame.K_RIGHT: return 1 # right
            case pygame.K_DOWN: return 2 # faster down 
            case pygame.K_UP: return 3 # rotation
            case pygame.K_z: return 4 # rotation
            case pygame.K_SPACE: return 5 # instant down
            case pygame.K_c | pygame.K_LSHIFT: return 6 # hold and change 
            case _: return 7