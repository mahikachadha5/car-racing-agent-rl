import cv2
import numpy as np
import gymnasium as gym

def preprocess(img):
    # resizing the 96x96 img to 84x84
    img = img[:84, 6:90] # only take the first 84 rows (cut off black bar at bottom) and take the columns at the center of the track
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255.0
    return img
    
class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def reset(self):
        # reset original environment
        s, info = self.env.reset()
        
        # do nothing for the next 'self.inital_no_op' steps
        for i in range(self.initial_no_op):
            s,r,terminated,truncated,info = self.env.step(0)
            
        # convert a frame to the 84x84 greyscale one
        s = preprocess(s)
        
        # the initial observation is a copy of the frame 's'
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))
        return self.stacked_state, info
    
    def step(self, action):
        # we will take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        s = preprocess(s)
        
        # push the current frame at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
        
        return self.stacked_state, reward, terminated, truncated, info