##########################################
#  Sutton & Barto - RL, An Introduction  #
##########################################


import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array

# fmt: off

track_0 = np.array([
    [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
])

def increase_track_size(track : np.ndarray) -> np.ndarray:
    track = track.copy()
    track = np.concatenate((track, track[-1][np.newaxis, :]), axis=0)
    track = np.concatenate((track, track.T[-1][:, np.newaxis]), axis=1)
    return track

track_L = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
])


track_R = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],     
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],     
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],     
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],   
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
])


# fmt: on

param_list = [
    {"track": track_0, "track_name": "track_0"},
    # {"track": track_L, "track_name": "track_L"},
    # {"track": track_R, "track_name": "track_R"},
]

for step in range(1, 5):
    track_0 = increase_track_size(track_0)
    param_list.append({"track": track_0, "track_name": "track_{}".format(step)})

default_params = param_list[0]


class Model(GenericModel):
    def __init__(self, params: dict = default_params):
        self.params = params
        self.track = params["track"]
        self.track_name = params["track_name"]

        self.build_matrices()
        self.name = "{}_{}_sutton_track_{}".format(
            self.state_dim, self.action_dim, self.track_name
        )

    def build_state_space(self):
        self.state_space = []
        for x in range(self.X):
            for y in range(self.Y):
                for sx in range(6):
                    for sy in range(6):
                        self.state_space = self.update_state_space_if_on_track(
                            self.state_space, sx, sy, x, y
                        )

    def update_state_space_if_on_track(
        self, state_space: list, sx: int, sy: int, x: int, y: int
    ) -> list:
        if sx + sy and self.check_on_track(x, y):
            state_space.append((x, y, sx, sy))
        return state_space

    def build_state_index(self):
        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

    def build_matrices(self):
        self.Y, self.X = self.track.shape
        # A state is made of coordinate and speed : (x, y, sx, sy)
        self.build_state_space()
        self.build_state_index()

        # An action is made of two increments (ax, ay)
        self.action_space = {(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)}
        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }
        self.state_dim = len(self.state_space)
        self.action_dim = len(self.action_space)

    def _build_model(self):
        self.build_matrices()

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = -1 * np.ones(
            (self.state_dim, self.action_dim), dtype=np.float16
        )
        for a, (ax, ay) in enumerate(self.action_space):
            for s1, (x, y, sx, sy) in enumerate(self.state_space):
                new_sx, new_sy = self.compute_new_speed(sx, sy, ax, ay)
                new_x, new_y, out_track = self.compute_new_position(x, y, sx, sy)
                if out_track:
                    self.reward_matrix[s1, a] = -5

                s2 = self.state_space_index[(new_x, new_y, new_sx, new_sy)]
                self.transition_matrix[a][s1, s2] += 0.5

                if new_x < self.X - 1 and self.check_on_track(new_x + 1, new_y):
                    s3 = self.state_space_index[(new_x + 1, new_y, new_sx, new_sy)]
                    self.transition_matrix[a][s1, s3] = 0.25
                else:
                    self.transition_matrix[a][s1, s2] += 0.25

                if new_y < self.Y - 1 and self.check_on_track(new_x, new_y + 1):
                    s4 = self.state_space_index[(new_x, new_y + 1, new_sx, new_sy)]
                    self.transition_matrix[a][s1, s4] = 0.25
                else:
                    self.transition_matrix[a][s1, s2] += 0.25

                if x == self.X:
                    self.reward_matrix[s1, a] = 0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def check_on_track(self, x, y):
        """
        Checks if the point of coordinates (x, y) is on the track
        """
        if not (x < self.X and y < self.Y):
            return False
        else:
            return bool(self.track[self.Y - y - 1, x])

    def compute_new_speed(self, sx, sy, ax, ay):
        """
        From the speed (sx, sy), compute the next
        speed using the action (ax, ay)
        """
        new_sx = max(0, min(5, sx + ax))
        new_sy = max(0, min(5, sy + ay))
        if new_sx == 0 and new_sy == 0:
            return sx, sy
        else:
            return new_sx, new_sy

    def evolve_x(self, x, y):
        assert self.check_on_track(x, y)
        if self.check_on_track(x + 1, y):
            return x + 1, True
        else:
            return x, False

    def evolve_y(self, x, y):
        assert self.check_on_track(x, y)
        if self.check_on_track(x, y + 1):
            return y + 1, True
        else:
            return y, False

    def compute_new_position(self, x, y, sx, sy):
        """
        For a position (x, y) and a speed (sx, sy),
        compute new position (x, y).
        """
        assert self.check_on_track(x, y)
        speed_x, speed_y = sx, sy
        evolved_x, evolved_y = True, True
        for _ in range(sx + sy):
            if speed_x > 0:
                x, evolved_x = self.evolve_x(x, y)
                speed_x -= 1

            if speed_y > 0:
                y, evolved_y = self.evolve_y(x, y)
                speed_y -= 1

        return x, y, evolved_x and evolved_y
