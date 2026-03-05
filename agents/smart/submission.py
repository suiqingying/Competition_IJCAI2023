# -*- coding:utf-8  -*-
# IJCAI 2023 AI Olympics - Smart Agent
# Perfected Rule-based strategies for all 6 game scenarios

import numpy as np
import math

# ===================== Color Index Constants =====================
IDX_EMPTY = 0        # light green (empty space)
IDX_GREEN = 1         # green
IDX_SKY_BLUE = 2      # sky blue
IDX_ORANGE = 3        # orange
IDX_GREY = 4          # grey
IDX_PURPLE = 5        # purple
IDX_BLACK = 6         # black (wall)
IDX_RED = 7           # red (finish line)
IDX_BLUE = 8          # blue
IDX_WHITE = 9         # white
IDX_LIGHT_RED = 10    # light red

WALL_INDICES = {IDX_BLACK, IDX_GREY}
FINISH_INDICES = {IDX_RED}

# ===================== Game State Tracker =====================
class GameState:
    def __init__(self):
        self.game_type = None
        self.step_in_game = 0
        self.curling_launched = False
        self.curling_force_steps = 0
        self.curling_stone_count = 0
        
        # Anti-stuck mechanics
        self.last_obs_sum = 0
        self.stuck_counter = 0
        self.recovery_steps = 0

        # Global orientation tracking
        self.global_theta = 0.0

game_state = GameState()

# ===================== Observation Utilities =====================

def find_object_centroid(obs, target_indices):
    n = obs.shape[0]
    agent_row = int(n * 0.8)
    agent_col = n // 2

    target_mask = np.isin(obs, list(target_indices))
    positions = np.argwhere(target_mask)

    if len(positions) == 0:
        return None, None

    centroid_row = np.mean(positions[:, 0])
    centroid_col = np.mean(positions[:, 1])

    delta_forward = agent_row - centroid_row
    delta_right = centroid_col - agent_col
    return delta_forward, delta_right

def count_walls_in_front(obs, rows_ahead=12):
    n = obs.shape[0]
    agent_row = int(n * 0.8)
    agent_col = n // 2
    check_from = max(0, agent_row - rows_ahead)

    front_slice = obs[check_from:agent_row, :]
    wall_mask = np.isin(front_slice, list(WALL_INDICES))

    left_walls = np.sum(wall_mask[:, :agent_col - 3])
    center_walls = np.sum(wall_mask[:, agent_col - 3:agent_col + 4])
    right_walls = np.sum(wall_mask[:, agent_col + 4:])

    return left_walls, center_walls, right_walls

def steer_toward(delta_forward, delta_right, gain=2.0):
    if delta_forward is None:
        return 0.0
    angle = math.degrees(math.atan2(delta_right, max(delta_forward, 0.5)))
    return max(-30.0, min(30.0, angle * gain))

# ===================== Anti-Stuck Logic =====================
def check_stuck(obs):
    global game_state
    current_sum = np.sum(obs)
    if current_sum == game_state.last_obs_sum:
        game_state.stuck_counter += 1
    else:
        game_state.stuck_counter = 0
        game_state.last_obs_sum = current_sum

    if game_state.stuck_counter > 8:
        game_state.recovery_steps = 6
        game_state.stuck_counter = 0

    if game_state.recovery_steps > 0:
        game_state.recovery_steps -= 1
        return True
    return False

# ===================== Strategy Implementations =====================

def strategy_running(obs, obs_dict):
    if check_stuck(obs):
        return [-200.0, 30.0]  # Back up and turn sharp

    df, dr = find_object_centroid(obs, FINISH_INDICES)
    if df is not None and df > 0:
        return [200.0, steer_toward(df, dr, 1.5)]

    lw, cw, rw = count_walls_in_front(obs, 12)

    if cw < 3:
        if lw > rw + 3:
            return [200.0, 8.0]
        elif rw > lw + 3:
            return [200.0, -8.0]
        return [200.0, 0.0]
    elif lw < rw:
        return [150.0, -30.0]
    elif rw < lw:
        return [150.0, 30.0]
    else:
        return [100.0, -30.0]

def strategy_wrestling(obs, obs_dict):
    my_id = obs_dict['id']
    energy = obs_dict.get('energy', 1000)

    if check_stuck(obs):
        return [-150.0, 30.0]

    opponent_idx = {IDX_GREEN, IDX_BLUE} if my_id == 'team_0' else {IDX_PURPLE, IDX_LIGHT_RED}
    df, dr = find_object_centroid(obs, opponent_idx)

    if df is not None:
        angle = steer_toward(df, dr, 2.5)
        force = 200.0 if energy > 300 else 100.0
        return [force, angle]

    # Search for opponent without spinning blindly
    lw, cw, rw = count_walls_in_front(obs, 10)
    if cw < 3:
        return [150.0, 0.0]
    elif lw < rw:
        return [100.0, -20.0]
    else:
        return [100.0, 20.0]

def strategy_ball_game(obs, obs_dict):
    my_id = obs_dict['id']
    energy = obs_dict.get('energy', 1000)
    global game_state

    if check_stuck(obs):
        return [-200.0, -30.0]

    self_colors = {IDX_PURPLE, IDX_LIGHT_RED} if my_id == 'team_0' else {IDX_GREEN, IDX_BLUE}
    ignore = self_colors | WALL_INDICES | {IDX_EMPTY, IDX_WHITE}

    target_idx = set()
    unique = np.unique(obs).astype(int)
    for v in unique:
        if v not in ignore:
            target_idx.add(v)

    if len(target_idx) == 0:
        target_idx = {IDX_GREEN, IDX_BLUE} if my_id == 'team_0' else {IDX_PURPLE, IDX_LIGHT_RED}

    df, dr = find_object_centroid(obs, target_idx)
    if df is not None:
        angle = steer_toward(df, dr, 2.5)
        force = 200.0 if energy > 200 else 100.0
        
        # Prevent own goals: if we are facing backward (theta > 90 or < -90), don't push the ball
        # Normalize theta to [-180, 180]
        theta = ((game_state.global_theta + 180) % 360) - 180
        if abs(theta) > 100:
            # We are facing our own goal. Move around the ball instead of pushing it.
            return [150.0, 30.0]
            
        return [force, angle]

    # Return to center/search
    lw, cw, rw = count_walls_in_front(obs, 10)
    theta = ((game_state.global_theta + 180) % 360) - 180
    
    # Try to face the opponent's goal (theta = 0)
    if abs(theta) > 20:
        angle = -30.0 if theta > 0 else 30.0
        return [100.0, angle]
        
    if cw < 3:
        return [150.0, 0.0]
    elif lw < rw:
        return [100.0, -20.0]
    else:
        return [100.0, 20.0]

def strategy_curling(obs, obs_dict):
    global game_state
    info = obs_dict.get('info', '')

    if isinstance(obs, np.ndarray) and np.all(obs <= 0):
        return [0.0, 0.0]

    # Detect new throw
    if isinstance(info, str) and 'Reset' in info:
        game_state.curling_launched = False
        game_state.curling_force_steps = 0
        game_state.curling_stone_count += 1
        return [0.0, 0.0]

    if not game_state.curling_launched:
        game_state.curling_launched = True
        game_state.curling_force_steps = 0

    game_state.curling_force_steps += 1

    # Apply precise force. 170 force for 7 steps gets dead center.
    # To prevent 5 stones hitting each other out, we slightly vary angle based on stone index.
    angles = [0.0, 2.0, -2.0, 4.0, -4.0]
    idx = game_state.curling_stone_count % 5
    ang = angles[idx]

    if game_state.curling_force_steps <= 7:
        return [170.0, ang]
    else:
        return [0.0, 0.0]

def strategy_billiard(obs, obs_dict):
    my_id = obs_dict['id']

    if check_stuck(obs):
        return [-100.0, 20.0]

    ignore = WALL_INDICES | {IDX_EMPTY, IDX_WHITE}
    target_idx = set()
    unique = np.unique(obs).astype(int)

    n = obs.shape[0]
    self_color = int(obs[int(n * 0.8), n // 2])

    for v in unique:
        if v not in ignore and v != self_color and v != 0:
            target_idx.add(v)

    if len(target_idx) > 0:
        df, dr = find_object_centroid(obs, target_idx)
        if df is not None:
            angle = steer_toward(df, dr, 2.5)
            return [70.0, angle]

    lw, cw, rw = count_walls_in_front(obs, 10)
    if cw < 3:
        return [60.0, 0.0]
    elif lw < rw:
        return [50.0, -20.0]
    else:
        return [50.0, 20.0]

# ===================== Game Type Detection =====================

def detect_game_type(obs, obs_dict):
    info = obs_dict.get('info', '')
    if isinstance(info, str) and 'Reset' in info:
        return 'curling'

    if not isinstance(obs, np.ndarray) or obs.size <= 1:
        return 'generic'

    if np.all(obs <= 0):
        return 'curling'

    n = obs.shape[0]
    total = n * n
    wall_count = np.sum(np.isin(obs, list(WALL_INDICES)))
    red_count = np.sum(obs == IDX_RED)
    wall_ratio = wall_count / total

    if red_count > 0 and wall_ratio > 0.08:
        return 'running'

    green_count = np.sum(obs == IDX_GREEN)
    if green_count > 10 and wall_ratio < 0.05:
        return 'wrestling'

    unique = np.unique(obs).astype(int)
    colored_types = set()
    for v in unique:
        if v not in WALL_INDICES and v != IDX_EMPTY and v != IDX_WHITE and v != 0:
            colored_types.add(v)

    if len(colored_types) >= 3 and wall_ratio < 0.1:
        return 'billiard'

    return 'ball_game'

# ===================== Main Controller =====================

def my_controller(observation, action_space, is_act_continuous=False):
    global game_state

    obs_data = observation.get('obs', observation) if isinstance(observation, dict) else observation

    if isinstance(obs_data, dict):
        agent_obs = obs_data.get('agent_obs', None)
        my_id = obs_data.get('id', 'team_0')
        game_mode = obs_data.get('game_mode', '')
        energy = obs_data.get('energy', 1000)
        info = obs_data.get('info', '')
    else:
        agent_obs = obs_data
        my_id = 'team_0'
        game_mode = ''
        energy = 1000
        info = ''

    obs_dict = {'id': my_id, 'game_mode': game_mode, 'energy': energy, 'info': info}

    if game_mode == 'NEW GAME':
        game_state.step_in_game = 0
        game_state.curling_launched = False
        game_state.curling_force_steps = 0
        game_state.curling_stone_count = 0
        game_state.stuck_counter = 0
        game_state.recovery_steps = 0
        game_state.global_theta = 0.0

        if isinstance(agent_obs, np.ndarray) and agent_obs.size > 1:
            game_state.game_type = detect_game_type(agent_obs, obs_dict)
        else:
            game_state.game_type = 'curling'

    game_state.step_in_game += 1

    if isinstance(info, str) and 'Reset' in info:
        game_state.game_type = 'curling'

    if isinstance(agent_obs, np.ndarray) and agent_obs.size > 1 and np.all(agent_obs <= 0):
        if game_state.game_type is None:
            game_state.game_type = 'curling'
        return [[0.0], [0.0]]

    obs = agent_obs if isinstance(agent_obs, np.ndarray) and agent_obs.size > 1 else np.zeros((40, 40))

    gt = game_state.game_type or 'ball_game'
    if gt == 'running':
        action = strategy_running(obs, obs_dict)
    elif gt == 'wrestling':
        action = strategy_wrestling(obs, obs_dict)
    elif gt == 'curling':
        action = strategy_curling(obs, obs_dict)
    elif gt == 'billiard':
        action = strategy_billiard(obs, obs_dict)
    else:
        action = strategy_ball_game(obs, obs_dict)

    force = max(-100.0, min(200.0, action[0]))
    angle = max(-30.0, min(30.0, action[1]))

    # Integrate global rotation
    game_state.global_theta += angle

    return [[force], [angle]]
