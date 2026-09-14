# main.py

import pygame
import numpy as np
import random
import sys
import os

#
# 1. CONFIGURATION
#
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Q-Learning
ALPHA = 0.8
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

EPISODES = 2000

# Episode ที่ต้องการแสดง Animation
SHOW_EPISODES = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# เวลาหน่วงการเดิน (วินาที)
STEP_DELAY = 0.25

# เวลาหน่วงหลังจบ Episode
EPISODE_DELAY = 1.0

# จำกัดจำนวนก้าวต่อ Episode
MAX_STEPS = 100

# Q-Table Persistence
Q_TABLE_PATH = "q_table.npy"

# Window
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
FPS = 60

#
# 2. GRID DEFINITION
#
grid = ["SFFF", "FHFH", "FFFH", "HFFG"]

N_ROWS = len(grid)
N_COLS = len(grid[0])
N_STATES = N_ROWS * N_COLS
N_ACTIONS = 4

# Action
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# ทิศทางของลูกศร (dx, dy) สำหรับ policy arrows
ACTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

#
# 3. COLORS
#
BACKGROUND = (25, 30, 40)
PANEL_COLOR = (35, 42, 55)
GRID_LINE = (60, 70, 85)
FLOOR_COLOR = (70, 120, 80)
START_COLOR = (70, 130, 200)
HOLE_COLOR = (25, 25, 30)
GOAL_COLOR = (230, 190, 60)
AGENT_COLOR = (80, 220, 255)
TEXT_COLOR = (240, 240, 240)
RED = (230, 80, 80)
GREEN = (80, 220, 120)
YELLOW = (240, 220, 80)
GRAY = (160, 160, 160)
GRAPH_BG = (20, 25, 35)
ARROW_COLOR = (255, 255, 255)


def state_to_rc(state):
    """แปลง State เป็น Row และ Column"""
    row = state // N_COLS
    col = state % N_COLS
    return row, col


def rc_to_state(row, col):
    """แปลง Row และ Column เป็น State"""
    return row * N_COLS + col


def get_cell(row, col):
    """อ่านค่าของช่องใน Grid"""
    return grid[row][col]


def get_start_state():
    """ค้นหาตำแหน่ง Start"""
    for row in range(N_ROWS):
        for col in range(N_COLS):
            if grid[row][col] == "S":
                return rc_to_state(row, col)
    return 0


def step(state, action):
    """Environment Step"""
    row, col = state_to_rc(state)

    if action == UP:
        row = max(row - 1, 0)
    elif action == DOWN:
        row = min(row + 1, N_ROWS - 1)
    elif action == LEFT:
        col = max(col - 1, 0)
    elif action == RIGHT:
        col = min(col + 1, N_COLS - 1)

    new_state = rc_to_state(row, col)
    cell = get_cell(row, col)

    if cell == "H":
        return new_state, -1.0, True, "HOLE"
    elif cell == "G":
        return new_state, 1.0, True, "GOAL"
    else:
        return new_state, -0.01, False, "MOVE"


def choose_action(Q, state, epsilon):
    """Epsilon-Greedy"""
    random_value = random.uniform(0, 1)
    if random_value < epsilon:
        action = random.randint(0, N_ACTIONS - 1)
        mode = "EXPLORE"
    else:
        action = int(np.argmax(Q[state]))
        mode = "EXPLOIT"
    return action, mode


def update_q_table(Q, state, action, reward, next_state, done):
    """Q-Learning Update
    Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
    """
    current_q = Q[state, action]

    if done:
        target = reward
    else:
        best_next_q = np.max(Q[next_state])
        target = reward + GAMMA * best_next_q

    Q[state, action] = current_q + ALPHA * (target - current_q)


def save_q_table(Q, path=Q_TABLE_PATH):
    """บันทึก Q-table ลงไฟล์"""
    np.save(path, Q)
    print(f" [Saved] Q-table -> {path}")


def load_q_table(path=Q_TABLE_PATH):
    """โหลด Q-table จากไฟล์ ถ้ามี ไม่งั้นคืน None"""
    if os.path.exists(path):
        Q = np.load(path)
        print(f"[Loaded] Q-table <- {path}")
        return Q
    return None


pygame.init()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Q-Learning Grid World - Learning Visualization")
clock = pygame.time.Clock()

font_large = pygame.font.SysFont("Arial", 32, bold=True)
font_medium = pygame.font.SysFont("Arial", 22, bold=True)
font_small = pygame.font.SysFont("Arial", 18)
font_tiny = pygame.font.SysFont("Arial", 14)

CELL_SIZE = 120

GRID_X = 50
GRID_Y = 120

GRID_WIDTH = N_COLS * CELL_SIZE
GRID_HEIGHT = N_ROWS * CELL_SIZE

PANEL_X = 570
PANEL_Y = 80

PANEL_WIDTH = 480
PANEL_HEIGHT = 500


def draw_text(text, font, color, x, y):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))


def draw_center_text(text, font, color, rect):
    surface = font.render(text, True, color)
    text_rect = surface.get_rect(center=rect.center)
    screen.blit(surface, text_rect)


def draw_policy_arrow(rect, action):
    """วาดลูกศรบอกทิศทางที่ policy เลือกในช่องนี้"""
    dx, dy = ACTION_VECTORS[action]
    center_x, center_y = rect.center

    length = CELL_SIZE * 0.28

    tip = (center_x + dx * length, center_y + dy * length)
    base_left = (
        center_x - dx * length * 0.5 + (-dy) * length * 0.5,
        center_y - dy * length * 0.5 + dx * length * 0.5,
    )
    base_right = (
        center_x - dx * length * 0.5 - (-dy) * length * 0.5,
        center_y - dy * length * 0.5 - dx * length * 0.5,
    )

    pygame.draw.polygon(screen, ARROW_COLOR, [tip, base_left, base_right])


def draw_grid(agent_state, Q=None, show_policy=False):
    max_q_value = None
    min_q_value = None

    if show_policy and Q is not None:
        best_q_per_state = np.max(Q, axis=1)
        max_q_value = float(np.max(best_q_per_state))
        min_q_value = float(np.min(best_q_per_state))

    for row in range(N_ROWS):
        for col in range(N_COLS):
            x = GRID_X + col * CELL_SIZE
            y = GRID_Y + row * CELL_SIZE
            cell = grid[row][col]
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            if cell == "S":
                color = START_COLOR
            elif cell == "F":
                color = FLOOR_COLOR
            elif cell == "H":
                color = HOLE_COLOR
            elif cell == "G":
                color = GOAL_COLOR

            # ----- Q-value heatmap (เฉพาะช่อง Floor/Start) -----
            if show_policy and Q is not None and cell in ("S", "F"):
                state = rc_to_state(row, col)
                q_val = float(np.max(Q[state]))
                if max_q_value != min_q_value:
                    normalized = (q_val - min_q_value) / (max_q_value - min_q_value)
                else:
                    normalized = 0.5
                normalized = max(0.0, min(1.0, normalized))

                # ผสมสีพื้นเดิมกับ เขียว ตามค่า Q
                heat_r = int(FLOOR_COLOR[0] * (1 - normalized) + 40 * normalized)
                heat_g = int(FLOOR_COLOR[1] * (1 - normalized) + 200 * normalized)
                heat_b = int(FLOOR_COLOR[2] * (1 - normalized) + 80 * normalized)
                color = (heat_r, heat_g, heat_b)

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 3)

            if cell == "S":
                draw_center_text("START", font_small, TEXT_COLOR, rect)
            elif cell == "H":
                center_x = x + CELL_SIZE // 2
                center_y = y + CELL_SIZE // 2
                pygame.draw.circle(screen, (5, 5, 5), (center_x, center_y), 35)
                pygame.draw.circle(screen, (70, 70, 80), (center_x, center_y), 35, 3)
                draw_center_text("HOLE", font_tiny, GRAY, rect)
            elif cell == "G":
                draw_center_text("GOAL", font_small, (30, 30, 30), rect)

            # Policy Arrows (เฉพาะช่อง Floor/Start)
            if show_policy and Q is not None and cell in ("S", "F"):
                state = rc_to_state(row, col)
                best_action = int(np.argmax(Q[state]))
                draw_policy_arrow(rect, best_action)

            # ----- Agent -----
            current_state = rc_to_state(row, col)
            if current_state == agent_state:
                center_x = x + CELL_SIZE // 2
                center_y = y + CELL_SIZE // 2

                pygame.draw.circle(screen, AGENT_COLOR, (center_x, center_y), 32)
                pygame.draw.circle(
                    screen, (255, 255, 255), (center_x - 10, center_y - 5), 7
                )
                pygame.draw.circle(
                    screen, (255, 255, 255), (center_x + 10, center_y - 5), 7
                )
                pygame.draw.circle(
                    screen, (20, 20, 20), (center_x - 10, center_y - 5), 3
                )
                pygame.draw.circle(
                    screen, (20, 20, 20), (center_x + 10, center_y - 5), 3
                )

                draw_center_text(
                    "AI", font_tiny, (20, 20, 20), pygame.Rect(x, y + 55, CELL_SIZE, 40)
                )


def draw_info_panel(
    episode,
    epsilon,
    step_count,
    total_score,
    last_reward,
    last_action,
    mode,
    result,
    success_count,
    death_count,
    timeout_count,
    episode_scores,
):
    panel_rect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, PANEL_COLOR, panel_rect, border_radius=15)

    draw_text("Q-LEARNING STATUS", font_large, TEXT_COLOR, PANEL_X + 25, PANEL_Y + 20)

    y = PANEL_Y + 80
    draw_text(
        f"Episode: {episode} / {EPISODES}", font_medium, TEXT_COLOR, PANEL_X + 25, y
    )
    y += 38

    draw_text(f"Step: {step_count}", font_medium, TEXT_COLOR, PANEL_X + 25, y)
    y += 38

    score_color = RED if total_score < 0 else GREEN
    draw_text(f"Score: {total_score:.3f}", font_medium, score_color, PANEL_X + 25, y)
    y += 38

    draw_text(
        f"Last Reward: {last_reward:.3f}", font_medium, TEXT_COLOR, PANEL_X + 25, y
    )
    y += 38

    draw_text(f"Epsilon: {epsilon:.3f}", font_medium, YELLOW, PANEL_X + 25, y)
    y += 38

    draw_text(f"Action: {last_action}", font_medium, TEXT_COLOR, PANEL_X + 25, y)
    y += 38

    mode_color = GREEN if mode == "EXPLOIT" else YELLOW
    draw_text(f"Mode: {mode}", font_medium, mode_color, PANEL_X + 25, y)
    y += 38

    result_color = TEXT_COLOR
    if result == "GOAL":
        result_color = GREEN
    elif result == "HOLE":
        result_color = RED

    draw_text(f"Result: {result}", font_medium, result_color, PANEL_X + 25, y)
    y += 38

    stat_y = y + 15

    # FIX: รวม timeout เข้าไปในตัวหารของ success rate ด้วย
    # ไม่งั้น success_rate จะดูสูงเกินจริงถ้ามี episode ที่ timeout เยอะ
    total_finished = success_count + death_count + timeout_count

    if total_finished > 0:
        success_rate = success_count / total_finished * 100
    else:
        success_rate = 0

    average_score = np.mean(episode_scores) if len(episode_scores) > 0 else 0

    draw_text(f"GOAL: {success_count}", font_small, GREEN, PANEL_X + 25, stat_y)
    draw_text(f"HOLE: {death_count}", font_small, RED, PANEL_X + 160, stat_y)
    draw_text(f"TIMEOUT: {timeout_count}", font_small, GRAY, PANEL_X + 300, stat_y)

    draw_text(
        f"Success Rate: {success_rate:.1f}%",
        font_small,
        TEXT_COLOR,
        PANEL_X + 25,
        stat_y + 35,
    )
    draw_text(
        f"Average Score: {average_score:.3f}",
        font_small,
        TEXT_COLOR,
        PANEL_X + 25,
        stat_y + 70,
    )


def draw_score_graph(scores):
    graph_x = 50
    graph_y = 620
    graph_width = 1000
    graph_height = 60

    graph_rect = pygame.Rect(graph_x, graph_y, graph_width, graph_height)
    pygame.draw.rect(screen, GRAPH_BG, graph_rect)
    pygame.draw.rect(screen, GRID_LINE, graph_rect, 2)

    draw_text("Score History", font_tiny, TEXT_COLOR, graph_x, graph_y - 20)

    if len(scores) < 2:
        return

    display_scores = scores[-100:]
    minimum = min(display_scores)
    maximum = max(display_scores)

    if maximum == minimum:
        maximum += 1
        minimum -= 1

    points = []
    for i, score in enumerate(display_scores):
        x = graph_x + i / (len(display_scores) - 1) * graph_width
        normalized = (score - minimum) / (maximum - minimum)
        y = graph_y + graph_height - normalized * graph_height
        points.append((int(x), int(y)))

    if len(points) > 1:
        pygame.draw.lines(screen, GREEN, False, points, 2)


def draw_screen(
    agent_state,
    episode,
    epsilon,
    step_count,
    total_score,
    last_reward,
    last_action,
    mode,
    result,
    success_count,
    death_count,
    timeout_count,
    episode_scores,
):
    screen.fill(BACKGROUND)

    draw_text("Q-LEARNING GRID WORLD", font_large, TEXT_COLOR, 50, 40)
    draw_text("Watch the AI learn through Trial and Error", font_small, GRAY, 50, 80)

    draw_grid(agent_state)

    draw_info_panel(
        episode,
        epsilon,
        step_count,
        total_score,
        last_reward,
        last_action,
        mode,
        result,
        success_count,
        death_count,
        timeout_count,
        episode_scores,
    )

    draw_score_graph(episode_scores)

    pygame.display.flip()


paused = False


def handle_events():
    global paused

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused


def wait_while_paused():
    """ค้างไว้ระหว่าง pause จนกว่าจะกด SPACE อีกครั้ง"""
    while paused:
        handle_events()
        clock.tick(FPS)


def wait_with_events(seconds):
    start_time = pygame.time.get_ticks()

    while True:
        handle_events()
        wait_while_paused()
        current_time = pygame.time.get_ticks()

        elapsed = (current_time - start_time) / 1000
        if elapsed >= seconds:
            break
        clock.tick(FPS)


def train():
    # โหลด Q-table เดิมถ้ามี ไม่งั้นเริ่มจาก 0
    Q = load_q_table()

    if Q is None:
        Q = np.zeros((N_STATES, N_ACTIONS))

    epsilon = EPSILON_START

    episode_scores = []

    success_count = 0
    death_count = 0
    timeout_count = 0

    start_state = get_start_state()

    for episode in range(1, EPISODES + 1):
        show_animation = episode in SHOW_EPISODES

        state = start_state
        total_score = 0
        step_count = 0
        done = False

        last_reward = 0
        last_action = "-"
        mode = "-"
        result = "RUNNING"

        if show_animation:
            draw_screen(
                state,
                episode,
                epsilon,
                step_count,
                total_score,
                last_reward,
                last_action,
                mode,
                result,
                success_count,
                death_count,
                timeout_count,
                episode_scores,
            )
            wait_with_events(0.7)

        while not done and step_count < MAX_STEPS:
            handle_events()
            wait_while_paused()

            action, mode = choose_action(Q, state, epsilon)
            last_action = ACTION_NAMES[action]

            next_state, reward, done, event_result = step(state, action)

            update_q_table(Q, state, action, reward, next_state, done)

            total_score += reward
            step_count += 1
            last_reward = reward
            state = next_state

            if done:
                result = event_result

            if show_animation:
                draw_screen(
                    state,
                    episode,
                    epsilon,
                    step_count,
                    total_score,
                    last_reward,
                    last_action,
                    mode,
                    result,
                    success_count,
                    death_count,
                    timeout_count,
                    episode_scores,
                )
                wait_with_events(STEP_DELAY)

        if result == "GOAL":
            success_count += 1
        elif result == "HOLE":
            death_count += 1
        else:
            result = "TIMEOUT"
            timeout_count += 1

        episode_scores.append(total_score)

        if show_animation:
            draw_screen(
                state,
                episode,
                epsilon,
                step_count,
                total_score,
                last_reward,
                last_action,
                mode,
                result,
                success_count,
                death_count,
                timeout_count,
                episode_scores,
            )
            wait_with_events(EPISODE_DELAY)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    save_q_table(Q)

    return Q, episode_scores


def show_final_result(Q, episode_scores):
    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nQ-TABLE:")
    print(np.round(Q, 3))

    print("\nEpisode Scores:")
    for i, score in enumerate(episode_scores, start=1):
        print(f"Episode {i:4d} : {score:.3f}")

    running = True

    while running:
        handle_events()

        screen.fill(BACKGROUND)

        draw_text("TRAINING COMPLETE!", font_large, GREEN, 50, 50)
        draw_text(f"Total Episodes: {EPISODES}", font_medium, TEXT_COLOR, 50, 110)
        draw_text(
            f"Final Average Score: {np.mean(episode_scores):.3f}",
            font_medium,
            TEXT_COLOR,
            50,
            155,
        )
        draw_text(
            "Arrows show the learned policy | Close window to exit",
            font_small,
            GRAY,
            50,
            210,
        )

        # Grid พร้อมลูกศร policy + heatmap ของ Q-value
        draw_grid(agent_state=-1, Q=Q, show_policy=True)

        draw_score_graph(episode_scores)

        pygame.display.flip()
        clock.tick(FPS)


#
# 16. RUN PROGRAM
#
if __name__ == "__main__":
    Q, episode_scores = train()
    show_final_result(Q, episode_scores)