# test_sudoku.py

import random
from collections import defaultdict

# ขนาดกระดาน 4x4
N = 4
BOX = 2 # ขนาดบล็อกย่อย 2x2
EMPTY = 0

class SudokuEnv:
    """Environment สำหรับซูโดกุ 4x4 ในรูปแบบ RL"""

    def __init__(self, puzzle):
        self.start = tuple(puzzle) # กระดานตั้งต้น (16 ช่อง)
        self.board = list(self.start)

    def reset(self):
        self.board = list(self.start)
        return tuple(self.board)

    def empty_cells(self):
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def valid_actions(self):
        """คืนค่า action ที่เป็นไปได้ทั้งหมด: (index, value)"""
        actions = []
        for idx in self.empty_cells():
            for val in range(1, N + 1):
                actions.append((idx, val))
        return actions

    def is_valid_move(self, idx, val):
        row, col = idx // N, idx % N
        # เช็คแถว
        for c in range(N):
            if self.board[row * N + c] == val:
                return False

        # เช็คคอลัมน์
        for r in range(N):
            if self.board[r * N + col] == val:
                return False

        # เช็คบล็อก 2x2
        br, bc = (row // BOX) * BOX, (col // BOX) * BOX
        for r in range(br, br + BOX):
            for c in range(bc, bc + BOX):
                if self.board[r * N + c] == val:
                    return False

        return True

    def step(self, action):
        idx, val = action
        if self.board[idx] != EMPTY or not self.is_valid_move(idx, val):
            return tuple(self.board), -1, True # ผิดกฎ -> จบเกม

        self.board[idx] = val
        if EMPTY not in self.board:
            return tuple(self.board), 20, True # แก้สำเร็จ

        return tuple(self.board), 1, False # เดินถูกกฎ ไปต่อ


class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.3):
        self.q = defaultdict(float) # Q[(state, action)] = ค่า Q
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # อัตราการสำรวจแบบสุ่ม (exploration)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions) # สำรวจแบบสุ่ม

        # เลือก action ที่ Q สูงสุด (ใช้ประโยชน์จากความรู้เดิม)
        q_values = [self.q[(state, a)] for a in actions]
        max_q = max(q_values)
        best = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_actions, done):
        current_q = self.q[(state, action)]
        if done or not next_actions:
            target = reward
        else:
            target = reward + self.gamma * max(
                self.q[(next_state, a)] for a in next_actions
            )
        self.q[(state, action)] += self.alpha * (target - current_q)


def print_board(board, title=None):
    """ช่วยพิมพ์กระดาน 4x4 ให้อ่านง่าย"""
    if title:
        print(title)
    for r in range(N):
        row = board[r * N : (r + 1) * N]
        print(" " + " ".join(str(v) if v != EMPTY else "." for v in row))


def train(env, agent, episodes=20000):
    solved_count = 0

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # สร้างรายการ Action ที่สามารถเลือกได้
            actions = [(i, v) for i in env.empty_cells() for v in range(1, N + 1)]
            if not actions:
                break

            # Agent เลือก Action
            action = agent.choose_action(state, actions)

            # ทำ action
            next_state, reward, done = env.step(action)

            # สร้าง Action ของ State ถัดไป
            next_actions = [(i, v) for i in env.empty_cells() for v in range(1, N + 1)]

            # Update Q-value
            agent.update(state, action, reward, next_state, next_actions, done)

            state = next_state

        # ตรวจสอบว่าแก้ Sudoku สำเร็จหรือไม่
        if reward == 20:
            solved_count += 1

        # ลดค่า Epsilon
        # จาก Exploration -> Exploitation
        agent.epsilon = max(0.01, agent.epsilon * 0.9995)

        # แสดงความคืบหน้าเป็นระยะ ๆ ระหว่างฝึก
        if (ep + 1) % 5000 == 0:
            print(
                f" [ฝึกแล้ว {ep + 1}/{episodes} รอบ] "
                f"แก้สำเร็จสะสม {solved_count} ครั้ง, epsilon={agent.epsilon:.3f}"
            )

    print(f"แก้สำเร็จ {solved_count} จาก {episodes} รอบการฝึก")


def solve_with_policy(env, agent, verbose=True):
    """ใช้ policy ที่ฝึกแล้ว (ไม่มีการสุ่ม) เพื่อลองแก้กระดานจริง
    verbose=True จะ print แสดงกระดานเริ่มต้นและทุกสเต็ปที่ agentตัดสินใจ
    """
    state = env.reset()
    agent.epsilon = 0 # ปิดการสำรวจแบบสุ่ม ใช้ความรู้ล้วนๆ
    done = False
    steps = 0
    reward = 0

    if verbose:
        print("\n" + "=" * 40)
        print("เริ่มแก้ซูโดกุด้วย policy ที่ฝึกแล้ว")
        print("=" * 40)
        print_board(state, "กระดานเริ่มต้น:")
        print("-" * 40)

    while not done and steps < 20:
        actions = [(i, v) for i in env.empty_cells() for v in range(1, N + 1)]
        if not actions:
            break

        action = agent.choose_action(state, actions)
        idx, val = action
        row, col = idx // N, idx % N

        state, reward, done = env.step(action)
        steps += 1

        if verbose:
            status = (
                "ถูกกฎ"
                if reward == 1
                else " กระดานสมบูรณ์!" if reward == 20 else "X ผิดกฎ (จบเกม)"
            )
            print(
                f"สเต็ปที่ {steps}: เดิมค่า {val} ที่ตำแหน่ง (แถว {row}, คอลัมน์ {col}) "
                f"-> reward={reward} [{status}]"
            )
            print_board(state)
            print("-" * 40)

    if verbose:
        print("=" * 40)
        if reward == 20:
            print(f" แก้สำเร็จภายใน {steps} สเต็ป!")
        else:
            print(
                f" ยังไม่สำเร็จหลังจาก {steps} สเต็ป (อาจคิดค่าที่ผิดกฎ หรือฝึกไม่พอ)"
            )
        print("=" * 40 + "\n")

    return env.board, reward == 20


if __name__ == "__main__":
    puzzle = [
        1, 0, 0, 4,
        4, 1, 0, 0,
        0, 1, 4, 0,
        4, 0, 0, 1
    ]

    env = SudokuEnv(puzzle)
    agent = QLearningAgent()

    print("เริ่มฝึก agent ด้วย Q-learning...")
    train(env, agent, episodes=20000)

    board, solved = solve_with_policy(env, agent, verbose=True)
    print("ผลลัพธ์สุดท้าย:")
    for r in range(N):
        print(board[r * N : (r + 1) * N])
    print("แก้สำเร็จ!" if solved else "ยังแก้ไม่สำเร็จ ลองเพิ่มจำนวนรอบฝึก")