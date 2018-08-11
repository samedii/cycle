#import cupy as cp
import numpy as np

from functools import reduce
import operator

board_size = 4

def empty_boards(n_boards=2, N=board_size):
    return np.zeros((n_boards, N, N), dtype=np.int)

def random_boards(n_boards, N=board_size):
    return np.random.randint(0, 3, (n_boards, N, N), dtype=np.int)

def add_number(board):
    board = board.copy()
    board = np.reshape(board, (-1, board_size**2) )
    empty_pos = board == 0
    n_empty = np.sum(empty_pos,axis=(1), keepdims=True)
    pos = np.floor(np.random.random((board.shape[0],1))*n_empty).astype(np.int)

    e_sum = np.cumsum(empty_pos, axis=1)-1
    e_sum[~empty_pos] = -1
    mask = (e_sum == pos)
    board[mask] = 2
    board = np.reshape(board, (-1, board_size, board_size))
    return board

empty = 0
player = 1
goal = 2
wall = 3

def create_labyrinth1(n_boards):
    # 0 3 2 3
    # 0 0 0 0
    # 0 3 3 3
    # 0 1 0 0

    boards = empty_boards(n_boards=n_boards, N=4)
    boards[:, [0, 2]] = wall
    boards[:, [0, 2], 0] = empty
    boards[:, 3, 1] = player
    boards[:, 0, 3] = goal
    return boards

def move(boards, direction):
    boards = boards.copy()

    player_index = np.where(boards == 1)
    boards[player_index] = 0

    player_index = np.stack(player_index, axis=1)

    direction = direction[player_index[:, 0]]
    player_index[direction == 0, 1] -= 1 # up
    player_index[direction == 1, 2] += 1 # right
    player_index[direction == 2, 1] += 1 # down
    player_index[direction == 3, 2] -= 1 # left

    moved_outside = np.logical_or(
        player_index[:, [1,2]] >= 4,
        player_index[:, [1,2]] <= -1
    ).any(axis=1)

    player_index = player_index[~moved_outside]

    if player_index.shape[0] >= 1:
        boards[tuple(player_index.T)] = player

    return boards

def status(boards, previous_boards):
    status_value = np.zeros(boards.shape[0])
    win = (boards != goal).all(axis=(1,2)) # walked onto goal
    loss = np.logical_or.reduce((
        (boards == previous_boards).all(axis=(1,2)), # no change
        (boards != player).all(axis=(1,2)), # no player
        (boards == wall).sum(axis=(1,2))  !=  (previous_boards == wall).sum(axis=(1,2)), # walked into wall
    ))
    status_value[loss] = -1
    status_value[win] = 1
    is_game_over = np.logical_or(win, loss)
    return status_value, is_game_over

class Games:
    def __init__(self, n_boards):
        self.boards = create_labyrinth1(n_boards)
        self.n_boards = n_boards
        self.previous_boards = None

    def step(self, actions):
        self.previous_boards = self.boards


        self.boards = move(self.boards, actions)

        if self.previous_boards is None:
            return self.boards, np.zeros(self.n_boards), np.zeros(self.n_boards, dtype=np.bool)
        
        status_value, is_game_over = status(self.boards, self.previous_boards)

        return self.boards, status_value, is_game_over

def main():

    # previous_boards = create_labyrinth1(2)
    # print(previous_boards)
    # boards = move(previous_boards, np.array([0,1]))
    # print(boards)
    # status_value, game_over = status(boards, previous_boards)
    # print(status_value)
    # print(game_over)

    import time

    start = time.time()
    n_boards = 1
    game = Games(n_boards)
    board = game.boards

    board, reward, is_game_over = game.step(np.array([0]))
    print(board[0])
    board, reward, is_game_over = game.step(np.array([0]))
    print(board[0])
    board, reward, is_game_over = game.step(np.array([0]))
    print(board[0])
    board, reward, is_game_over = game.step(np.array([0]))
    print(board[0])

    print(time.time() - start)

#main()