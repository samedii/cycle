
import numpy as np
import cupy as cp
#np.asnumpy = lambda self: self
#cp = np
import matplotlib.pyplot as plt
from functools import reduce

import game_2048

def apply_organism(organism, x):

  x = x.reshape(x.shape[0], 1, -1)

  for layer_index, (layer_weight, layer_bias) in enumerate(organism):
    if layer_index != 0:
        x = cp.maximum(0, x)
        
    x = cp.matmul(x, layer_weight)
    x += layer_bias

  return x
    
def init_layer(n_organisms, n_rows, n_cols):
    return (
        cp.random.normal(loc=0, scale=0.001, size=(n_organisms, n_rows, n_cols)),
        cp.random.normal(loc=0, scale=0.001, size=(n_organisms, 1, n_cols))
    )

def init_organism(n_organisms, size):
    size = np.array(size)
    from_size = size[:-1]
    to_size = size[1:]
    return [init_layer(n_organisms, from_size[i], to_size[i]) for i in range(len(size)-1)]

def mutate(organism, n_children, mutation_scale=0.01):
    return [
        (
            cp.random.normal(loc=layer_weight, scale=mutation_scale, size=layer_weight.shape),
            cp.random.normal(loc=layer_bias, scale=mutation_scale, size=layer_bias.shape)
        )
        for (layer_weight, layer_bias) in organism
    ]

def get_loss(y, y_pred):
    return -y_pred*cp.sin(y_pred)

def get_organism(organism, index):
    return [(layer_weight[index].copy(), layer_bias[index].copy()) for (layer_weight, layer_bias) in organism]

def concatenate_organism(organism_a, organism_b):
    return [
        (
            cp.concatenate((organism_a[i][0], organism_b[i][0]), axis=0), # layer_weight
            cp.concatenate((organism_a[i][1], organism_b[i][1]), axis=0) # layer_bias
        )
        for i in range(len(organism_a))
    ]

def flatten_organism(organism):
    size = (organism[0][0].shape[0], -1)
    flat_layers = [[layer_weight.reshape(size), layer_bias.reshape(size)] for layer_weight, layer_bias in organism]
    all_layers = reduce(lambda x,y: x+y, flat_layers)
    return cp.concatenate(all_layers, axis=1)

def reform_organism(flat_organism, size):
    size = np.array(list(size))
    from_size = size[:-1]
    to_size = size[1:]

    nn = []
    start = 0
    for i in range(len(size)-1):
        layer_weight_size = [from_size[i], to_size[i]]
        layer_bias_size = [to_size[i]]

        stop = start + np.prod(layer_weight_size)
        layer_weight = flat_organism[:, start:stop].reshape([-1]+layer_weight_size)
        start = stop
        stop = start + np.prod(layer_bias_size)
        layer_bias = flat_organism[:, start:stop].reshape([-1, 1]+layer_bias_size)
        start = stop

        nn.append((layer_weight, layer_bias))
    return nn

def train():

    n_iterations = 1000
    n_moves = 1000

    learning_rate = 0.01
    mutation_scale = 0.01
    size = [2*4**2 + 12*2, 40, 4]

    n_organisms = 1000

    plt.ion()
    plt.show()

    best_loss_iteration = []
    mean_loss_iteration = []
    worst_loss_iteration = []

    initial_organism = init_organism(1, size)
    flat_initial_organism = flatten_organism(initial_organism)
    loc = flatten_organism(initial_organism)
    previous_loc = loc
    N = cp.random.normal(loc=0, scale=mutation_scale, size=(int(n_organisms/2), flat_initial_organism.shape[1]))
    flat_organism = cp.concatenate((loc + N, loc - N), axis=0)
    organism = reform_organism(flat_organism, size)

    for iteration in range(1, n_iterations+1):

        game = game_2048.Games(n_boards=n_organisms)
        board = game.boards

        is_game_over = np.repeat(False, n_organisms)
        action = np.zeros((n_organisms,))
        for move in range(n_moves):
            board = cp.asarray(board)
            is_empty = (board == 0).astype(cp.float)
            vertical = (board[:, :-1] == board[:, 1:])
            horisontal = (board[:, :, :-1] == board[:, :, 1:])
            board[board == 0] = 1
            board = cp.log2(board)
            data = cp.concatenate((
                cp.log2(board).reshape((-1, 16)),
                is_empty.reshape((-1, 16)),
                vertical.reshape((-1, 12)),
                horisontal.reshape((-1, 12)),
            ), axis=1)
            output = apply_organism(get_organism(organism, ~is_game_over), data[~is_game_over])
            new_action = cp.reshape(output, (-1, 4))
            new_action = cp.argmax(new_action, axis=1)
            # new_action = cp.exp(new_action)
            # new_action = new_action/new_action.sum(axis=1, keepdims=True)
            # p = cp.random.rand(new_action.shape[0], 1)
            # c = cp.cumsum(new_action, axis=1)
            # new_action = (p <= c).sum(axis=1)
            new_action = cp.asnumpy(new_action)
            action[~is_game_over] = new_action
            board, reward, is_game_over = game.step(action)
           
            if is_game_over.all():
                break

        loss = -cp.asarray(reward)
        organism_loss = loss
        best_loss_iteration.append(cp.asnumpy(organism_loss.min()))
        mean_loss_iteration.append(cp.asnumpy(organism_loss.mean()))
        worst_loss_iteration.append(cp.asnumpy(organism_loss.max()))

        print('iteration: {}, loss: {}, std: {:.2f}, gradient norm: {:.4f}'.format(
            iteration,
            organism_loss.min(),
            cp.asnumpy(organism_loss).std(),
            cp.asnumpy(cp.linalg.norm(loc - previous_loc))
        ))

        #print('best final board')
        #print(board[best_organism_index])
        if iteration % 1 == 0:
            best_organism_index = cp.argmin(organism_loss)

            plt.clf()
            # plt.subplot(2, 2, 3)
            # #plt.yscale('log')
            # plt.plot(best_board_iteration)
            # plt.plot(mean__iteration)
            # plt.plot(worst_loss_iteration)

            #plt.subplot(2, 2, 3)
            #plt.yscale('log')
            plt.plot(best_loss_iteration)
            plt.plot(mean_loss_iteration)
            plt.plot(worst_loss_iteration)

            plt.draw()
            plt.pause(0.001)

        best_organism_index = cp.argmin(loss)
        loc = flatten_organism(organism)[[best_organism_index]]

        previous_loc = loc
        organism, loc, N = evolve(organism, organism_loss, mutation_scale, size, learning_rate)


def evolve(organism, organism_loss, scale, size, learning_rate=0.001):
    flat_organism = flatten_organism(organism)
    loc = flat_organism.mean(axis=0, keepdims=True)
    n_organisms = flat_organism.shape[0]
    N = flat_organism - loc
    A = (organism_loss - cp.mean(organism_loss))/(1e-6 + cp.std(organism_loss))
    loc = loc - learning_rate/(n_organisms*scale)*cp.dot(N.T, A)
    N = cp.random.normal(loc=0, scale=scale, size=(int(n_organisms/2), flat_organism.shape[1]))
    new_flat_organism = cp.concatenate((loc + N, loc - N), axis=0)
    return reform_organism(new_flat_organism, size), loc, N

def main():
    train()

if __name__ == '__main__':
    main()
