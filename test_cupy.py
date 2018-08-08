
import numpy as np
import cupy as cp
#np.asnumpy = lambda self: self
#cp = np
import game_2048
import pickle
import argparse

parser = argparse.ArgumentParser(description='Cupy evolution')
parser.add_argument('--no_graphics', default=False, type=bool, nargs='?', const=True, help='no graphics')
args = parser.parse_args()

if not args.no_graphics:
    import matplotlib.pyplot as plt

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

def generate_data():
    n_obs = 100
    X = cp.random.normal(loc=0, scale=1, size=(n_obs, 1))
    y = cp.random.normal(loc=cp.sin(2*X), scale=0.1, size=X.shape)
    return X, y

#def get_loss(y, y_pred):
#    return cp.ceil(10*cp.square(y.reshape(1, y.shape[0], 1, y.shape[1]) - y_pred))

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

def distribute_food(loss, amount):
    organism_loss = cp.asnumpy(loss)
    sort = np.argsort(organism_loss)

    n_organisms = len(sort)
    ranked_food = np.arange(start=n_organisms, stop=0, step=-1)
    ranked_food = ranked_food/ranked_food.sum()*amount
    food = np.zeros_like(ranked_food)
    food[sort] = ranked_food
    return food

def train():

    n_iterations = 1000000
    n_moves = 10000

    mutation_scale = 0.1
    n_children = 1
    size = [16 + 2, 10, 4]
    food_amount = 2000
    n_initial_organisms = int(food_amount/10)

    if not args.no_graphics:
        plt.ion()
        plt.show()

    best_loss_iteration = []
    mean_loss_iteration = []
    worst_loss_iteration = []
    best_food_loss_iteration = []
    best_food_organism_iteration = []

    organism = init_organism(n_initial_organisms, size)
    food = np.ones(n_initial_organisms)
    for iteration in range(1, n_iterations+1):
        n_organisms = len(food)
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
                #cp.log2(board).reshape((-1, 16)),
                is_empty.reshape((-1, 16)),
                #vertical.reshape((-1, 12)),
                #horisontal.reshape((-1, 12)),
                vertical.sum(axis=(1,2)).reshape((-1, 1)),
                horisontal.sum(axis=(1,2)).reshape((-1, 1))
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
        
        food += distribute_food(loss, amount=food_amount - food.sum())
        food -= 1

        best_loss_iteration.append(cp.asnumpy(organism_loss.min()))
        mean_loss_iteration.append(cp.asnumpy(organism_loss.mean()))
        worst_loss_iteration.append(cp.asnumpy(organism_loss.max()))
        best_food_loss_iteration.append(cp.asnumpy(organism_loss[np.argsort(food)[-10:]].mean()))
        best_food_organism_iteration.append(get_organism(organism, np.argmax(food)))

        is_living = (food >= 0)
        survivor = get_organism(organism, is_living)
        food = food[is_living].copy()

        is_new_parent = (food >= 1)
        if is_new_parent.sum() >= 1:
            parent = get_organism(survivor, is_new_parent)
            children = mutate(parent, n_children, mutation_scale=mutation_scale)
            organism = concatenate_organism(survivor, children)
            n_new_children = int(is_new_parent.sum()*n_children)
            food[is_new_parent] -= 1
            food = np.concatenate((food, np.ones(n_new_children)))
        else:
            organism = survivor

        print('iteration: {}, loss: {}, std: {:.2f}, n_organisms: {}, food: {:.2f}, std: {:.2f}, dead: {:.2f}, children: {}'.format(
            iteration,
            organism_loss.min(),
            cp.asnumpy(organism_loss).std(),
            loss.shape[0],
            food.mean(),
            food.std(),
            (~is_living).sum()/loss.shape[0],
            is_new_parent.sum()*n_children
        ))

        best_organism_index = np.argmin(cp.asnumpy(organism_loss))
        print('best final board')
        print(board[best_organism_index])

        if iteration % 1 == 0 and not args.no_graphics:
            best_organism_index = cp.argmin(organism_loss)

            plt.clf()
            plt.subplot(2, 1, 1)
            plt.title('Loss (neg. sum of board)')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.plot(best_loss_iteration)
            plt.plot(mean_loss_iteration)
            plt.plot(worst_loss_iteration)
            plt.plot(best_food_loss_iteration)

            plt.subplot(2, 1, 2)
            plt.title('Food')
            plt.xlabel('Food')
            plt.ylabel('n_organisms')
            plt.hist(food)

            plt.draw()
            plt.pause(0.001)

            plt.savefig('loss.png')

        if iteration % 10 == 0:
            with open('loss_iteration','wb') as fp:
                pickle.dump({
                    'best_loss_iteration': best_loss_iteration,
                    'mean_loss_iteration': mean_loss_iteration,
                    'worst_loss_iteration': worst_loss_iteration,
                    'best_food_loss_iteration': best_food_loss_iteration,
                }, fp)

            with open('organism_iteration','wb') as fp:
                pickle.dump({
                    'best_food_organism_iteration': best_food_organism_iteration
                }, fp)

def main():
    train()

if __name__ == '__main__':
    main()
