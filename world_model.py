import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import argparse
#from torchvision import datasets, transforms
import game_labyrinth

from tensorboardX import SummaryWriter
tb = SummaryWriter()

# Possible things to look at:
# monte carlo agent
# random agent
# monte carlo agent with true world (comparison)
# make more labyrinths (later)
# pycolab version of human game
# rotate input

def gather_random_data():
    n_boards = 100
    game = MemorizedGame(game_labyrinth.Games(n_boards))
    
    for _ in range(100):
        random_action = np.random.choice(4, size=n_boards)
        game.step(random_action)

    return game.get_memory()

# wrapper for games that will create dataset of moves and responses
class MemorizedGame:
    def __init__(self, game):
        self.game = game
        self.cumulative_game_over = None

        self.memory = {
            'board': [],
            'action': [],
            'next_board': [],
            'reward': [],
            'is_game_over': []
        }

    def step(self, action):
        board = self.game.boards.copy()
        next_board, reward, is_game_over = self.game.step(action)

        if self.cumulative_game_over is None:
            self.cumulative_game_over = np.zeros_like(is_game_over, dtype=np.bool)

        self.memory['board'].append(board[~self.cumulative_game_over].copy())
        self.memory['action'].append(action[~self.cumulative_game_over].copy())
        self.memory['next_board'].append(next_board[~self.cumulative_game_over].copy())
        self.memory['reward'].append(reward[~self.cumulative_game_over].copy())
        self.memory['is_game_over'].append(is_game_over[~self.cumulative_game_over].copy())

        self.cumulative_game_over = np.logical_or(self.cumulative_game_over, is_game_over)
        return next_board, reward, is_game_over

    def get_memory(self):
        data = [
            torch.from_numpy(np.concatenate(self.memory[key], axis=0).astype(np.float32))
            for key in self.memory
        ]
        data = torch.utils.data.TensorDataset(*data)
        return data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4 + 4, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 1, stride=1, padding=0)

        self.conv_board = nn.Conv2d(8, 4, 1, stride=1, padding=0)

        self.conv_reward = nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.fc_reward1 = nn.Linear(4*4*4, 16)
        self.fc_reward2 = nn.Linear(16, 3)

        self.conv_game_over = nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.fc_game_over1 = nn.Linear(4*4*4, 16)
        self.fc_game_over2 = nn.Linear(16, 2)


    def forward(self, board, action):
        one_hot_board = torch.eye(4)[board.view(-1, 4*4).long()].to(action.device)
        one_hot_action = torch.eye(4)[action.view(-1, 1).long()].to(action.device)

        x = torch.cat((
            one_hot_board.view(-1, 4, 4, 4),
            one_hot_action.view(-1, 1, 1, 4).repeat((1, 4, 4, 1))
        ), dim=-1)
        x = F.relu(self.conv1(x.permute(0, 3, 1, 2)))
        x = F.relu(self.conv2(x))

        next_board = one_hot_board.view(-1, 4, 4, 4) + self.conv_board(x).permute(0, 2, 3, 1)

        reward = F.relu(self.conv_game_over(x))
        reward = F.relu(self.fc_reward1(reward.view(-1, 4*4*4)))
        reward = self.fc_reward2(reward)

        is_game_over = F.relu(self.conv_game_over(x))
        is_game_over = F.relu(self.fc_game_over1(is_game_over.view(-1, 4*4*4)))
        is_game_over = self.fc_game_over2(is_game_over)

        return next_board, reward, is_game_over

class World():
    def __init__(self, net):
        self.net = net

    def dist(self, board, action):
        next_board_pred, reward_pred, is_game_over_pred = self.net(board, action)

        dist_next_board = torch.distributions.one_hot_categorical.OneHotCategorical(logits=next_board_pred)
        dist_reward = torch.distributions.one_hot_categorical.OneHotCategorical(logits=reward_pred)
        dist_game_over = torch.distributions.one_hot_categorical.OneHotCategorical(logits=is_game_over_pred)
        
        return dist_next_board, dist_reward, dist_game_over

    def log_prob(self, board, action, next_board, reward, is_game_over):
        dist_next_board, dist_reward, dist_game_over = self.dist(board, action)

        one_hot_next_board = torch.eye(4)[next_board.view(-1, 4*4).long()]
        one_hot_next_board = one_hot_next_board.view(-1, 4, 4, 4).to(action.device)

        one_hot_reward = torch.eye(3)[reward.view(-1, 1).long() + 1]
        one_hot_reward = one_hot_reward.view(-1, 3).to(action.device)

        one_hot_game_over = torch.eye(2)[is_game_over.view(-1, 1).long()]
        one_hot_game_over = one_hot_game_over.view(-1, 2).to(action.device)

        log_prob_next_board = dist_next_board.log_prob(one_hot_next_board)
        log_prob_reward = dist_reward.log_prob(one_hot_reward)
        log_prob_game_over = dist_game_over.log_prob(one_hot_game_over)

        return log_prob_next_board, log_prob_reward, log_prob_game_over

    def sample(self, board, action):
        next_board, reward, is_game_over = self.net(board, action)
        dist_next_board, dist_reward, dist_game_over = self.dist(board, action)
        
        print(dist_next_board.probs)
        print('reward', dist_reward.probs)
        print('game_over', dist_game_over.probs)
        return (
            (torch.arange(4).float().reshape((1,-1)).to(action.device) * dist_next_board.sample()).sum(dim=-1),
            ((torch.arange(3) - 1).float().reshape((1,-1)).to(action.device) * dist_reward.sample()).sum(dim=-1),
            (torch.arange(2).float().reshape((1,-1)).to(action.device) * dist_game_over.sample()).sum(dim=-1),
        )

class WorldGame():
    def __init__(self, world, board, device):
        self.world = world
        self.device = device
        self.set_state(board)

    def set_state(self, board):
        self.board = torch.from_numpy(board.astype(np.float32)).to(self.device).detach()
        self.previous_board = self.board

    def step(self, action):
        self.previous_board = self.board
        self.board, reward, is_game_over = self.world.sample(
            self.board,
            torch.from_numpy(action.astype(np.float32)).to(self.device).detach()
        )
        return self.board.cpu().detach().numpy(), reward.cpu().detach().numpy(), is_game_over.cpu().detach().numpy()


class HumanGame:
    def __init__(self, game):
        self.game = game
    def play(self):
        import sys,tty,termios
        class _Getch:
            def __call__(self):
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(3)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch

        def get():
            key = _Getch()()
            if key == '\x1b[A':
                return 0
            elif key == '\x1b[B':
                return 2
            elif key == '\x1b[C':
                return 1
            elif key == '\x1b[D':
                return 3
            else:
                raise Exception("not an arrow")

        start_board = self.game.board
        while(True):
            self.game.board = start_board
            print(self.game.board)
            print("======")
            while(True):
                board, reward, is_game_over = self.game.step(
                    np.array([get()])
                )
                print("{} reward={}".format(board, reward))
                if (is_game_over == 1).all():
                    print("its game over bruh")
                    break
            input()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    world = World(model)
    for batch_idx, (board, action, next_board, reward, is_game_over) in enumerate(train_loader):
        board = board.to(device)
        action = action.to(device)
        next_board = next_board.to(device)
        reward = reward.to(device)
        is_game_over = is_game_over.to(device)
        optimizer.zero_grad()

        log_prob_next_board, log_prob_reward, log_prob_game_over = world.log_prob(board, action, next_board, reward, is_game_over)
        loss = -(log_prob_next_board.mean()*16 + log_prob_game_over.mean() + log_prob_reward.mean())

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(board), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            tb.add_scalar(
                "data/loss",
                loss.item(),
                epoch,
            )
            tb.add_scalars(
                "data/logprob",
                {
                    "board": log_prob_game_over.mean(),
                    "reward": log_prob_reward.mean(),
                    "game over": log_prob_game_over.mean(),
                },
                epoch,
            )

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    default_batch_size = 16
    parser.add_argument('--batch-size', type=int, default=default_batch_size, metavar='N',
                        help='input batch size for training (default: {})'.format(default_batch_size))
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    default_epochs = 1000
    parser.add_argument(
        '--epochs',
        type=int, default=default_epochs, metavar='N',
        help='number of epochs to train (default: {default_epochs})'
            .format(default_epochs=default_epochs)
    )
    default_lr = 1e-3
    parser.add_argument('--lr', type=float, default=default_lr, metavar='LR',
                        help='learning rate (default: {})'.format(default_lr))
    parser.add_argument('--momentum', type=float, default=0.3, metavar='M',
                        help='SGD momentum (default: 0.3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')


    dataset = gather_random_data()

    d = dataset[:]
    print('gathered player positions:')
    print((d[0] == 1).float().sum(dim=0))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    #test_loader = torch.utils.data.DataLoader(dataset,
    #    batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader)

    torch.save(
        {
            'model': model,
        },
        'save'
    )

    n_boards = 1
    game = game_labyrinth.Games(n_boards)
    board = game.boards
    print(np.round(board, decimals=1))
    world = World(model)
    world_game = WorldGame(world, board, device)
    HumanGame(world_game).play()

    board, reward, is_game_over = world_game.step(np.ones(1))
    print(np.round(board, decimals=1))
    print(reward)
    print(is_game_over)

def human():

    HumanGame(
        game_labyrinth.Games(n_boards=1),
    ).play()

if __name__ == '__main__':
    main()