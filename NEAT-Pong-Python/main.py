from pong import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
import neat.reporting
import glob


# Create a custom reporting class for retrieving the fitness of each iteration
class FunctionReporter(neat.reporting.BaseReporter):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def post_evaluate(self, config, population, species, best_genome):
        self.function(best_genome.fitness)


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

    def test_ai(self, net):
        """
        Test the AI against a human player by passing a NEAT neural network
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(120)  # set this timer for the frame rate when playing human vs AI
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate((self.right_paddle.y, abs(self.right_paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            if decision == 1:  # AI moves up
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:  # AI moves down
                self.game.move_paddle(left=False, up=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome1, genome2, config, draw=False):
        """
        Train the AI by passing two NEAT neural networks and the NEAt config object.
        These AI's will play against each other to determine their fitness.
        """
        run = True
        start_time = time.time()

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        self.genome1 = genome1
        self.genome2 = genome2

        max_hits = 50

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_info = self.game.loop()

            self.move_ai_paddles(net1, net2)

            if draw:
                self.game.draw(draw_score=True, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            if game_info.left_score == 1 or game_info.right_score == 1 or game_info.left_hits >= max_hits:
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, net1, net2):
        """Determine where to move the left and the right paddle based on the two neural networks that control them."""
        players = [(self.genome1, net1, self.left_paddle, True), (self.genome2, net2, self.right_paddle, False)]
        for (genome, net, paddle, left) in players:
            output = net.activate(
                (paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            valid = True
            if decision == 0:  # Don't move
                genome.fitness -= 0.01  # we want to discourage this
            elif decision == 1:  # Move up
                valid = self.game.move_paddle(left=left, up=True)
            else:  # Move down
                valid = self.game.move_paddle(left=left, up=False)

            if not valid:  # If the movement makes the paddle go off the screen punish the AI
                genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration


def eval_genomes(genomes, config, show_window):
    """Run each genome against each other one time to determine the fitness."""
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i / len(genomes) * 100), end=" ")
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            pong = PongGame(win, width, height)

            force_quit = pong.train_ai(genome1, genome2, config, draw=show_window)
            if force_quit:
                quit()


max_fitness_per_generation = []


def gather_stats(max_fitness):
    max_fitness_per_generation.append(max_fitness)

    line1, = ax.plot(range(len(max_fitness_per_generation)), max_fitness_per_generation, 'r-', label='Max Fitness')

    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)  # You can adjust this value based on how fast you want the graph to update


def run_neat(config, start_fresh, show_window):
    global ax  # Make ax global so that it can be used in gather_stats

    if not start_fresh[0]:
        checkpoints = sorted(glob.glob("neat-checkpoint-*"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Restoring from checkpoint: {latest_checkpoint}")
            p = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
        else:
            print("No checkpoint found, starting fresh.")
            p = neat.Population(config)
    else:
        print("Training starting from fresh.")
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # plot a graph which demonstrated the evolution of the Neural network
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    plt.title("Maximum Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Max Fitness")

    function_reporter = FunctionReporter(gather_stats)
    p.add_reporter(function_reporter)

    winner = p.run(lambda genomes, cfg: eval_genomes(genomes, cfg, show_window), 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    # Make plot permanent
    plt.show(block=True)


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, width, height)
    pong.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Configuration of the training:
    mode = "train"  # Write here "train" for training the AI or "test" for playing against the AI
    start_fresh = [True]  # Define if the training will start from fresh or continue
    show_window = False  # Set to True if you want to see the game window (False for speeding up the training)

    if mode == "train":
        run_neat(config, start_fresh, show_window)
    elif mode == "test":
        test_best_network(config)
    test_best_network(config)
