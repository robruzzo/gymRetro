import retro
import numpy as np
import cv2 
import neat
import pickle

game='MikeTysonsPunchout-Nes'
state='GlassJoeStart'
inttype=retro.data.Integrations.CONTRIB
env = retro.make(game, state,inttype, record='.')


imgarray = []

xpos_end = 0



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


with open('MikeTysonsPunchout-Nes.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)


ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False

while not done:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    for x in ob:
        for y in x:
            imgarray.append(y)
    
    nnOutput = net.activate(imgarray)
    
    ob, rew, done, info = env.step(nnOutput)
    imgarray.clear()
    
    fitness_current+=rew
    if done:
        done = True 
        genome.fitness= fitness_current

#python -m retro.scripts.playback_movie MikeTysonsPunchout-Nes-GlassJoeStart-000000.bk2