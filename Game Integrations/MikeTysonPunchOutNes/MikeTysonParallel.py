import retro
import numpy as np
import cv2
import neat
import pickle


class Worker(object):
    def __init__(self, genome, config):
        self.genome=genome
        self.config=config
        self.game='MikeTysonsPunchout-Nes'
        self.state='GlassJoeStart'
        
    def work(self):
        #create an environment
        self.env=retro.make(self.game,self.state,inttype=retro.data.Integrations.CONTRIB)
        #Reset the environment
        ob=self.env.reset()
        ac = self.env.action_space.sample()

        #variables to hold an observation, a screenshot
        inx, iny, inc = self.env.observation_space.shape

        inx=int(inx/8)
        iny=int(iny/8)

        net = neat.nn.FeedForwardNetwork.create(self.genome,self.config)

        current_max_fitness =0
        fitness_current=0
        frame=0
        counter=0
        done = False
        xpos_hi=0
        xpos_hi_max=0
        xpos_low_max=0

        while not done:
            #env.render()
            frame+=1
            ob = cv2.resize(ob, (inx,iny))
            ob = cv2.cvtColor(ob,cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob,(inx,iny))
            imgarray=np.ndarray.flatten(ob)

    
            #get the output of the neural network
            nnOutput = net.activate(imgarray)
            #print("Network Output",nnOutput)
            #step the environment and receive the output for use in the fitness functions
            ob,rew,done,info = self.env.step(nnOutput)
            fitness_current+=rew
            if done:
                done = True
                #print(genome_id, fitness_current)
            self.genome.fitness= fitness_current
        return self.genome.fitness

def eval_genomes(genome, config):
    task=Worker(genome,config)
    return task.work()

config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,neat.DefaultStagnation,
                     'config-feedforward')

if __name__ == '__main__':
    game='MikeTysonsPunchout-Nes'
    state='GlassJoeStart'
    #Set up a population using a prior checkpoint
    p=neat.Checkpointer.restore_checkpoint("neat-checkpoint-107")
    #set up population using the config parameters
    #p = neat.Population(config)
    #make stats reporter
    p.add_reporter(neat.StdOutReporter(True))
    stats= neat.StatisticsReporter()
    p.add_reporter(stats)
    #every 10 generations save checkpoint
    p.add_reporter(neat.Checkpointer(20))

    pe=neat.ParallelEvaluator(20,eval_genomes)

    #When the state is complete, save the winning genome
    winner = p.run(pe.evaluate)

    with open(game+'.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)