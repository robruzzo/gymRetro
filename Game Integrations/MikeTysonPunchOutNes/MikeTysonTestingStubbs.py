import retro
import numpy as np
import cv2
import neat
import pickle



#for each genome divide the resolution making it smaller, and make a network 1 for each population
def eval_genomes(genomes, config):
	for genome_id, genome in genomes:
		ob = env.reset()
		ac = env.action_space.sample()

		#variables to hold essentially the observation of the screen
		inx, iny, inc = env.observation_space.shape

		inx=int(inx/8)
		iny=int(iny/8)

		net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)

		current_max_fitness =0
		fitness_current=0
		frame=0
		counter=0
		done = False
		foeRdKOCurrent=0
		macDoc=1
		#uncomment with the below to see neural network input
		#cv2.namedWindow("main",cv2.WINDOW_NORMAL)
		
		
		while not done:
			#render the window to see whats going on in the game, not necessary
			env.render()
			frame+=1
			#uncomment one to see what the network input will be
			#scaledimg = cv2.cvtColor(ob,cv2.COLOR_BGR2RGB)
			#scaledimg = cv2.cvtColor(ob,cv2.COLOR_BGR2GRAY)
			#scaledimg = cv2.resize(scaledimg, (iny,inx))
			ob = cv2.resize(ob, (inx,iny))
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob,(inx,iny))
			#uncomment both with one above to see the network input
			#cv2.imshow('main',scaledimg)
			#cv2.waitKey(1)
			
			imgarray=np.ndarray.flatten(ob)
			
			if macDoc==1:
				nnOutput=[0,0,0,1,0,0,0,0,0]
				env.step(nnOutput)
			else:
				nnOutput = net.activate(imgarray)
			#print("Network Output",nnOutput)
			ob,rew,done,info = env.step(nnOutput)
			foeKO=info['FoeRdKO']
			macDoc=info["MacDocSplashFlag"]

			fitness_current+=rew

			if foeKO>foeRdKOCurrent:
				foeRdKOCurrent=foeKO
				fitness_current+=500

			if done:
				done = True
				#print(genome_id, fitness_current)
			genome.fitness= fitness_current


if __name__ == '__main__':
	
	game='MikeTysonsPunchOut-Nes'
	state = 'PistonHondaSplash'

	env= retro.make(game, state, inttype=retro.data.Integrations.ALL,use_restricted_actions=retro.Actions.ALL)
	imgarray=[]

	config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
					 	neat.DefaultSpeciesSet,neat.DefaultStagnation,
					 	'config-feedforward')

	#p=neat.Checkpointer.restore_checkpoint("neat-checkpoint-210")
	#set up population using the config parameters
	p = neat.Population(config)
	#make stats reporter
	p.add_reporter(neat.StdOutReporter(True))
	stats= neat.StatisticsReporter()
	p.add_reporter(stats)
	#every 10 generations save checkpoint
	#p.add_reporter(neat.Checkpointer(10))

	winner = p.run(eval_genomes)

	fileName=game+'-'+state

	with open(fileName+'.pkl', 'wb') as output:
		pickle.dump(winner, output, 1)

#C:\Users\robru\Documents\Python Scripts\gymRetro\Game Integrations\MikeTysonPunchOutNes