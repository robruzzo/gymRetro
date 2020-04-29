import retro

game='MikeTysonsPunchOut-Nes'
state = 'PistonHondaSplash'
n=0
env= retro.make(game, state, inttype=retro.data.Integrations.ALL)
while n<100:
	action=env.action_space.sample()
	print(env.buttons)
	print(action)
	n+=1
