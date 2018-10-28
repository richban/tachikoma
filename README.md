# Comparing Braintenberg vehicles evolved using classical evolutionary algorithms and neuroevolution
The purpose of this project is to compare different evolution strategies to the evolution 
and optimization of Brainteberg vehicle using classical evolutionary algorithms 
and neuroevolution. Braitenberg vehicles are special class of agents that can autonomously 
move around based on its sensor inputs. Braintenberg vehicles are controlled by number 
of parameters, depending how the sensors and wheels are connected we can exhibit different 
behaviors. We show how basic evolutionary algorithms (EA’s) and neuroevolution can emulate a 
Braintenberg vehicle in a way that it avoids obstacles. Although the experiments will consists of 
a simple task of navigation and obstacle avoidance, our major goal of the project on autonomous agents 
is to emphasise the main differences  of both approaches. V-Rep simulator will be used to test 
and track the evolution, however we intend to test on a real physical agent.


### Classical evolutionary algorithm

```
python evolution.py --pop 40 --n_gen 20 --time 120 --cxpb 0.2 --mutpb 0.1
```

### Neauroevolution (NEAT)

```
python neuroevolution.py --n_gen 20 --time 120 
```

