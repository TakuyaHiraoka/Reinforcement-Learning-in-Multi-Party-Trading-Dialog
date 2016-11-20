## Name
RL4MPTD

## Overview
This code is for replicating experiments in [my SigDial paper](http://www.sigdial.org/workshops/conference16/proceedings/pdf/SIGDIAL5.pdf) and [JSAI paper](https://www.jstage.jst.go.jp/article/tjsai/31/4/31_B-FC1/_pdf). 

## Description
In RL4MPTD, several combinations of reinforcement algorithms and reward functions are applied to learn effective dialogue policy in multi-party trading situation. 
RL4MPTD provides simulated trading environment for comparing learned policies with random policy or hand-crafted policy. 

## Demo
TBA

## Requirement
TBA

## Usage
This program has two modes; 1) Learning mode and 2) Test mode.
In Leaning mode a trading agent's policy is learnt, and in Test mode it's learnt policy is evaluated. We can activate these modes as follows: 

* Learning mode: set  variable "isLearning" in "MDPforMultiPlayerNegotiation.py" as "True". (Learner's) opponents' strategies can be setted up by "listCombinationofOpponentsStrategy". In the default setting, learner is learnt in "HxH" situation. See my SigDial paper for details of these settings. 
After executing Learning mode, this program produce learning result file (e.g., "HxH2Experiment_Learning_NFQ201631_1234_NumberOfPolicy0.txt"), it contains weights for learnt policies. 

* Test mode: 1) set  variable "isTest" in "MDPforMultiPlayerNegotiation.py" as "True". 2) Set variable "listCombinationofOpponentsStrategy" as a list containing one element that represents opponent strategies (e.g., "listCombinationofOpponentsStrategy = ["HxH"]"). 3) Prepare learnt policy file "BestPolicyNFQ.txt" that represents weights of learnt policy. We can create this file just copy and paste "Bestweight.txt" in learning result file into "BestPolicyNFQ.txt". 

## Tips
TBA

## Contribution
TBA
## Licence
TBA

## Author

[TakuyaHiroka](http://isw3.naist.jp/~takuya-h/)