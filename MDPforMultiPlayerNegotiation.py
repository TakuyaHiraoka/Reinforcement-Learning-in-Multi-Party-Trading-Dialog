# coding:shift-jis
#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with several optimization algorithms
# on the CartPoleEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
#########################################################################

#�o�O�C��2014
#AgentLearner�̓��s��
#Offer,Accept���Ă�CountKeep��0�ɂȂ�Ȃ�
#Offer���󂯂����肪���s��������
#ExecuteOffer����Ofer�e�[�u���̌v�Z���@�̃o�O
#
#�o�C�A�X�H��ǉ�2014/12/13

#������item1�������o���āA�I�t�@�[���ꂽ�G�[�W�F���g(agent)��item2���󂯎��\���o�B
#DialogueProtocols.listSystemAction.append("Offer_"+agent+"_"+item1+"_"+item2)

#�I�t�@�[���Ƃ̃G�[�W�F���g�iagent�j��item1�ƃI�t�@�[���ꂽ�G�[�W�F���g��item2�̌����B
#DialogueProtocols.listSystemState.append("OfferedFrom_"+agent+"_"+item1+"_"+item2)
#DialogueProtocols.listDialogStateinSpecialCase.append("OfferedFrom_"+agent+"_"+item1+"_"+item2)


__author__ = "Takuya Hiraoka"

from numpy import *
from scipy import ndarray
import re
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.explorers import BoltzmannExplorer
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.environments import *
from pybrain.rl.environments.environment import *
from pybrain.rl.agents import *
from pybrain.rl.agents.linearfa import *
from pybrain.rl.learners import *
from pybrain.rl.learners.valuebased.linearfa import *
from pybrain.rl.learners.directsearch import *
from pybrain.rl.explorers import *
from pybrain.structure import SoftmaxLayer
import sys,datetime
import copy
import unicodedata
import warnings
import traceback
import pickle
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer

#�������������߂�ϐ�
class ExperimentalConditions:#�m�[�g ���󂩂瓮���Ȃ��t�r�̏���
    #�����������F�Η��Ґ���2�̏ꍇ�̕]��.LinQ,LSPI,NFQ
    #Pybrian�ύX�_
    #NFQ��ActivationNetwork���̒��ԑw*10
    #LSPI��ExprolationReward->400
    
    isLearning=False#�w�K���s����
    isTest=True#�����]�����s����
    
    #�w�K����
    isUseLinearQ=False#���`�֐��łp�֐����ߎ����邩�B�e���������̏ꍇ�A�j���[�����l�b�g�𗘗p����
    #�ꕔ��������[���Œu�������邩�B��D���ō��̐��ʂ̂Ƃ���Keep�ɂ���
    isUsePartiallyRulePolicy=False#False
    
    #�x�[�X���C������1
    isUseHandCraftedPolicy=False#�萻�̕���Œu�������邩�H
    #
    normofLearner="Uninformative"
    #�v�����j���O���ǂ̒��x�̐[���܂ōs����
    searchDepth=2
    
    #�x�[�X���C������2
    isUseRandomPolicy=False#�����_������Œu�������邩�H
    #�x�[�X���C������R
    isAlwaysKeeping=False#��ɃL�[�v���邩
    
    #��V�����݂̎�̉��P���ɉ����Ē����^���邩
    #False�̏ꍇ�͑Θb���I����Ă���^������
    isFeedRewardAsImprovement=False
    
    #�e���̃A�C�e�����ƃI�t�@�[�e�[�u���ȊO�ɑΘb���𗘗p����
    #-���^�[���ȓ��ɒB���\�ȃS�[��
    #-�e���̏����A�C�e��
    #-�o�߃^�[��
    #-�~���݂̕�V
    #-�w�K�G�[�W�F���g������Ă���O�̑S�G�[���Ƃ�Accept/offer�e�[�u��
    #-�ߋ���Offer/Accept�̗���
    ###Not working (i.e., additional state are doesn't added), but it always should be true. 
    isAddAdditionalDStoLearner=True
    
    #-�w�K/�����]�����̎����p�����[�^
    numSystems=1#�w�K���s���V�X�e������̐��B���ꂼ�ꃂ�f���̏����l���قȂ�
    numBatch=10#200#�w�K/�����]�����̃p�����[�^�̍X�V��
    numEpisode=2000#2000#�w�K/�����]�����̊e�o�b�`�ōs���Θb�̉�
    numBatchStartLogging=2#�w�K���̍œK�ȕ���̋L�^���J�n����Œ�o�b�`��

    #�ő�^�[���� 50->20
    iMaximumTurn=20#�g���[�X���������20���x�ŏ\��
    
    #�����������Ƃ��Ċw�K��ł��؂�臒l
    dBadConversion=-500.0
    
    #�G�[�W�F���g�̍s���̃g���[�X
    isTraceAgent=False
    #Learner(Agent0)�̓��v�ʌv�Z
    isCalculateStatisticsOfLearner=True
    
    #�������s���ۂ̑ΐ푊��̑g�ݍ��킹�BH,R�Ő헪�w��Bx�ŋ�؁B��:H, HxH, HxRxH
    #H=�n���h�N���t�g�AR=�����_��
    listCombinationofOpponentsStrategy=[
                                        #"H","R",
                                        "HxH","HxR","RxR", 
                                        #"HxHxH","HxHxR","HxRxR","RxRxR"
                                        ]
    #���݂̑g�ݍ��킹
    currentCombinationofOpponentsStrategy=None

#JSAI�p�̕��͂̂��߂�Learner(Agent0�̓��v�� #�����S�̂�ʂ���)
class SatisticsOfLearnerInExperiment:
    #Distribution of Action
    #-Offer
    totalNumberOfOffer=0.0#offer�̑���
    totalNumberOfOfferAcceptedByAddressee=0.0#����ɃA�N�Z�v�g���ꂽoffer�̑���
    #totalNumberOfOfferRejectedByAddressee=0.0#����Ƀ��W�F�N�g���ꂽoffer�̑���
    
    #-Accept
    totalNumberOfAccept=0.0#�I�t�@�[�ɑ΂���A�N�Z�v�g�̐�
    totalNumberOfIlligalAccept=0.0#�s���ȃA�N�Z�v�g�̐��ie.g., Offer�������̂ɃA�N�Z�v�g����Ȃǁj
    totalNumberOflligalAccept=0.0#�s���ȃA�N�Z�v�g�̐��ie.g., Offer�������̂ɃA�N�Z�v�g����Ȃǁj
    #-Keep
    totalNumberOfKeep=0.0#Keep�̑���
    
    #
    totalNumberOfAction=0.0#Agent0���A�N�V�������s��������
    totalNumberOfTotalTurn=0.0#�Θb�̑��^�[����
    totalNumberOfDialogue=0.0#���Θb��
    
    
#���̍ۂ̘g�g�݂̋K��
class DialogueProtocols:
    #�V�X�e���̃A�N�V����
    #e.g. 20 actions are generated.
    #['Accept', 'Keep', 'Offer_AgentLeaner_Apple_Orange', 'Offer_AgentLeaner_Apple_Grape', 'Offer_AgentLeaner_Orange_Apple', 'Offer_AgentLeaner_Orange_Grape', 'Offer_AgentLeaner_Grape_Apple', 'Offer_AgentLeaner_Grape_Orange', 'Offer_Agent0_Apple_Orange', 'Offer_Agent0_Apple_Grape', 'Offer_Agent0_Orange_Apple', 'Offer_Agent0_Orange_Grape', 'Offer_Agent0_Grape_Apple', 'Offer_Agent0_Grape_Orange', 'Offer_Agent1_Apple_Orange', 'Offer_Agent1_Apple_Grape', 'Offer_Agent1_Orange_Apple', 'Offer_Agent1_Orange_Grape', 'Offer_Agent1_Grape_Apple', 'Offer_Agent1_Grape_Orange']
    listSystemAction=None
    #���ʂȃA�N�V����������������
    listSystemActionForLearner=None
    
    #�V�X�e���̏��
    #27 states are generated.
    #['OfferedFrom_AgentLeaner_Apple_Orange', 'OfferedFrom_AgentLeaner_Apple_Grape', 'OfferedFrom_AgentLeaner_Orange_Apple', 'OfferedFrom_AgentLeaner_Orange_Grape', 'OfferedFrom_AgentLeaner_Grape_Apple', 'OfferedFrom_AgentLeaner_Grape_Orange', 'OfferedFrom_Agent0_Apple_Orange', 'OfferedFrom_Agent0_Apple_Grape', 'OfferedFrom_Agent0_Orange_Apple', 'OfferedFrom_Agent0_Orange_Grape', 'OfferedFrom_Agent0_Grape_Apple', 'OfferedFrom_Agent0_Grape_Orange', 'OfferedFrom_Agent1_Apple_Orange', 'OfferedFrom_Agent1_Apple_Grape', 'OfferedFrom_Agent1_Orange_Apple', 'OfferedFrom_Agent1_Orange_Grape', 'OfferedFrom_Agent1_Grape_Apple', 'OfferedFrom_Agent1_Grape_Orange', 'NumItem_AgentLeaner_Apple', 'NumItem_AgentLeaner_Orange', 'NumItem_AgentLeaner_Grape', 'NumItem_Agent0_Apple', 'NumItem_Agent0_Orange', 'NumItem_Agent0_Grape', 'NumItem_Agent1_Apple', 'NumItem_Agent1_Orange', 'NumItem_Agent1_Grape']
    listSystemState=None
    
    #�h���C���ŗL�̃��[��
    #�G�[�W�F���g�̖̂��O
    listAgentsName=None
    #�A�C�e��
    listItems=["Apple", "Orange", "Grape"]
    #�A�C�e���ւ̚n�D
    dicPreference={"Like":100.0, "Neutral":0.0, "Hate":-100.0}
    #����
    dicRole={"Rich":4, "Middle":3, "Poor":2}
    
    #�A�C�e���̌��ɉ������B���\�ȖڕW�e�[�u��@Rich man
    listAchievableRichHands=[
                        {"Apple":0, "Orange":0, "Grape":4},
                        {"Apple":0, "Orange":1, "Grape":3},
                        {"Apple":1, "Orange":0, "Grape":3},
                        {"Apple":0, "Orange":2, "Grape":2},
                        {"Apple":1, "Orange":1, "Grape":2},
                        {"Apple":2, "Orange":0, "Grape":2},
                        {"Apple":0, "Orange":3, "Grape":1},
                        {"Apple":1, "Orange":2, "Grape":1},
                        {"Apple":2, "Orange":1, "Grape":1},
                        {"Apple":3, "Orange":0, "Grape":1},
                        {"Apple":0, "Orange":4, "Grape":0},
                        {"Apple":1, "Orange":3, "Grape":0},
                        {"Apple":2, "Orange":2, "Grape":0},
                        {"Apple":3, "Orange":1, "Grape":0},
                        {"Apple":4, "Orange":0, "Grape":0},
                        ]
    #�A�C�e���̌��ɉ������B���\�ȖڕW�e�[�u��@Normal man
    listAchievableNormalHands=[
                        {"Apple":0, "Orange":0, "Grape":3},
                        {"Apple":0, "Orange":1, "Grape":2},
                        {"Apple":1, "Orange":0, "Grape":2},
                        {"Apple":0, "Orange":2, "Grape":1},
                        {"Apple":1, "Orange":1, "Grape":1},
                        {"Apple":2, "Orange":0, "Grape":1},
                        {"Apple":0, "Orange":3, "Grape":0},
                        {"Apple":1, "Orange":2, "Grape":0},
                        {"Apple":2, "Orange":1, "Grape":0},
                        {"Apple":3, "Orange":0, "Grape":0},
                        ]
    #�A�C�e���̌��ɉ������B���\�ȖڕW�e�[�u��@Poor man
    listAchievablePoorHands=[
                        {"Apple":0, "Orange":0, "Grape":2},
                        {"Apple":0, "Orange":1, "Grape":1},
                        {"Apple":1, "Orange":0, "Grape":1},
                        {"Apple":0, "Orange":2, "Grape":0},
                        {"Apple":1, "Orange":1, "Grape":0},
                        {"Apple":2, "Orange":0, "Grape":0},
                        ]
    #�ΐ푊��̐���
    listNormOpponent=["Optimist", "Uninformative","Pessimist"]
    
    #�ǉ������Θb��Ԃ̕]���悤
    listDialogStateinSpecialCase=[]
    
    #�ΐ푊��̐U�镑���̃��X�g
    listTypeofOpponentsStrategy=["Random","Handcrafted"]
    
#     #TEST
#     totalOffer=0.0
#     totalAccept=0.0
    
class TestEnv(Environment):
    def __init__(self):
        Environment.__init__(self)
        #��ԏ������̐ݒ�
        self.reset()
    def reset(self):
        #����
        #�e�ΐ푊��i�G�[�W�F���g�j�̏��(PayOff�s��, �����A�C�e��)�̏����� 
        self.dicAgents={}
        #�e�G�[�W�F���g�̃A�C�e���̑���
        tNumItemAgent={}
        for agent in DialogueProtocols.listAgentsName:
            #Payoff�s��̐���--���ꂼ��̚n�D���ЂƂ��A�C�e���ɑΉ�
            #�A�C�e��:�n�D
            dicPayoff={}
            temp=DialogueProtocols.dicPreference.keys()
            for item in DialogueProtocols.listItems: 
                dicPayoff[item]=None
            for item in DialogueProtocols.listItems: 
                num=random.randint(low=0,high=99999)%len(temp)
                dicPayoff[item]=temp[num]
                temp.pop(num)
            #print dicPayoff
            #�����A�C�e���̐���
            numTotalItem=DialogueProtocols.dicRole[DialogueProtocols.dicRole.keys()[random.randint(low=0,high=99999)%len(DialogueProtocols.dicRole.keys())]]
            tNumItemAgent[agent]=numTotalItem
            #numTotalItem=3#�݂�ȓ������R�����Ă���
            dicNumItems={}
            for i in range(len(DialogueProtocols.listItems)):
                dicNumItems[DialogueProtocols.listItems[i]]=0
            for i in range(numTotalItem):
                dicNumItems[DialogueProtocols.listItems[random.randint(low=0,high=99999)%len(DialogueProtocols.listItems)]] +=1
            #������Ԑ���
            dicInfo={}
            dicInfo["InitNumItems"]=dicNumItems
            dicInfo["Payoff"]=dicPayoff
            self.dicAgents[agent]=dicInfo
        #print self.dicAgents
        #Learner�̏����̌Œ�
        dicNumItems={}
        for i in range(len(DialogueProtocols.listItems)):
            dicNumItems[DialogueProtocols.listItems[i]]=0
        for i in range(3):
            dicNumItems[DialogueProtocols.listItems[random.randint(low=0,high=99999)%len(DialogueProtocols.listItems)]] +=1
        self.dicAgents["AgentLearner"]["InitNumItems"]=dicNumItems.copy()
        self.dicAgents["AgentLearner"]["Payoff"]={DialogueProtocols.listItems[0]:DialogueProtocols.dicPreference.keys()[0],DialogueProtocols.listItems[1]:DialogueProtocols.dicPreference.keys()[1],DialogueProtocols.listItems[2]:DialogueProtocols.dicPreference.keys()[2]}
        #Pay Learner�ȊO��Payoff�̌Œ�
        #self.dicAgents["Agent0"]["Payoff"]={DialogueProtocols.listItems[0]:DialogueProtocols.dicPreference.keys()[0],DialogueProtocols.listItems[1]:DialogueProtocols.dicPreference.keys()[1],DialogueProtocols.listItems[2]:DialogueProtocols.dicPreference.keys()[2]}
        #self.dicAgents["Agent1"]["Payoff"]={DialogueProtocols.listItems[1]:DialogueProtocols.dicPreference.keys()[0],DialogueProtocols.listItems[2]:DialogueProtocols.dicPreference.keys()[1],DialogueProtocols.listItems[0]:DialogueProtocols.dicPreference.keys()[2]}

        #print self.dicAgents
        
        #���O�̃G�[�W�F���g�̃A�N�V����
        self.mostPreviousAction=""
        self.mostPreviousAgent=""
        #�����_���I��p�̃G�[�W�F���g�̃X�^�b�N
        self.listOrgderAgentsTakingAction=[]
        #�o�߃^�[��
        self.turn=0
        #�A��Keep��
        self.numContKeep=0
        
        #�e�G�[�W�F���g���B���\�Ȑ��ʁF�v�����j���O������p�ϐ�
        self.achievableOutcome=None
        #�e�G�[�W�F���g�̃A�C�e���̑����F�v�����j���O������p�ϐ�
        self.totalNumofItemsofAgents=None
        
        #�G�[�W�F���g�����̌���
        self.normofAgents={}
        for agent in DialogueProtocols.listAgentsName:
            self.normofAgents[agent]="Uninformative"#DialogueProtocols.listNormOpponent[random.randint(low=0,high=99999)%len(DialogueProtocols.listNormOpponent)]
        self.normofAgents["AgentLearner"]=ExperimentalConditions.normofLearner        
        #�e�G�[�W�F���g���v�����x�[�X�̒T���ŉ����܂Ōv�Z���邩1~4
        self.searchDepth={}
        for agent in DialogueProtocols.listAgentsName:
            self.searchDepth[agent]=tNumItemAgent[agent]#(random.randint(low=0,high=9999)%(tNumItemAgent[agent]-1))+1
        self.searchDepth["AgentLearner"]=ExperimentalConditions.searchDepth
        #�ΐ푊��̐헪�̌���
        self.listOpponentAgentStrategy={"AgentLearner":None}
        i=0
        for agent in DialogueProtocols.listAgentsName:
            if agent == "AgentLearner":
                pass
            else:
                if ExperimentalConditions.currentCombinationofOpponentsStrategy.split("x")[i] == "H":
                    self.listOpponentAgentStrategy[agent]=DialogueProtocols.listTypeofOpponentsStrategy[1]
                elif ExperimentalConditions.currentCombinationofOpponentsStrategy.split("x")[i] == "R":
                    self.listOpponentAgentStrategy[agent]=DialogueProtocols.listTypeofOpponentsStrategy[0]
                else:
                    assert False, "Illegal ordering for combination"
                i+=1
        #Test
        #print self.listOpponentAgentStrategy
        
        #���L���ꂽ�Θb��Ԃ̏�����
        self.dicsharedDialogState={}
        for ds in DialogueProtocols.listSystemState: 
            self.dicsharedDialogState[ds]=0.0
        #�A�C�e���̏�����
        for ds in self.dicsharedDialogState.keys():
            for agent in DialogueProtocols.listAgentsName:
                for item in DialogueProtocols.listItems: 
                    if (re.search("NumItem", ds)!=None) and (re.search(agent, ds)!=None) and (re.search(item, ds)!=None):
                        self.dicsharedDialogState[ds] = self.dicAgents[agent]["InitNumItems"][item]
        #-�o�C�A�X�e�̐ݒ�
        self.dicsharedDialogState["Bias"]=1.0
        
        #print self.dicsharedDialogState
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            #-�����A�C�e����ݒ�
            pass
        #print self.dicsharedDialogState
        
        
        
        #�G�[�W�F���g�̍s���̃g���[�X
        if ExperimentalConditions.isTraceAgent:
            print "\n\n"
            print "New Dialogue started"
            for agent in DialogueProtocols.listAgentsName:
                print agent + "'s preference: "
                for item in DialogueProtocols.listItems:
                    print item + ":" + str(self.dicAgents[agent]["Payoff"][item])
            for agent in DialogueProtocols.listAgentsName:
                print agent + " have: "
                for item in DialogueProtocols.listItems:
                    print item + ":" + str(self.dicsharedDialogState["NumItem_"+agent+"_"+item])
            print "Opponents Strategy"
            print self.listOpponentAgentStrategy

        #�G�[�W�F���g���S�[���ɒB������
        self.isLearnerReachMaximumOutcome=False

        #1�^�[���O�̊w�K�҂̏��
        self.previousLeanerDialogState=copy.copy(self.dicsharedDialogState)
                        
    def performAction(self, action):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            SatisticsOfLearnerInExperiment.totalNumberOfAction+=1.0
        
        #�Θb��Ԃ̃o�b�N�A�b�v
        self.previousLeanerDialogState=copy.copy(self.dicsharedDialogState)
        
        #�G�[�W�F���g�̃g���[�X�p
        if ExperimentalConditions.isTraceAgent:
            print"AgentLearners action=" + DialogueProtocols.listSystemActionForLearner[int(action[0])]
        #Offer/Accept Table�̏�����
        for agent1 in DialogueProtocols.listAgentsName:#Offer�����G�[�W�F���g
            for agent2 in DialogueProtocols.listAgentsName:#Offer���ꂽ�G�[�W�F���g
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if item1 != item2:
                            self.dicsharedDialogState["Accept_"+agent1+"_"+agent2+"_"+item1+"_"+item2]=0
                            self.dicsharedDialogState["FO_"+agent1+"_"+agent2+"_"+item1+"_"+item2]=0

        #�V�X�e���G�[�W�F���Ƃ̍s��
        #-Learner�̏ꍇ
        #-�����_��������g���ꍇ
        if ExperimentalConditions.isUseRandomPolicy:
            #�������[���̓K��            #TEST Fruit sarad�̏ꍇ�͋����I�Ƀp�X
            numEachItem=[]
            for item in DialogueProtocols.listItems:
                numEachItem.append(self.dicsharedDialogState["NumItem_AgentLearner_"+item])
            min(numEachItem)
            if ExperimentalConditions.isUsePartiallyRulePolicy and (min(numEachItem) > 0.5):
                self._DSupdateDoNothing("AgentLearner")
                self.isLearnerReachMaximumOutcome=True
            else:
                self._RandomValidateActionPolicy("AgentLearner")
        #��ɃL�[�v��������ꍇ
        elif ExperimentalConditions.isAlwaysKeeping:
            self._DSupdateDoNothing("AgentLearner")
        #-�w�K������g���ꍇ
        elif (not ExperimentalConditions.isUseHandCraftedPolicy):
#             #TEST
#             action[0]=random.randint(low=0,high=9999)%len(DialogueProtocols.listSystemActionForLearner)
            
            #�������[���̓K��            #TEST Fruit sarad�̏ꍇ�͋����I�Ƀp�X
            numEachItem=[]
            for item in DialogueProtocols.listItems:
                numEachItem.append(self.dicsharedDialogState["NumItem_AgentLearner_"+item])
            min(numEachItem)
            if ExperimentalConditions.isUsePartiallyRulePolicy and (min(numEachItem) > 0.5):
                self._DSupdateDoNothing("AgentLearner")
                self.isLearnerReachMaximumOutcome=True
            #�Ȍ�w�K���ꂽ����
            elif re.search("Offer", DialogueProtocols.listSystemActionForLearner[int(action[0])]) != None:
                if re.search("AgentLearner", DialogueProtocols.listSystemActionForLearner[int(action[0])]) == None:
                    pAct=DialogueProtocols.listSystemActionForLearner[int(action[0])].split("_")
                    self._DSupdateExecuteOffer("AgentLearner", pAct[1], pAct[2], pAct[3])
#                     #TEST
#                     DialogueProtocols.totalOffer+=1.0

                else:
                    self._DSupdateDoNothing("AgentLearner")#�����ɃI�t�@�[�����Ȃ�
                    #TEST Keep���G�[�W�F���g�����Ȃ��悤���w�K�������ŏI���悤��Keep�����u�ԂɑΘb���I������
                    #self.numContKeep+=1000
                    #print "Self offer is Acvoided"                
            elif re.search("Accept", DialogueProtocols.listSystemActionForLearner[int(action[0])]) != None:
                if re.search("Offer", self.mostPreviousAction) != None:
                    pAct=self.mostPreviousAction.split("_")
                    self._DSupdateExecuteTrade("AgentLearner", self.mostPreviousAgent, pAct[3], pAct[2])
#                     #TEST
#                     DialogueProtocols.totalOffer+=1.0
                else:
                    self._DSupdateDoNothing("AgentLearner")
                    #TEST Keep���G�[�W�F���g�����Ȃ��悤���w�K�������ŏI���悤��Keep�����u�ԂɑΘb���I������
                    #self.numContKeep+=1000
            elif re.search("Keep", DialogueProtocols.listSystemActionForLearner[int(action[0])]) != None:
                self._DSupdateDoNothing("AgentLearner")
                #TEST Keep���G�[�W�F���g�����Ȃ��悤���w�K�������ŏI���悤��Keep�����u�ԂɑΘb���I������
                #self.numContKeep+=1000
            else:
                assert False, "System didnt take any action."
        else:
            #self._GreedywisePolicy("AgentLearner")
            self._PlanbasedPolicy("AgentLearner")
            #self._RandomPlanInBetterGoalbasedPolicy("AgentLearner")#TEST
        self.turn+=1
        #print DialogueProtocols.listSystemAction[int(action[0])]
        
        #���̌�̏�������&�^�[���i�s
        agentTakingAction=""
        while agentTakingAction !="AgentLearner" and (self.turn < ExperimentalConditions.iMaximumTurn):
            if (self.numContKeep >=len(self.dicAgents.keys())):
                break
            elif self.isLearnerReachMaximumOutcome:
                break
            #��������
            #-���O�̃G�[�W�F���g�̃A�N�V����������̃G�[�W�F���g�ɑΐ�Offer�̏ꍇ�AOffer���󂯂��G�[�W�F���g���D��I�ɓ�����
            if re.search("Offer_",self.mostPreviousAction) !=None:
                for agent in DialogueProtocols.listAgentsName: 
                    if re.search(agent,self.mostPreviousAction) != None:
                        agentTakingAction=agent
                        #Offer���󂯂��G�[�W�F���g�̘A���s�����K��
                        if not self.listOrgderAgentsTakingAction == None:
                            for elem in self.listOrgderAgentsTakingAction:
                                if agentTakingAction == elem:
                                    self.listOrgderAgentsTakingAction.remove(agentTakingAction)
            #-�����łȂ��ꍇ�́A�����_������
            else:
                if len(self.listOrgderAgentsTakingAction) ==0: #���X�g�̒������O�Ȃ珉����
                    temp=copy.copy(DialogueProtocols.listAgentsName)
                    while len(temp) > 0:
                        self.listOrgderAgentsTakingAction.append(temp.pop(random.randint(low=0,high=99999)%len(temp)))
                    self.listOrgderAgentsTakingAction.remove(self.mostPreviousAgent)
                    self.listOrgderAgentsTakingAction.append(self.mostPreviousAgent)
                agentTakingAction=self.listOrgderAgentsTakingAction.pop(0)
            assert agentTakingAction != None, "Illegal agent selection"
            #print self.listOrgderAgentsTakingAction
            
            #�w�肳�ꂽ�G�[�W�F���g�̍s��-Learner�ł͖����ꍇ
            if agentTakingAction != "AgentLearner":
                #self._GreedywisePolicy(agentTakingAction)
                if self.listOpponentAgentStrategy[agentTakingAction]=="Random":
                    self._RandomValidateActionPolicy(agentTakingAction)
                elif self.listOpponentAgentStrategy[agentTakingAction]=="Handcrafted":
                    self._PlanbasedPolicy(agentTakingAction)
                else:
                    assert False, "Illegal Opponents strategy"
                self.turn+=1

        #�ǉ��̑Θb��Ԃ̒ǉ�
        #
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            self.dicsharedDialogState["CurrentTurn"]=self.turn
    
    #�L���ȃA�N�V�����݂̂���Ȃ郉���_������
    def _RandomValidateActionPolicy(self,agentTakingAction):
        #�L����Offer�W���𒲂ׂ�
        validateAct=[]
        for act in DialogueProtocols.listSystemAction:
            if re.search("Offer",act) != None:
                if re.search(agentTakingAction, act) == None:
                    pAct=act.split("_")
                    if self._isAvailableTrade(agentTakingAction, pAct[1], pAct[2], pAct[3]):
                        validateAct.append(act)
        validateAct.append("Accept")
        validateAct.append("Keep")
        
                    
        action=random.randint(low=0,high=99999)%len(validateAct)
        
        if re.search("Offer", validateAct[int(action)]) != None:
            if re.search(agentTakingAction, validateAct[int(action)]) == None:
                pAct=validateAct[int(action)].split("_")
                self._DSupdateExecuteOffer(agentTakingAction, pAct[1], pAct[2], pAct[3])
            else:
                self._DSupdateDoNothing(agentTakingAction)#�����ɃI�t�@�[�����Ȃ�
                #print "Self offer is Acvoided"                
        elif re.search("Accept", validateAct[int(action)]) != None:
            if re.search("Offer", self.mostPreviousAction) != None:
                pAct=self.mostPreviousAction.split("_")
                self._DSupdateExecuteTrade(agentTakingAction, self.mostPreviousAgent, pAct[3], pAct[2])
            else:
                self._DSupdateDoNothing(agentTakingAction)
        elif re.search("Keep", validateAct[int(action)]) != None:
            self._DSupdateDoNothing(agentTakingAction)
        else:
            assert False, "System didnt take any action."
    
    #�����_������
    def _RandomPolicy(self,agentTakingAction):
        action=random.randint(low=0,high=99999)%len(DialogueProtocols.listSystemAction)

        if re.search("Offer", DialogueProtocols.listSystemAction[int(action)]) != None:
            if re.search(agentTakingAction, DialogueProtocols.listSystemAction[int(action)]) == None:
                pAct=DialogueProtocols.listSystemAction[int(action)].split("_")
                self._DSupdateExecuteOffer(agentTakingAction, pAct[1], pAct[2], pAct[3])
            else:
                self._DSupdateDoNothing(agentTakingAction)#�����ɃI�t�@�[�����Ȃ�
                #print "Self offer is Acvoided"                
        elif re.search("Accept", DialogueProtocols.listSystemAction[int(action)]) != None:
            if re.search("Offer", self.mostPreviousAction) != None:
                pAct=self.mostPreviousAction.split("_")
                self._DSupdateExecuteTrade(agentTakingAction, self.mostPreviousAgent, pAct[3], pAct[2])
            else:
                self._DSupdateDoNothing(agentTakingAction)
        elif re.search("Keep", DialogueProtocols.listSystemAction[int(action)]) != None:
            self._DSupdateDoNothing(agentTakingAction)
        else:
            assert False, "System didnt take any action."

    #Hand-crafted policy�O��(Proposed by David-san) 2014/1/06
    #-���݂̏�Ԃ���B���\�ȃS�[���Ɍ����Ẵv�����𐶐�����B
    #-�����āA���̒B���\�ȃv�������烉���_���ɏ]���v������I��
    #-�����āA��������鐬�ʂ��ł��ǂ����̂�I�ԁB������ǂ����ʂ����҂ł��Ȃ��ꍇ�́AKeep.
    def _RandomPlanInBetterGoalbasedPolicy(self, agentTakingAction):
        #�e�G�[�W�F���g�̏����A�C�e����Payoff�ɉ��������ʃe�[�u�����쐬�F
        if self.achievableOutcome == None:
            self.achievableOutcome={}#�A�C�e���`�Q�A�C�e���`�̌�_�A�C�e���a�Q�A�C�e���a�̌�_�A�C�e���b�Q�A�C�e���b�̌�
            self.totalNumofItemsofAgents={}
            #-���ʃe�[�u���̐���
            #--�e�G�[�W�F���g�̃A�C�e���̑��������߂�
            dicTotalNumofItemofAgents={}
            for agent in self.dicAgents.keys():
                totalNum= 0
                for item in self.dicAgents[agent]["InitNumItems"].keys():
                    totalNum+=self.dicAgents[agent]["InitNumItems"][item]
                dicTotalNumofItemofAgents[agent]=totalNum
            self.totalNumofItemsofAgents=copy.copy(dicTotalNumofItemofAgents)
            #--���ꂼ��̎������̒ʂ�ɉ��������ʂ��v�Z����
            #---�A�C�e�����ɂ��������\�Ȏ�D�̐ݒ�
            dicAchievableHandofAgents={}
            for agent in dicTotalNumofItemofAgents.keys():
                if dicTotalNumofItemofAgents[agent] == 2:
                    dicAchievableHandofAgents[agent]=copy.copy(DialogueProtocols.listAchievablePoorHands)
                elif dicTotalNumofItemofAgents[agent] == 3:
                    dicAchievableHandofAgents[agent]=copy.copy(DialogueProtocols.listAchievableNormalHands)
                elif dicTotalNumofItemofAgents[agent] == 4:
                    dicAchievableHandofAgents[agent]=copy.copy(DialogueProtocols.listAchievableRichHands)
                else:
                    assert False, "Illegal achievable goal"
            #---�e����Payoff�s��ɉ��������ʂ̐ݒ�
            for agent in self.dicAgents.keys():
                #�e�G�[�W�F���g�̎育�Ƃ̕�V
                outcomes={}
                for hand in dicAchievableHandofAgents[agent]:
                    rew=0.0
                    numEachItem=[]
                    for item in DialogueProtocols.listItems:
                        rew+=DialogueProtocols.dicPreference[self.dicAgents[agent]["Payoff"][item]]*hand[item]
                        numEachItem.append(hand[item])
                    #�t���[�c�T���_
                    rew+=500*min(numEachItem)
                    outcomes[hand.keys()[0]+"_"+str(hand[hand.keys()[0]])+"_"+hand.keys()[1]+"_"+str(hand[hand.keys()[1]])+"_"+hand.keys()[2]+"_"+str(hand[hand.keys()[2]])]=rew
                self.achievableOutcome[agent]=outcomes
                
        #���݂̃G�[�W�F���g�̎莝���̐��ʂ����������ʂ��e�[�u�������珜�O����
        achievableProfitOutcomATA={}
        #-���݂̃G�[�W�F���g�̐��ʂ��v�Z
        currentrew=0.0
        numEachItem=[]
        for item in DialogueProtocols.listItems:
            currentrew+=DialogueProtocols.dicPreference[self.dicAgents[agentTakingAction]["Payoff"][item]]*self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item]
            numEachItem.append(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item])
        currentrew+=500*min(numEachItem)
        #-���O�̎��s
        for outcomes in self.achievableOutcome[agentTakingAction].keys():
            if self.achievableOutcome[agentTakingAction][outcomes] > currentrew:
                achievableProfitOutcomATA[outcomes]=self.achievableOutcome[agentTakingAction][outcomes]
        
        #�v�����j���O�ii.e.�ŒZ�o�H�T���j
        #TODO<------------------------------------------------------------------
        #-�����̎���܂߂��e�A�C�e���̑����̃J�E���g (TODO �ŏ��ɌĂ΂ꂽ�Ƃ��ɂ̂݌v�Z����悤�ɕύX)
        numofEachItems={}
        for item in DialogueProtocols.listItems:
            totalNum=0
            for agent in DialogueProtocols.listAgentsName:
                totalNum+=self.dicAgents[agent]["InitNumItems"][item]
            numofEachItems[item]=totalNum
            
        #-�c��̃^�[���Ŗ��炩�ɒB���o���Ȃ��ꍇ�͍폜(�ڕW�ƃA�C�e����L1�������c��̎萔�ȏ�̏ꍇ�͍폜)
        availablePlanatEachHand={}
        for hand in achievableProfitOutcomATA.keys():
            #print hand.split("_")
            initHand=hand.split("_")[0]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[0]]))+"_"
            initHand+=hand.split("_")[2]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[2]]))+"_"
            initHand+=hand.split("_")[4]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[4]]))
            queue=[[initHand]]#�A�C�e�����̏�Ԍn��
            correctPlan=[]
            #���D��T��
            while len(queue) > 0:
                currentPlan=queue.pop(0)
                dist=0
                #---�����̌v�Z
                for i in range(len(hand.split("_"))/2):
                    #print int(hand.split("_")[(i*2)+1])
                    #print int(currentPlan[-1].split("_")[(i*2)+1])
                    #print str(i)
                    #print hand.split("_")
                    #print currentPlan[-1].split("_")
                    dist+=abs(int(float(hand.split("_")[(i*2)+1]))-int(float(currentPlan[-1].split("_")[(i*2)+1])))
                #--�m�[�h������ȏ�W�J����K�v�������ꍇ
                if dist == 0: #�I�[��Ԃ̏ꍇ
                    correctPlan.append(copy.copy(currentPlan))
                #TESTING NOW
                #elif dist > (self.totalNumofItemsofAgents[agentTakingAction]-len(currentPlan)+1): #����̎萔�œ��B�ł��Ȃ��ꍇ
                elif dist > (self.searchDepth[agentTakingAction]-len(currentPlan)+1): #����̎萔�œ��B�ł��Ȃ��ꍇ
                    pass
                else: 
                    #�m�[�h��W�J����K�v������ꍇ
                    #--���ȃm�[�h�̒ǋL
                    #--�m�[�h�̓W�J�F
                    for itemGive in DialogueProtocols.listItems:
                        for itemGiven in DialogueProtocols.listItems:
                            if itemGive == itemGiven:
                                continue
                            expandedHand=copy.copy(currentPlan[-1])
                            dicExpandedHand={}
                            for i in range((len(expandedHand.split("_"))/2)):
                                dicExpandedHand[expandedHand.split("_")[i*2]]=int(float(expandedHand.split("_")[(i*2)+1]))
                            #�m�[�h�̓K�ؐ��̌v�Z(i.e. ���݂̏�ԂŃA�C�e���������\���H)
                            if (dicExpandedHand[itemGive]-1) < 0:
                                continue
                            elif (numofEachItems[itemGiven]-(dicExpandedHand[itemGiven]+1) < 0):
                                continue
                            else :#�K�؂ȃm�[�h�̒ǉ�
                                #--�����ɂ������m�[�h�̊i�[:
                                for item in dicExpandedHand.keys():
                                    dicExpandedHand[item]=int(dicExpandedHand[item])#int�ɃL���X�g
                                dicExpandedHand[itemGive]-=1
                                dicExpandedHand[itemGiven]+=1
                                #
                                expandedPlan=copy.copy(currentPlan)
                                codedHand=hand.split("_")[0]+"_"+str(dicExpandedHand[hand.split("_")[0]])+"_"
                                codedHand+=hand.split("_")[2]+"_"+str(dicExpandedHand[hand.split("_")[2]])+"_"
                                codedHand+=hand.split("_")[4]+"_"+str(dicExpandedHand[hand.split("_")[4]])
                                expandedPlan.append(codedHand)
                                queue.append(expandedPlan)
            #-�e�S�[�����Ƃɓ��B�\�ȃv�����W����ǉ�����
            availablePlanatEachHand[hand]=copy.copy(correctPlan)
        #---4Test@
        #if len(availablePlanatEachHand) > 0:
        #    print availablePlanatEachHand
        #else:
        #    print "No valuable plan"
        #�v������K���ɑI��
        bestPlan=[]
        totalNumPlan=0
        for hand in achievableProfitOutcomATA.keys():
            totalNumPlan+=len(availablePlanatEachHand[hand])
        if totalNumPlan >0: 
            planIndex=random.randint(low=0,high=999999)%totalNumPlan
    
            cumNumber=0
            for hand in achievableProfitOutcomATA.keys():
                ind=0
                for plan in availablePlanatEachHand[hand]:
                    if (ind+cumNumber) == planIndex:
                        bestPlan=plan
                    ind+=1
                cumNumber+=len(availablePlanatEachHand[hand])
            
        #print "best plan=" + str(plan)
        #print "expected outcome=" + str(bestExpectedOutcome)
        #���݂̐��ʂƃv�����j���O�ŗǂ̊��Ґ��ʂ��r���āAKeep��(Offer or accept�������߂�)
        if (len(bestPlan)<=1):#Keep����ꍇ
            self._DSupdateDoNothing(agentTakingAction)
        else:#Offer or Accept����ꍇ
            # TODO-----------------------------
            #-������A�C�e���Ƃ��炤�A�C�e�����v�Z����
            itemGive=None
            itemGiven=None
            for i in range(len(bestPlan[0].split("_"))/2):
                spBestNext=bestPlan[1].split("_")
                spBestCurrent=bestPlan[0].split("_")
                if (int(spBestNext[(i*2)+1]) - int(spBestCurrent[(i*2)+1])) < 0:
                    itemGive=spBestNext[(i*2)]
                if (int(spBestNext[(i*2)+1]) - int(spBestCurrent[(i*2)+1])) > 0:
                    itemGiven=spBestNext[(i*2)]
            
            #--4 test
            #print "Given"+itemGiven
            #print "Give"+itemGive
            #-offered������A�v�����ɍ��v���Ă���Ύ󂯓����
            #----------------------------------------------�����܂Ńf�o�b�O�ς�
            isAccepted=False
            for ds in self.dicsharedDialogState.keys():
                if (re.search("Offered",ds) != None) and (self.dicsharedDialogState[ds] == 1.0):
                    offerAgent=None
                    offerItemtoGive=None
                    offerItemtoGiven=None
                    for agent in DialogueProtocols.listAgentsName:
                        if re.search(agent, ds) != None:
                            offerAgent=agent
                    for item in DialogueProtocols.listItems:
                        if re.search("_"+item+"_", ds) != None:
                            offerItemtoGiven=item
                    for item in DialogueProtocols.listItems:
                        if re.search("_"+item+"$", ds) != None:
                            offerItemtoGive=item
                    assert (offerAgent != None) and (offerItemtoGive != None) and (offerItemtoGiven != None), "Illegal Offer"
                    if (offerItemtoGiven==itemGiven) and (offerItemtoGive==itemGive):
                        self._DSupdateExecuteTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven)
                        isAccepted=True
            if not isAccepted:#-�����łȂ��ꍇ��Offer����A�C�e����T��
                isOffered=False
                #-Given�������Ă���G�[�W�F���g����
                agentsHavingGivenItem=[]
                for agent in DialogueProtocols.listAgentsName:
                    if (agent != agentTakingAction):
                        if (self.dicsharedDialogState["NumItem_"+agent+"_"+itemGiven] > 0.0):
                            agentsHavingGivenItem.append(agent)
                if len(agentsHavingGivenItem) > 0:
                    self._DSupdateExecuteOffer(agentTakingAction, agentsHavingGivenItem[random.randint(0,9999)%len(agentsHavingGivenItem)], itemGive,itemGiven)
                    isOffered=True
                else:
                    print "Err"
            assert (isAccepted or isOffered), "System could perfome nerither accept offer nor offer."


    #Hand-crafted policy��(Proposed by David-san) 2014/12/03
    #-���݂̏�Ԃ���B���\�ȃS�[���Ɍ����Ẵv�����𐶐�����B
    #-�����āA��������鐬�ʂ��ł��ǂ����̂�I�ԁB������ǂ����ʂ����҂ł��Ȃ��ꍇ�́AKeep.
    def _PlanbasedPolicy(self, agentTakingAction):
        #�e�G�[�W�F���g�̏����A�C�e����Payoff�ɉ��������ʃe�[�u�����쐬�F
        if self.achievableOutcome == None:
            self.achievableOutcome={}#�A�C�e���`�Q�A�C�e���`�̌�_�A�C�e���a�Q�A�C�e���a�̌�_�A�C�e���b�Q�A�C�e���b�̌�
            self.totalNumofItemsofAgents={}
            #-���ʃe�[�u���̐���
            #--�e�G�[�W�F���g�̃A�C�e���̑��������߂�
            dicTotalNumofItemofAgents={}
            for agent in self.dicAgents.keys():
                totalNum= 0
                for item in self.dicAgents[agent]["InitNumItems"].keys():
                    totalNum+=self.dicAgents[agent]["InitNumItems"][item]
                dicTotalNumofItemofAgents[agent]=totalNum
            self.totalNumofItemsofAgents=copy.copy(dicTotalNumofItemofAgents)
            #--���ꂼ��̎������̒ʂ�ɉ��������ʂ��v�Z����
            #---�A�C�e�����ɂ��������\�Ȏ�D�̐ݒ�
            dicAchievableHandofAgents={}
            for agent in dicTotalNumofItemofAgents.keys():
                if dicTotalNumofItemofAgents[agent] == 2:
                    dicAchievableHandofAgents[agent]=copy.copy(DialogueProtocols.listAchievablePoorHands)
                elif dicTotalNumofItemofAgents[agent] == 3:
                    dicAchievableHandofAgents[agent]=copy.copy(DialogueProtocols.listAchievableNormalHands)
                elif dicTotalNumofItemofAgents[agent] == 4:
                    dicAchievableHandofAgents[agent]=copy.copy(DialogueProtocols.listAchievableRichHands)
                else:
                    assert False, "Illegal achievable goal"
            #---�e����Payoff�s��ɉ��������ʂ̐ݒ�
            for agent in self.dicAgents.keys():
                #�e�G�[�W�F���g�̎育�Ƃ̕�V
                outcomes={}
                for hand in dicAchievableHandofAgents[agent]:
                    rew=0.0
                    numEachItem=[]
                    for item in DialogueProtocols.listItems:
                        rew+=DialogueProtocols.dicPreference[self.dicAgents[agent]["Payoff"][item]]*hand[item]
                        numEachItem.append(hand[item])
                    #�t���[�c�T���_
                    rew+=500*min(numEachItem)
                    outcomes[hand.keys()[0]+"_"+str(hand[hand.keys()[0]])+"_"+hand.keys()[1]+"_"+str(hand[hand.keys()[1]])+"_"+hand.keys()[2]+"_"+str(hand[hand.keys()[2]])]=rew
                self.achievableOutcome[agent]=outcomes
                
        #���݂̃G�[�W�F���g�̎莝���̐��ʂ����������ʂ��e�[�u�������珜�O����
        achievableProfitOutcomATA={}
        #-���݂̃G�[�W�F���g�̐��ʂ��v�Z
        currentrew=0.0
        numEachItem=[]
        for item in DialogueProtocols.listItems:
            currentrew+=DialogueProtocols.dicPreference[self.dicAgents[agentTakingAction]["Payoff"][item]]*self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item]
            numEachItem.append(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item])
        currentrew+=500*min(numEachItem)
        #-���O�̎��s
        for outcomes in self.achievableOutcome[agentTakingAction].keys():
            if self.achievableOutcome[agentTakingAction][outcomes] > currentrew:
                achievableProfitOutcomATA[outcomes]=self.achievableOutcome[agentTakingAction][outcomes]
        
        #�v�����j���O�ii.e.�ŒZ�o�H�T���j
        #TODO<------------------------------------------------------------------
        #-�����̎���܂߂��e�A�C�e���̑����̃J�E���g (TODO �ŏ��ɌĂ΂ꂽ�Ƃ��ɂ̂݌v�Z����悤�ɕύX)
        numofEachItems={}
        for item in DialogueProtocols.listItems:
            totalNum=0
            for agent in DialogueProtocols.listAgentsName:
                totalNum+=self.dicAgents[agent]["InitNumItems"][item]
            numofEachItems[item]=totalNum
            
        #-�c��̃^�[���Ŗ��炩�ɒB���o���Ȃ��ꍇ�͍폜(�ڕW�ƃA�C�e����L1�������c��̎萔�ȏ�̏ꍇ�͍폜)
        availablePlanatEachHand={}
        for hand in achievableProfitOutcomATA.keys():
            #print hand.split("_")
            initHand=hand.split("_")[0]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[0]]))+"_"
            initHand+=hand.split("_")[2]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[2]]))+"_"
            initHand+=hand.split("_")[4]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[4]]))
            queue=[[initHand]]#�A�C�e�����̏�Ԍn��
            correctPlan=[]
            #���D��T��
            while len(queue) > 0:
                currentPlan=queue.pop(0)
                dist=0
                #---�����̌v�Z
                for i in range(len(hand.split("_"))/2):
                    #print int(hand.split("_")[(i*2)+1])
                    #print int(currentPlan[-1].split("_")[(i*2)+1])
                    #print str(i)
                    #print hand.split("_")
                    #print currentPlan[-1].split("_")
                    dist+=abs(int(float(hand.split("_")[(i*2)+1]))-int(float(currentPlan[-1].split("_")[(i*2)+1])))
                #--�m�[�h������ȏ�W�J����K�v�������ꍇ
                if dist == 0: #�I�[��Ԃ̏ꍇ
                    correctPlan.append(copy.copy(currentPlan))
                #TESTING NOW
                #elif dist > (self.totalNumofItemsofAgents[agentTakingAction]-len(currentPlan)+1): #����̎萔�œ��B�ł��Ȃ��ꍇ
                elif dist > (self.searchDepth[agentTakingAction]-len(currentPlan)+1): #����̎萔�œ��B�ł��Ȃ��ꍇ
                    pass
                else: 
                    #�m�[�h��W�J����K�v������ꍇ
                    #--���ȃm�[�h�̒ǋL
                    #--�m�[�h�̓W�J�F
                    for itemGive in DialogueProtocols.listItems:
                        for itemGiven in DialogueProtocols.listItems:
                            if itemGive == itemGiven:
                                continue
                            expandedHand=copy.copy(currentPlan[-1])
                            dicExpandedHand={}
                            for i in range((len(expandedHand.split("_"))/2)):
                                dicExpandedHand[expandedHand.split("_")[i*2]]=int(float(expandedHand.split("_")[(i*2)+1]))
                            #�m�[�h�̓K�ؐ��̌v�Z(i.e. ���݂̏�ԂŃA�C�e���������\���H)
                            if (dicExpandedHand[itemGive]-1) < 0:
                                continue
                            elif (numofEachItems[itemGiven]-(dicExpandedHand[itemGiven]+1) < 0):
                                continue
                            else :#�K�؂ȃm�[�h�̒ǉ�
                                #--�����ɂ������m�[�h�̊i�[:
                                for item in dicExpandedHand.keys():
                                    dicExpandedHand[item]=int(dicExpandedHand[item])#int�ɃL���X�g
                                dicExpandedHand[itemGive]-=1
                                dicExpandedHand[itemGiven]+=1
                                #
                                expandedPlan=copy.copy(currentPlan)
                                codedHand=hand.split("_")[0]+"_"+str(dicExpandedHand[hand.split("_")[0]])+"_"
                                codedHand+=hand.split("_")[2]+"_"+str(dicExpandedHand[hand.split("_")[2]])+"_"
                                codedHand+=hand.split("_")[4]+"_"+str(dicExpandedHand[hand.split("_")[4]])
                                expandedPlan.append(codedHand)
                                queue.append(expandedPlan)
            #-�e�S�[�����Ƃɓ��B�\�ȃv�����W����ǉ�����
            availablePlanatEachHand[hand]=copy.copy(correctPlan)
        #---4Test@
        #if len(availablePlanatEachHand) > 0:
        #    print availablePlanatEachHand
        #else:
        #    print "No valuable plan"
        #�ŗǂ̃v������I��
        #�v�����j���O�ŋ��߂��p�X�̊��Ғl���v�Z
        bestPlan=[]
        bestExpectedOutcome=-9999
        probSuccessofproceeding=None#�e�v�����𐳂����i�s�o����m���B
        if self.normofAgents[agentTakingAction]=="Optimist":
            probSuccessofproceeding=1.0
        elif self.normofAgents[agentTakingAction]=="Pessimist":
            probSuccessofproceeding=0.1
        elif self.normofAgents[agentTakingAction]=="Uninformative":
            probSuccessofproceeding=0.5
        elif self.normofAgents[agentTakingAction]=="Random":
            probSuccessofproceeding=random.random()
        else:
            assert False, "Illegal norm of planner"

        for hand in achievableProfitOutcomATA.keys():
            for plan in availablePlanatEachHand[hand]:
                expectedOutcome=0
                i=0
                for proc in plan:
                    expectedOutcome+=self.achievableOutcome[agentTakingAction][proc]*(probSuccessofproceeding**i)*((1.0-probSuccessofproceeding)**(len(plan)-i))
                if bestExpectedOutcome < expectedOutcome:
                    bestExpectedOutcome = expectedOutcome
                    bestPlan=plan
        #print "best plan=" + str(plan)
        #print "expected outcome=" + str(bestExpectedOutcome)
        #���݂̐��ʂƃv�����j���O�ŗǂ̊��Ґ��ʂ��r���āAKeep��(Offer or accept�������߂�)
        if (len(bestPlan)<=1) or (bestExpectedOutcome < currentrew):#Keep����ꍇ
            self._DSupdateDoNothing(agentTakingAction)
        else:#Offer or Accept����ꍇ
            # TODO-----------------------------
            #-������A�C�e���Ƃ��炤�A�C�e�����v�Z����
            itemGive=None
            itemGiven=None
            for i in range(len(bestPlan[0].split("_"))/2):
                spBestNext=bestPlan[1].split("_")
                spBestCurrent=bestPlan[0].split("_")
                if (int(spBestNext[(i*2)+1]) - int(spBestCurrent[(i*2)+1])) < 0:
                    itemGive=spBestNext[(i*2)]
                if (int(spBestNext[(i*2)+1]) - int(spBestCurrent[(i*2)+1])) > 0:
                    itemGiven=spBestNext[(i*2)]
            
            #--4 test
            #print "Given"+itemGiven
            #print "Give"+itemGive
            #-offered������A�v�����ɍ��v���Ă���Ύ󂯓����
            #----------------------------------------------�����܂Ńf�o�b�O�ς�
            isAccepted=False
            for ds in self.dicsharedDialogState.keys():
                if (re.search("Offered",ds) != None) and (self.dicsharedDialogState[ds] == 1.0):
                    offerAgent=None
                    offerItemtoGive=None
                    offerItemtoGiven=None
                    for agent in DialogueProtocols.listAgentsName:
                        if re.search(agent, ds) != None:
                            offerAgent=agent
                    for item in DialogueProtocols.listItems:
                        if re.search("_"+item+"_", ds) != None:
                            offerItemtoGiven=item
                    for item in DialogueProtocols.listItems:
                        if re.search("_"+item+"$", ds) != None:
                            offerItemtoGive=item
                    assert (offerAgent != None) and (offerItemtoGive != None) and (offerItemtoGiven != None), "Illegal Offer"
                    if (offerItemtoGiven==itemGiven) and (offerItemtoGive==itemGive):
                        self._DSupdateExecuteTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven)
                        isAccepted=True
            if not isAccepted:#-�����łȂ��ꍇ��Offer����A�C�e����T��
                isOffered=False
                #-Given�������Ă���G�[�W�F���g����
                agentsHavingGivenItem=[]
                for agent in DialogueProtocols.listAgentsName:
                    if (agent != agentTakingAction):
                        if (self.dicsharedDialogState["NumItem_"+agent+"_"+itemGiven] > 0.0):
                            agentsHavingGivenItem.append(agent)
                if len(agentsHavingGivenItem) > 0:
                    self._DSupdateExecuteOffer(agentTakingAction, agentsHavingGivenItem[random.randint(0,9999)%len(agentsHavingGivenItem)], itemGive,itemGiven)
                    isOffered=True
                else:
                    print "Err"
            assert (isAccepted or isOffered), "System could perfome nerither accept offer nor offer."
    
    #Hand-crafted policy�ꍆ�@�Q�O�P�S�E�P�Q�E�O�P
    #-Greedy wise����:����Offer���󂯂āA����Offer���L�v�Ȃ��́i�X�R�A��������j�̂ł���΁AAccept����    
    #-Ofeer�������ꍇ�ŁA�����g�������A�C�e���������Ă���ꍇ�ALike�̃A�C�e�������l�Ƀ����_���ɂn������������B�����A���Ȃ���΁ANeutral�̃A�C�e���������Ă���l��T���Ăn������������
    #�����g�������A�C�e���������Ă��Ȃ��āA�m�������������A�C�e���������Ă���ꍇ�ALike�̃A�C�e�������l�Ƀ����_���ɂn������������B
    #����ȊO�̏ꍇ�́AKeep����
    def _GreedywisePolicy(self, agentTakingAction):
        isAccepted=False
        for ds in self.dicsharedDialogState.keys():
            if (re.search("Offered",ds) != None) and (self.dicsharedDialogState[ds] == 1.0):
                offerAgent=None
                offerItemtoGive=None
                offerItemtoGiven=None
                for agent in DialogueProtocols.listAgentsName:
                    if re.search(agent, ds) != None:
                        offerAgent=agent
                for item in DialogueProtocols.listItems:
                    if re.search("_"+item+"_", ds) != None:
                        offerItemtoGiven=item
                for item in DialogueProtocols.listItems:
                    if re.search("_"+item+"$", ds) != None:
                        offerItemtoGive=item
                assert (offerAgent != None) and (offerItemtoGive != None) and (offerItemtoGiven != None), "Illegal Offer"
                #--Offe���ł�����A�C�e����Hate�̏ꍇ�������ł�����(������)
                if self.dicAgents[agentTakingAction]["Payoff"][offerItemtoGive] =="Hate":
                    if self._isAvailableTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven):
                        self._DSupdateExecuteTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven)
                        isAccepted=True
                #--Offer�ł�����A�C�e����Neutral�̏ꍇ�́A�����A�C�e���̏ꍇ��Like�̏ꍇ�̂ݑΏ�
                elif (self.dicAgents[agentTakingAction]["Payoff"][offerItemtoGive] =="Neutral") and (self.dicAgents[agentTakingAction]["Payoff"][offerItemtoGiven] =="Like"):
                    if self._isAvailableTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven):
                        self._DSupdateExecuteTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven)
                        isAccepted=True
        
        #�A�N�Z�v�g���Ȃ��ꍇ�́A�X�R�A�����������Ȑl�������_���ɑI��Ō���
        isOffered=False
        likeItem=None
        for item in self.dicAgents[agentTakingAction]["Payoff"].keys():
            if self.dicAgents[agentTakingAction]["Payoff"][item] == "Like":
                likeItem=item
        assert likeItem != None, "Illegal access to payoff mat"
        if not isAccepted and not (self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+likeItem] == self.dicAgents[agentTakingAction]["InitNumItems"]):
            #Hate����A�C�e���������Ă���ꍇ
            hateItem=None
            for item in self.dicAgents[agentTakingAction]["Payoff"].keys():
                if self.dicAgents[agentTakingAction]["Payoff"][item] == "Hate":
                    hateItem = item
            if self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hateItem] > 0.0:
                #-�G�[�W�F���gLike�������Ă���G�[�W�F���g����
                agentsHavingLikeItem=[]
                for agent in DialogueProtocols.listAgentsName:
                    if (agent != agentTakingAction):
                        for item in DialogueProtocols.listItems:
                            if (self.dicAgents[agentTakingAction]["Payoff"][item]=="Like") and (self.dicsharedDialogState["NumItem_"+agent+"_"+item] > 0.0):
                                agentsHavingLikeItem.append(agent)
                if len(agentsHavingLikeItem) > 0:
                    self._DSupdateExecuteOffer(agentTakingAction, agentsHavingLikeItem[random.randint(0,9999)%len(agentsHavingLikeItem)], hateItem,likeItem)
                    isOffered=True
                else:#��������C�N�������Ă��Ȃ��ꍇ�̓j���[�g�������݂�
                    agentsHavingNeutralItem=[]
                    neutralItem=""
                    for agent in DialogueProtocols.listAgentsName:
                        if (agent != agentTakingAction):
                            for item in DialogueProtocols.listItems:
                                if (self.dicAgents[agentTakingAction]["Payoff"][item]=="Neutral") and (self.dicsharedDialogState["NumItem_"+agent+"_"+item] > 0.0):
                                    agentsHavingNeutralItem.append(agent)
                                    neutralItem=item
                    if len(agentsHavingNeutralItem) > 0:
                        self._DSupdateExecuteOffer(agentTakingAction, agentsHavingNeutralItem[random.randint(0,9999)%len(agentsHavingNeutralItem)], hateItem,neutralItem)
                        isOffered=True
            #HateItem���Ȃ��āANeutra�A�C�e���������Ă���ꍇ
            if not isOffered:
                neutralItem=None
                for item in self.dicAgents[agentTakingAction]["Payoff"].keys():
                    if self.dicAgents[agentTakingAction]["Payoff"][item] == "Neutral":
                        neutralItem = item
                if self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+neutralItem] > 0.0:
                    #-�G�[�W�F���gLike�������Ă���G�[�W�F���g����
                    agentsHavingLikeItem=[]
                    for agent in DialogueProtocols.listAgentsName:
                        if (agent != agentTakingAction):
                            for item in DialogueProtocols.listItems:
                                if (self.dicAgents[agentTakingAction]["Payoff"][item]=="Like") and (self.dicsharedDialogState["NumItem_"+agent+"_"+item] > 0.0):
                                    agentsHavingLikeItem.append(agent)
                    if len(agentsHavingLikeItem) > 0:
                        self._DSupdateExecuteOffer(agentTakingAction, agentsHavingLikeItem[random.randint(0,9999)%len(agentsHavingLikeItem)], neutralItem,likeItem)
                        isOffered=True                            
        #����ȊO�̏ꍇ�͂j������
        if (not isAccepted) and (not isOffered):
            self._DSupdateDoNothing(agentTakingAction)

       
    #sourceAgent��item1�������o���BTargetAgent��item2�������o�� 
    def _DSupdateExecuteTrade(self,sourceAgent, targetAgent, item1, item2):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            if sourceAgent == "AgentLearner":
                SatisticsOfLearnerInExperiment.totalNumberOfAccept+=1.0

        
        if self._isAvailableTrade(sourceAgent, targetAgent, item1, item2):
            #ageru
            self.dicsharedDialogState["NumItem_"+sourceAgent+"_"+item1] -=1.0
            self.dicsharedDialogState["NumItem_"+targetAgent+"_"+item2] -=1.0
            #fueru
            self.dicsharedDialogState["NumItem_"+sourceAgent+"_"+item2] +=1.0
            self.dicsharedDialogState["NumItem_"+targetAgent+"_"+item1] +=1.0
            self.mostPreviousAction="Accept"
            self.mostPreviousAgent=sourceAgent
            #
            if ExperimentalConditions.isCalculateStatisticsOfLearner:
                if sourceAgent == "AgentLearner":
                    SatisticsOfLearnerInExperiment.totalNumberOflligalAccept+=1.0
                if targetAgent == "AgentLearner":
                    SatisticsOfLearnerInExperiment.totalNumberOfOfferAcceptedByAddressee+=1.0

        else:
            self.mostPreviousAction="Accept (but Trading are failed)"
            self.mostPreviousAgent=sourceAgent
            #
            if ExperimentalConditions.isCalculateStatisticsOfLearner:
                if sourceAgent == "AgentLearner":
                    SatisticsOfLearnerInExperiment.totalNumberOfIlligalAccept+=1.0
            
        #OfferTable�̏�����
        for ds in self.dicsharedDialogState.keys():
            if re.search("Offered",ds) != None:
                self.dicsharedDialogState[ds]=0.0
        #
        self.numContKeep=0
        
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            #print "Full offered/accepted table"
            self.dicsharedDialogState["Accept_"+sourceAgent+"_"+targetAgent+"_"+item1+"_"+item2]=1.0
            self.dicsharedDialogState["HistoryA_"+sourceAgent+"_"+targetAgent+"_"+item1+"_"+item2]+=1.0
        
        #�g���[�X
        if ExperimentalConditions.isTraceAgent:
            print "Turn "+str(self.turn)
            print sourceAgent + " accept " + targetAgent + "'s offer that " + sourceAgent + " give " + item1 + ", and " + targetAgent + " give " + item2
            for agent in DialogueProtocols.listAgentsName:
                print agent + " have: "
                for item in DialogueProtocols.listItems:
                    print item + ":" + str(self.dicsharedDialogState["NumItem_"+agent+"_"+item])
    
    #�Θb��Ԃ̍X�V�N����Offer�����Ƃ�
    #sourceAgent��item1�������o���BTargetAgent��item2�������o�� 
    def _DSupdateExecuteOffer(self,sourceAgent, targetAgent, item1, item2):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            if sourceAgent == "AgentLearner":
                SatisticsOfLearnerInExperiment.totalNumberOfOffer+=1.0

        #RejectTable�̃A�b�v�f�[�g
        if (re.search("Offer_",self.mostPreviousAction) != None):
            spAct=self.mostPreviousAction.split("_")
            self.dicsharedDialogState["Rejected_"+self.mostPreviousAgent+"_"+sourceAgent+"_"+spAct[2]+"_"+spAct[3]]=1.0

        for ds in self.dicsharedDialogState.keys():
            if re.search("OfferedFrom_",ds) != None:
                self.dicsharedDialogState[ds]=0.0
        self.dicsharedDialogState["OfferedFrom_"+sourceAgent+"_"+item1+"_"+item2]=1.0
        
        self.mostPreviousAction="Offer_"+targetAgent+"_"+item1+"_"+item2
        self.mostPreviousAgent=sourceAgent
        assert DialogueProtocols.listSystemAction.count(self.mostPreviousAction)>0, "Illegal Offer"
        self.numContKeep=0
        
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            #print "Full offered/accepted table"
            self.dicsharedDialogState["FO_"+sourceAgent+"_"+targetAgent+"_"+item1+"_"+item2]=1.0
            self.dicsharedDialogState["HistoryO_"+sourceAgent+"_"+targetAgent+"_"+item1+"_"+item2]+=1.0
        #�g���[�X
        if ExperimentalConditions.isTraceAgent:
            print "Turn "+str(self.turn)
            print sourceAgent + " offer to " + targetAgent + " that " + sourceAgent + " give " + item1 + ", and " + targetAgent + " give " + item2
    
    #�������Ȃ� i.e. Keep
    def _DSupdateDoNothing(self,sourceAgent, targetAgent=None, item1=None, item2=None):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            if sourceAgent == "AgentLearner":
                SatisticsOfLearnerInExperiment.totalNumberOfKeep+=1.0
            
        #RejectTable�̃A�b�v�f�[�g
        if (re.search("Offer_",self.mostPreviousAction) != None):
            spAct=self.mostPreviousAction.split("_")
            self.dicsharedDialogState["Rejected_"+self.mostPreviousAgent+"_"+sourceAgent+"_"+spAct[2]+"_"+spAct[3]]=1.0
        
        self.mostPreviousAction="Keep"
        self.mostPreviousAgent=sourceAgent
        self.numContKeep+=1
        #OfferTable�̏�����
        for ds in self.dicsharedDialogState.keys():
            if re.search("Offered",ds) != None:
                self.dicsharedDialogState[ds]=0.0
        #�g���[�X
        if ExperimentalConditions.isTraceAgent:
            print "Turn "+str(self.turn)
            print sourceAgent + " keeping"
            
    
    #�g���[�h���\���`�F�b�N���� return boolean
    #������ 20141215
    def _isAvailableTrade(self,sourceAgent, targetAgent, item1, item2):
        #sourceaAgent�̎������`�F�b�N
        isAvailableSource=False
        if self.dicsharedDialogState["NumItem_"+sourceAgent+"_"+item1] > 0.0: 
            isAvailableSource=True
        isAvailableTarget=False
        if self.dicsharedDialogState["NumItem_"+targetAgent+"_"+item2] > 0.0: 
            isAvailableTarget=True
        
        if (isAvailableSource and isAvailableTarget):
            return True
        else: 
            return False
    
    def getSensors(self):
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            tempDS=[]
            for sds in DialogueProtocols.listDialogStateinSpecialCase:
                tempDS.append(self.dicsharedDialogState[sds])
            
            #�Θb��Ԃ̃g���[�X
            if ExperimentalConditions.isTraceAgent:
                DSforTrace={}
                for sds in DialogueProtocols.listDialogStateinSpecialCase:
                    DSforTrace[sds]=self.dicsharedDialogState[sds]
                print DSforTrace
            
            return array(tempDS)
        
        
        return array(self.dicsharedDialogState.values())
    
class TestTask(EpisodicTask):
    def __init__(self):
        self.testEnv=TestEnv()
        EpisodicTask.__init__(self,self.testEnv)
        self.reset()
    def performAction(self,action):
        EpisodicTask.performAction(self, action)
        
    def getReward(self):#This reward function is called twice in one action taking (Bug of Pybrain? or only one call is used for calculate cumulative reward)
        rew=0.0
        
#         #TEST ������V�i�T�u�S�[���j
#         numEachItem=[]
#         for item in DialogueProtocols.listItems:
#             rew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item]
#             numEachItem.append(self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item])
#         #�t���[�c�T���_
#         rew+=500*min(numEachItem)

        #���P���ɉ�����������V
        if ExperimentalConditions.isFeedRewardAsImprovement:
            #
            #print "reward"
            
            #1�^�[���O�ł̕�V�v�Z
            numEachItem=[]
            tempPrevRew=0.0
            for item in DialogueProtocols.listItems:
                tempPrevRew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.previousLeanerDialogState["NumItem_AgentLearner_"+item]
                numEachItem.append(self.testEnv.previousLeanerDialogState["NumItem_AgentLearner_"+item])
            #TEST
            #print "Prev: ",
            #print numEachItem
            tempPrevRew+=500*min(numEachItem)
            
            #���݂̃^�[���ł̕�V�v�Z
            numEachItem=[]
            tempRew=0.0
            for item in DialogueProtocols.listItems:
                tempRew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item]
                numEachItem.append(self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item])
            #TEST
            #print "Current:", 
            #print numEachItem
            tempRew+=500*min(numEachItem)
            
            rew+=tempRew-tempPrevRew
            #TEST
            #print rew
            
        else:
            #�S�^�[���I����ɂ��������ɂ�����            
            if self.isFinished():
                numEachItem=[]
                for item in DialogueProtocols.listItems:
                    rew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item]
                    numEachItem.append(self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item])
                #�t���[�c�T���_
                rew+=500*min(numEachItem)


        if ExperimentalConditions.isTraceAgent:
            print "Reward=" + str(rew)
        
        return rew
    
    def isFinished(self):
        if (self.testEnv.turn >= ExperimentalConditions.iMaximumTurn) or (self.testEnv.numContKeep >=len(self.testEnv.dicAgents.keys())):
            #
            if ExperimentalConditions.isCalculateStatisticsOfLearner:
                SatisticsOfLearnerInExperiment.totalNumberOfTotalTurn+=(double)(self.testEnv.turn)
            
            return True
        elif self.testEnv.isLearnerReachMaximumOutcome:
            #
            if ExperimentalConditions.isCalculateStatisticsOfLearner:
                SatisticsOfLearnerInExperiment.totalNumberOfTotalTurn+=(double)(self.testEnv.turn)

            return True
        else:
            return False

#--------------------------------main �ȉ�,�����w�K�p�v���O����----------------------------------------------------------
warnings.filterwarnings(action="ignore")    
for combofOpponentsStrategy in ExperimentalConditions.listCombinationofOpponentsStrategy:
    ExperimentalConditions.currentCombinationofOpponentsStrategy=combofOpponentsStrategy
    currennumOpponents=len(combofOpponentsStrategy.split("x"))
    print "Current number of opponents=" + str(len(combofOpponentsStrategy.split("x")))
    print "Combination of Opponents=" + ExperimentalConditions.currentCombinationofOpponentsStrategy
    #���������ɉ������G�[�W�F���g�̖��O�E�A�N�V��������.............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................../................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................./////&��Ԑ���
    #-�G�[�W�F���g�̖��O
    DialogueProtocols.listAgentsName=["AgentLearner"]
    for i in range(currennumOpponents): 
        DialogueProtocols.listAgentsName.append("Agent"+str(i))
    #-�G�[�W�F���g�̃A�N�V����
    #--Offer�ȊO
    DialogueProtocols.listSystemAction=["Accept","Keep"]
    #--Offer
    for agent in DialogueProtocols.listAgentsName:
        for item1 in DialogueProtocols.listItems: 
            for item2 in DialogueProtocols.listItems:
                if item1 != item2:
                    #agent�Ɏ�����item1�������o���āAagent��item2���󂯎��\���o�B
                    DialogueProtocols.listSystemAction.append("Offer_"+agent+"_"+item1+"_"+item2)

    print "Protocol for System Action is generated." 
    print str(len(DialogueProtocols.listSystemAction))+" actions are generated."
    print DialogueProtocols.listSystemAction
    #-�G�[�W�F���g�̏��
    DialogueProtocols.listSystemState=[]
    DialogueProtocols.listDialogStateinSpecialCase=[]
    #--OfferedTable
    for agent in DialogueProtocols.listAgentsName:
        for item1 in DialogueProtocols.listItems: 
            for item2 in DialogueProtocols.listItems:
                if item1 != item2:
                    #�I�t�@�[���Ƃ̃G�[�W�F���g�iagent�j��item1�ƃI�t�@�[���ꂽ�G�[�W�F���g��item2�̌����B
                    DialogueProtocols.listSystemState.append("OfferedFrom_"+agent+"_"+item1+"_"+item2)
                    #
                    if agent != "AgentLearner":
                        DialogueProtocols.listDialogStateinSpecialCase.append("OfferedFrom_"+agent+"_"+item1+"_"+item2)
    
    #--�e�G�[�W�F���g�̃A�C�e���̌�
    for agent in DialogueProtocols.listAgentsName: 
        for item1 in DialogueProtocols.listItems: 
            DialogueProtocols.listSystemState.append("NumItem_"+agent+"_"+item1)
            #
            DialogueProtocols.listDialogStateinSpecialCase.append("NumItem_"+agent+"_"+item1)

    print "Protocol for System State is generated." 
    print str(len(DialogueProtocols.listSystemState))+" states are generated."

    #Bias���@After analysis
    print "Bias always 1" 
    DialogueProtocols.listSystemState.append("Bias")
    #
    DialogueProtocols.listDialogStateinSpecialCase.append("Bias")
    
    
    #�ǉ��̑Θb���29141205
    if ExperimentalConditions.isAddAdditionalDStoLearner:
        print "Additional dialog state is appended"
        print "1.CurrentTurn"
        DialogueProtocols.listSystemState.append("CurrentTurn")
        #
        #DialogueProtocols.listDialogStateinSpecialCase.append("CurrentTurn")
        
        print "Full offered/accepted table"
        print "Full offered/accepted table"
        for agent1 in DialogueProtocols.listAgentsName:#Offer/Accept�����G�[�W�F���g
            for agent2 in DialogueProtocols.listAgentsName:#Offer/Accept���ꂽ�G�[�W�F���g
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if item1 != item2:
                            DialogueProtocols.listSystemState.append("FO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            DialogueProtocols.listSystemState.append("Accept_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #
                            #DialogueProtocols.listDialogStateinSpecialCase.append("FO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #DialogueProtocols.listDialogStateinSpecialCase.append("Accept_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
        #�A�N�Z�v�g�̃q�X�g��
        print "Offer/accept History"
        for agent1 in DialogueProtocols.listAgentsName:#Offer/Accept�����G�[�W�F���g
            for agent2 in DialogueProtocols.listAgentsName:#Offer/Accept���ꂽ�G�[�W�F���g
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if item1 != item2:
                            DialogueProtocols.listSystemState.append("HistoryO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            DialogueProtocols.listSystemState.append("HistoryA_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #
                            #DialogueProtocols.listDialogStateinSpecialCase.append("HistoryO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #DialogueProtocols.listDialogStateinSpecialCase.append("HistoryA_"+agent1+"_"+agent2+"_"+item1+"_"+item2)        
        #�b���W�F�N�g�̃q�X�g��
        print "Reject History"
        for agent1 in DialogueProtocols.listAgentsName:#Reject���ꂽ�G�[�W�F���g
            for agent2 in DialogueProtocols.listAgentsName:#�����G�[�W�F���g
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if (item1 != item2) and (agent1 != agent2):
                            DialogueProtocols.listSystemState.append("Rejected_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #
                            if agent1 == "AgentLearner":
                                pass
                                #DialogueProtocols.listDialogStateinSpecialCase.append("Rejected_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            
    print "OfferedFrom_TargetAgentName_itemGive_itemGiven:���O�ɂǂ�TargetAgent���畨�X(itemGive��������itemGiven�����炤)������Offer���󂯂����B"
    print "NumItem_TargetAgentName_item:�e�G�[�W�F���g��Item�̌�"
    print "All shared DS"
    print DialogueProtocols.listSystemState
    print "DS for learner"
    print DialogueProtocols.listDialogStateinSpecialCase
    
    #���������̃A�i�E���X
    if ExperimentalConditions.isUseHandCraftedPolicy:
        print "Learner is using hand-crafted policy (Randomly select from all validate action)"
    if ExperimentalConditions.isUseRandomPolicy:
        print "Learner is using random policy"
    if ExperimentalConditions.isUseLinearQ:
        print "Learner is using linear-Q"
        if ExperimentalConditions.isUsePartiallyRulePolicy:
            print "Rule based policy is partially used"
    else:
        print "Learner is using NFQ"
        if ExperimentalConditions.isUsePartiallyRulePolicy:
            print "Rule based policy is partially used"
    if ExperimentalConditions.isAlwaysKeeping:
        print "The policy always try to keeping"
    if ExperimentalConditions.isFeedRewardAsImprovement:
        print "Reward is given according to improvement of hands at each turn"
    if ExperimentalConditions.isCalculateStatisticsOfLearner:
        print "Statistics of learner is calculated."
    
    #AgentLearner�p�̖��ʂ̖����A�N�V�����̒ǉ�
    DialogueProtocols.listSystemActionForLearner=copy.copy(DialogueProtocols.listSystemAction)
    for elem in DialogueProtocols.listSystemAction:
        if re.search("Offer_AgentLearner_",elem) != None: 
            DialogueProtocols.listSystemActionForLearner.remove(elem)
    print "Action for learner"
    print DialogueProtocols.listSystemActionForLearner
    
    #�^�[��
    print "Maximum turn=" + str(ExperimentalConditions.iMaximumTurn)
    
    #-create the task, experiment and agent
    task = TestTask()
    agent=None
    numInitialWeight=None
    if not ExperimentalConditions.isUseLinearQ:
        net=ActionValueNetwork(len(DialogueProtocols.listDialogStateinSpecialCase),len(DialogueProtocols.listSystemActionForLearner))
        agent = LearningAgent(net,NFQ())
        #agent.learner.gamma=1.0
        #
        #net = buildNetwork(len(DialogueProtocols.listDialogStateinSpecialCase), len(DialogueProtocols.listSystemActionForLearner), outclass=SigmoidLayer)
        #agent = LearningAgent(net, ENAC())
        agent.learner._explorer.epsilon=0.1
        #test
        numInitialWeight=len(agent.learner.module.network.params)

    else:
        agent = LinearFA_Agent(Q_LinFA(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
        #agent = LinearFA_Agent(QLambda_LinFA(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
        #agent = LinearFA_Agent(GQLambda(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
        #agent = LinearFA_Agent(SARSALambda_LinFA(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
        #agent = LinearFA_Agent(LSTDQLambda(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
        #agent = LinearFA_Agent(LSPI(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))        
        agent.learner.rewardDiscount=0.9#0.99����0.9�̕����������悳��
        agent.learner._lambda=0.99
        
    experiment = EpisodicExperiment(task, agent)
    
    #����-�w�K
    if ExperimentalConditions.isLearning:
        for pol in range(ExperimentalConditions.numSystems):
            #����d�݂�������
            if not ExperimentalConditions.isUseLinearQ:
                net=ActionValueNetwork(len(DialogueProtocols.listDialogStateinSpecialCase),len(DialogueProtocols.listSystemActionForLearner))
                agent = LearningAgent(net,NFQ())
                #agent.learner.gamma=1.0
                #
                #net = buildNetwork(len(DialogueProtocols.listDialogStateinSpecialCase), len(DialogueProtocols.listSystemActionForLearner), outclass=SigmoidLayer)
                #agent = LearningAgent(net, ENAC())
                agent.learner._explorer.epsilon=0.1#*=1.0#0.05
            else:
                agent = LinearFA_Agent(Q_LinFA(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
                #agent = LinearFA_Agent(QLambda_LinFA(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
                #agent = LinearFA_Agent(GQLambda(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
                #agent = LinearFA_Agent(SARSALambda_LinFA(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
                #agent = LinearFA_Agent(LSTDQLambda(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
                #agent = LinearFA_Agent(LSPI(len(DialogueProtocols.listSystemActionForLearner), len(DialogueProtocols.listDialogStateinSpecialCase)))
                agent.learner.rewardDiscount=0.9
                agent.learner._lambda=0.99
                
                
            experiment = EpisodicExperiment(task, agent)
            
            d=datetime.datetime.today()
            #�L�^�p�t�@�C���쐬
            f=None
            pkl=None
            if not ExperimentalConditions.isUseLinearQ:
                f=open(ExperimentalConditions.currentCombinationofOpponentsStrategy+str(currennumOpponents)+"Experiment_Learning_NFQ"+str(d.year)+str(d.month+d.day)+"_"+str(d.hour)+str(d.minute)+"_NumberOfPolicy"+str(pol)+".txt","w")
            else:
                f=open(ExperimentalConditions.currentCombinationofOpponentsStrategy+str(currennumOpponents)+"Experiment_Learning_LinQ"+str(d.year)+str(d.month+d.day)+"_"+str(d.hour)+str(d.minute)+"_NumberOfPolicy"+str(pol)+".txt","w")
                pkl=open(ExperimentalConditions.currentCombinationofOpponentsStrategy+str(currennumOpponents)+"Experiment_Learning_LinQ"+str(d.year)+str(d.month+d.day)+"_"+str(d.hour)+str(d.minute)+"_NumberOfPolicy"+str(pol)+".pkl","w")
            index=1
            
            bestCumReward=-9999999999.0
            bestPolicyW = None
            for elem in range(ExperimentalConditions.numBatch):
                if ExperimentalConditions.isUseLinearQ:
                    if elem < (ExperimentalConditions.numBatch*(4.0/4.0)):#�S�w�K��T���Ɏg��
                        agent.learner.exploring=True
                    else:
                        agent.learner.exploring=False
                try:
                    reward=experiment.doEpisodes(ExperimentalConditions.numEpisode)
                    ave=0.0
                    num=0.0
                    for bat in reward:
                        for epi in bat:
                            ave+=epi
                        num+=1
                    #num=size(reward)#�Θb�̍Ō�ɂ̂�reward��^���鎞�ȊO�̓R�����g�A�E�g
                    rew=ave/num
                    print "���ϕ�V@step"+str(index)+"=" + str(ave/num)
                    #
                    if ExperimentalConditions.isCalculateStatisticsOfLearner:
                        SatisticsOfLearnerInExperiment.totalNumberOfDialogue+=num
                    
                    #�w�K���ʂ̏o��
                    f.write("���ϕ�V@step"+str(index)+"="+str(ave/num)+"\n")
                    if bestCumReward < ((double)(ave)/(double)(num)) and (index >= ExperimentalConditions.numBatchStartLogging):
                        bestCumReward =((double)(ave)/(double)(num))
                        if not ExperimentalConditions.isUseLinearQ:
                            bestPolicyW=copy.deepcopy(agent.learner.module.network.params)
                        else:
                            bestPolicyW=copy.deepcopy(agent.learner._theta)
#                     #TEST
#                     print str(DialogueProtocols.totalAccept/DialogueProtocols.totalOffer)

                    #�����������ꍇ�͊w�K�ł��؂� 20141208
                    if (index >= 2) and (rew < ExperimentalConditions.dBadConversion):
                        print "Learning was stopped because of bad conversion"
                        break
                    if not ExperimentalConditions.isUseLinearQ:
                        agent.learn()
                        agent.reset()#�e�X�e�b�v���Ƃɏ������͂��邱�Ɓi�w�K��͕��􂪕ς�邽�߁A����܂ł̃f�[�^�͎g���Ȃ��j
                        #�ǉ��@�w�K���Ƃ̒T���̌��ہ@�Q�O�P�S1208
                        agent.learner._explorer.epsilon=0.1#*=1.0#0.05
                except:
                    print traceback.format_exc()
                index+=1
            if not ExperimentalConditions.isUseLinearQ:
                f.write("BestAverageCumReward="+str(bestCumReward)+"\n")
                f.write("Bestweight=\n")
                for e in bestPolicyW:
                    f.write(str(e)+",")
                f.write("\n")
                f.close()
            else:
                f.write("BestAverageCumReward="+str(bestCumReward)+"\n")
                f.write("Bestweight=\n")
                for e in bestPolicyW:
                    f.write(str(e)+",")
                f.write("\n")
                f.close()
                pickle.dump(bestPolicyW, pkl)
                pkl.close()
    
    
    #����-�œK�����]��
    if ExperimentalConditions.isTest:
        #�G�[�W�F���g�̃A�N�V������argmax�I���ɂ���
        if not ExperimentalConditions.isUseLinearQ:
            agent.learner._explorer.epsilon=0.0
        else:
            agent.learner._behaviorPolicy=agent.learner._greedyAction
            agent.learner.batchMode=True
            agent.learning=False
        #weight.txt����d�݂�ǂݍ���(�t�@�C���`����csv) 
        weight=[]
        weightStr=""
        fWeight=None
        if not ExperimentalConditions.isUseLinearQ:
            fWeight=open("BestPolicyNFQ.txt","r")
        else:
            fWeight=pickle.load(open("BestPolicyLinQ.pkl","r"))        
        assert fWeight != None, "Illegal experimental condition"
        
        if not ExperimentalConditions.isUseLinearQ:
            for wArray in fWeight:
                weightStr=wArray.split(",")
            for w in weightStr:
                if w != "":
                    weight.append(double(w))
            assert numInitialWeight== len(weight), "Total number of weight in initialization is incorrect. it should be same as that of weights loaded from File. check __init__ in ActionValueNetwoek in interface.py "
            agent.learner.module.network._setParameters(weight)
        else:
            agent.learner._theta=fWeight
            
        d=datetime.datetime.today()
        f=open(ExperimentalConditions.currentCombinationofOpponentsStrategy+str(currennumOpponents)+"Experiment_Evaluation"+str(d.year)+str(d.month+d.day)+"_"+str(d.hour)+str(d.minute)+".txt","w")
        index=1
        for elem in range(ExperimentalConditions.numBatch):
            try:
                reward=experiment.doEpisodes(ExperimentalConditions.numEpisode)
                ave=0.0
                num=0.0
                for bat in reward:
                    for epi in bat:
                        ave+=epi
                    num+=1
                print "���ϕ�V@step" +str(index)+"=" + str(ave/num)
                f.write("���ϕ�V@step"+str(index)+"="+str(ave/num)+"\n")
                #
                if ExperimentalConditions.isCalculateStatisticsOfLearner:
                    SatisticsOfLearnerInExperiment.totalNumberOfDialogue+=num

            except :
                print traceback.format_exc()
            index+=1
        f.close()
    
    
    #Output statistics of learner
    if ExperimentalConditions.isCalculateStatisticsOfLearner:
        print "statistics of learner (Total number):" 
        print "Offer, Accepted offer, Accept, lligal accept, Keep, Total Action, Total Turn, Total Dialogue, "
        print str(SatisticsOfLearnerInExperiment.totalNumberOfOffer)+", ",
        print str(SatisticsOfLearnerInExperiment.totalNumberOfOfferAcceptedByAddressee)+", ",
        print str(SatisticsOfLearnerInExperiment.totalNumberOfAccept)+", ",
        print str(SatisticsOfLearnerInExperiment.totalNumberOflligalAccept)+", ",
        print str(SatisticsOfLearnerInExperiment.totalNumberOfKeep)+", ",
        print str(SatisticsOfLearnerInExperiment.totalNumberOfAction)+", ",
        print str(SatisticsOfLearnerInExperiment.totalNumberOfTotalTurn/3.0)+", ",#as isFinished called 3 times at each turn
        print str(SatisticsOfLearnerInExperiment.totalNumberOfDialogue)+", ",
    
    
    
    