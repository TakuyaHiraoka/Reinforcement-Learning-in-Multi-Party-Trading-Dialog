# coding:shift-jis
#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with several optimization algorithms
# on the CartPoleEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
#########################################################################

#バグ修正2014
#AgentLearnerの二回行動
#Offer,AcceptしてもCountKeepが0にならない
#Offerを受けた相手が二回行動しうる
#ExecuteOffer時のOferテーブルの計算方法のバグ
#
#バイアス工を追加2014/12/13

#自分のitem1を差し出して、オファーされたエージェント(agent)のitem2を受け取る申し出。
#DialogueProtocols.listSystemAction.append("Offer_"+agent+"_"+item1+"_"+item2)

#オファーもとのエージェント（agent）のitem1とオファーされたエージェントのitem2の交換。
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

#実験条件を決める変数
class ExperimentalConditions:#ノート 現状から動かないＵＳの除去
    #実験メモ欄：対立者数が2の場合の評価.LinQ,LSPI,NFQ
    #Pybrian変更点
    #NFQのActivationNetwork中の中間層*10
    #LSPIのExprolationReward->400
    
    isLearning=False#学習を行うか
    isTest=True#方策評価を行うか
    
    #学習方策
    isUseLinearQ=False#線形関数でＱ関数を近似するか。Ｆａｌｓｅの場合、ニューラルネットを利用する
    #一部方策をルールで置き換えるか。手札が最高の成果のときはKeepにする
    isUsePartiallyRulePolicy=False#False
    
    #ベースライン方策1
    isUseHandCraftedPolicy=False#手製の方策で置き換えるか？
    #
    normofLearner="Uninformative"
    #プランニングをどの程度の深さまで行うか
    searchDepth=2
    
    #ベースライン方策2
    isUseRandomPolicy=False#ランダム方策で置き換えるか？
    #ベースライン方策３
    isAlwaysKeeping=False#常にキープするか
    
    #報酬を現在の手の改善率に応じて逐次与えるか
    #Falseの場合は対話が終わってから与えられる
    isFeedRewardAsImprovement=False
    
    #各自のアイテム数とオファーテーブル以外に対話情報を利用する
    #-数ターン以内に達成可能なゴール
    #-各自の初期アイテム
    #-経過ターン
    #-×現在の報酬
    #-学習エージェントが回ってくる前の全エー権とのAccept/offerテーブル
    #-過去のOffer/Acceptの履歴
    ###Not working (i.e., additional state are doesn't added), but it always should be true. 
    isAddAdditionalDStoLearner=True
    
    #-学習/方策評価時の実験パラメータ
    numSystems=1#学習を行うシステム方策の数。それぞれモデルの初期値が異なる
    numBatch=10#200#学習/方策評価時のパラメータの更新回数
    numEpisode=2000#2000#学習/方策評価時の各バッチで行う対話の回数
    numBatchStartLogging=2#学習時の最適な方策の記録を開始する最低バッチ回数

    #最大ターン数 50->20
    iMaximumTurn=20#トレースを見る限り20程度で十分
    
    #収束が悪いとして学習を打ち切る閾値
    dBadConversion=-500.0
    
    #エージェントの行動のトレース
    isTraceAgent=False
    #Learner(Agent0)の統計量計算
    isCalculateStatisticsOfLearner=True
    
    #実験を行う際の対戦相手の組み合わせ。H,Rで戦略指定。xで区切。例:H, HxH, HxRxH
    #H=ハンドクラフト、R=ランダム
    listCombinationofOpponentsStrategy=[
                                        #"H","R",
                                        "HxH","HxR","RxR", 
                                        #"HxHxH","HxHxR","HxRxR","RxRxR"
                                        ]
    #現在の組み合わせ
    currentCombinationofOpponentsStrategy=None

#JSAI用の分析のためのLearner(Agent0の統計量 #実験全体を通して)
class SatisticsOfLearnerInExperiment:
    #Distribution of Action
    #-Offer
    totalNumberOfOffer=0.0#offerの総数
    totalNumberOfOfferAcceptedByAddressee=0.0#相手にアクセプトされたofferの総数
    #totalNumberOfOfferRejectedByAddressee=0.0#相手にリジェクトされたofferの総数
    
    #-Accept
    totalNumberOfAccept=0.0#オファーに対するアクセプトの数
    totalNumberOfIlligalAccept=0.0#不正なアクセプトの数（e.g., Offerが無いのにアクセプトするなど）
    totalNumberOflligalAccept=0.0#不正なアクセプトの数（e.g., Offerが無いのにアクセプトするなど）
    #-Keep
    totalNumberOfKeep=0.0#Keepの総数
    
    #
    totalNumberOfAction=0.0#Agent0がアクションを行った総回数
    totalNumberOfTotalTurn=0.0#対話の総ターン数
    totalNumberOfDialogue=0.0#総対話数
    
    
#交渉の際の枠組みの規定
class DialogueProtocols:
    #システムのアクション
    #e.g. 20 actions are generated.
    #['Accept', 'Keep', 'Offer_AgentLeaner_Apple_Orange', 'Offer_AgentLeaner_Apple_Grape', 'Offer_AgentLeaner_Orange_Apple', 'Offer_AgentLeaner_Orange_Grape', 'Offer_AgentLeaner_Grape_Apple', 'Offer_AgentLeaner_Grape_Orange', 'Offer_Agent0_Apple_Orange', 'Offer_Agent0_Apple_Grape', 'Offer_Agent0_Orange_Apple', 'Offer_Agent0_Orange_Grape', 'Offer_Agent0_Grape_Apple', 'Offer_Agent0_Grape_Orange', 'Offer_Agent1_Apple_Orange', 'Offer_Agent1_Apple_Grape', 'Offer_Agent1_Orange_Apple', 'Offer_Agent1_Orange_Grape', 'Offer_Agent1_Grape_Apple', 'Offer_Agent1_Grape_Orange']
    listSystemAction=None
    #無駄なアクションを除いたもの
    listSystemActionForLearner=None
    
    #システムの状態
    #27 states are generated.
    #['OfferedFrom_AgentLeaner_Apple_Orange', 'OfferedFrom_AgentLeaner_Apple_Grape', 'OfferedFrom_AgentLeaner_Orange_Apple', 'OfferedFrom_AgentLeaner_Orange_Grape', 'OfferedFrom_AgentLeaner_Grape_Apple', 'OfferedFrom_AgentLeaner_Grape_Orange', 'OfferedFrom_Agent0_Apple_Orange', 'OfferedFrom_Agent0_Apple_Grape', 'OfferedFrom_Agent0_Orange_Apple', 'OfferedFrom_Agent0_Orange_Grape', 'OfferedFrom_Agent0_Grape_Apple', 'OfferedFrom_Agent0_Grape_Orange', 'OfferedFrom_Agent1_Apple_Orange', 'OfferedFrom_Agent1_Apple_Grape', 'OfferedFrom_Agent1_Orange_Apple', 'OfferedFrom_Agent1_Orange_Grape', 'OfferedFrom_Agent1_Grape_Apple', 'OfferedFrom_Agent1_Grape_Orange', 'NumItem_AgentLeaner_Apple', 'NumItem_AgentLeaner_Orange', 'NumItem_AgentLeaner_Grape', 'NumItem_Agent0_Apple', 'NumItem_Agent0_Orange', 'NumItem_Agent0_Grape', 'NumItem_Agent1_Apple', 'NumItem_Agent1_Orange', 'NumItem_Agent1_Grape']
    listSystemState=None
    
    #ドメイン固有のルール
    #エージェントのの名前
    listAgentsName=None
    #アイテム
    listItems=["Apple", "Orange", "Grape"]
    #アイテムへの嗜好
    dicPreference={"Like":100.0, "Neutral":0.0, "Hate":-100.0}
    #役割
    dicRole={"Rich":4, "Middle":3, "Poor":2}
    
    #アイテムの個数に応じた達成可能な目標テーブル@Rich man
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
    #アイテムの個数に応じた達成可能な目標テーブル@Normal man
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
    #アイテムの個数に応じた達成可能な目標テーブル@Poor man
    listAchievablePoorHands=[
                        {"Apple":0, "Orange":0, "Grape":2},
                        {"Apple":0, "Orange":1, "Grape":1},
                        {"Apple":1, "Orange":0, "Grape":1},
                        {"Apple":0, "Orange":2, "Grape":0},
                        {"Apple":1, "Orange":1, "Grape":0},
                        {"Apple":2, "Orange":0, "Grape":0},
                        ]
    #対戦相手の性質
    listNormOpponent=["Optimist", "Uninformative","Pessimist"]
    
    #追加した対話状態の評価よう
    listDialogStateinSpecialCase=[]
    
    #対戦相手の振る舞いのリスト
    listTypeofOpponentsStrategy=["Random","Handcrafted"]
    
#     #TEST
#     totalOffer=0.0
#     totalAccept=0.0
    
class TestEnv(Environment):
    def __init__(self):
        Environment.__init__(self)
        #状態初期化の設定
        self.reset()
    def reset(self):
        #実装
        #各対戦相手（エージェント）の情報(PayOff行列, 初期アイテム)の初期化 
        self.dicAgents={}
        #各エージェントのアイテムの総数
        tNumItemAgent={}
        for agent in DialogueProtocols.listAgentsName:
            #Payoff行列の生成--それぞれの嗜好がひとつ筒アイテムに対応
            #アイテム:嗜好
            dicPayoff={}
            temp=DialogueProtocols.dicPreference.keys()
            for item in DialogueProtocols.listItems: 
                dicPayoff[item]=None
            for item in DialogueProtocols.listItems: 
                num=random.randint(low=0,high=99999)%len(temp)
                dicPayoff[item]=temp[num]
                temp.pop(num)
            #print dicPayoff
            #初期アイテムの生成
            numTotalItem=DialogueProtocols.dicRole[DialogueProtocols.dicRole.keys()[random.randint(low=0,high=99999)%len(DialogueProtocols.dicRole.keys())]]
            tNumItemAgent[agent]=numTotalItem
            #numTotalItem=3#みんな等しく３つ持っている
            dicNumItems={}
            for i in range(len(DialogueProtocols.listItems)):
                dicNumItems[DialogueProtocols.listItems[i]]=0
            for i in range(numTotalItem):
                dicNumItems[DialogueProtocols.listItems[random.randint(low=0,high=99999)%len(DialogueProtocols.listItems)]] +=1
            #初期状態生成
            dicInfo={}
            dicInfo["InitNumItems"]=dicNumItems
            dicInfo["Payoff"]=dicPayoff
            self.dicAgents[agent]=dicInfo
        #print self.dicAgents
        #Learnerの条件の固定
        dicNumItems={}
        for i in range(len(DialogueProtocols.listItems)):
            dicNumItems[DialogueProtocols.listItems[i]]=0
        for i in range(3):
            dicNumItems[DialogueProtocols.listItems[random.randint(low=0,high=99999)%len(DialogueProtocols.listItems)]] +=1
        self.dicAgents["AgentLearner"]["InitNumItems"]=dicNumItems.copy()
        self.dicAgents["AgentLearner"]["Payoff"]={DialogueProtocols.listItems[0]:DialogueProtocols.dicPreference.keys()[0],DialogueProtocols.listItems[1]:DialogueProtocols.dicPreference.keys()[1],DialogueProtocols.listItems[2]:DialogueProtocols.dicPreference.keys()[2]}
        #Pay Learner以外のPayoffの固定
        #self.dicAgents["Agent0"]["Payoff"]={DialogueProtocols.listItems[0]:DialogueProtocols.dicPreference.keys()[0],DialogueProtocols.listItems[1]:DialogueProtocols.dicPreference.keys()[1],DialogueProtocols.listItems[2]:DialogueProtocols.dicPreference.keys()[2]}
        #self.dicAgents["Agent1"]["Payoff"]={DialogueProtocols.listItems[1]:DialogueProtocols.dicPreference.keys()[0],DialogueProtocols.listItems[2]:DialogueProtocols.dicPreference.keys()[1],DialogueProtocols.listItems[0]:DialogueProtocols.dicPreference.keys()[2]}

        #print self.dicAgents
        
        #直前のエージェントのアクション
        self.mostPreviousAction=""
        self.mostPreviousAgent=""
        #ランダム選択用のエージェントのスタック
        self.listOrgderAgentsTakingAction=[]
        #経過ターン
        self.turn=0
        #連続Keep数
        self.numContKeep=0
        
        #各エージェントが達成可能な成果：プランニング方策専用変数
        self.achievableOutcome=None
        #各エージェントのアイテムの総数：プランニング方策専用変数
        self.totalNumofItemsofAgents=None
        
        #エージェント性質の決定
        self.normofAgents={}
        for agent in DialogueProtocols.listAgentsName:
            self.normofAgents[agent]="Uninformative"#DialogueProtocols.listNormOpponent[random.randint(low=0,high=99999)%len(DialogueProtocols.listNormOpponent)]
        self.normofAgents["AgentLearner"]=ExperimentalConditions.normofLearner        
        #各エージェントがプランベースの探索で何手先まで計算するか1~4
        self.searchDepth={}
        for agent in DialogueProtocols.listAgentsName:
            self.searchDepth[agent]=tNumItemAgent[agent]#(random.randint(low=0,high=9999)%(tNumItemAgent[agent]-1))+1
        self.searchDepth["AgentLearner"]=ExperimentalConditions.searchDepth
        #対戦相手の戦略の決定
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
        
        #共有された対話状態の初期化
        self.dicsharedDialogState={}
        for ds in DialogueProtocols.listSystemState: 
            self.dicsharedDialogState[ds]=0.0
        #アイテムの初期化
        for ds in self.dicsharedDialogState.keys():
            for agent in DialogueProtocols.listAgentsName:
                for item in DialogueProtocols.listItems: 
                    if (re.search("NumItem", ds)!=None) and (re.search(agent, ds)!=None) and (re.search(item, ds)!=None):
                        self.dicsharedDialogState[ds] = self.dicAgents[agent]["InitNumItems"][item]
        #-バイアス稿の設定
        self.dicsharedDialogState["Bias"]=1.0
        
        #print self.dicsharedDialogState
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            #-初期アイテムを設定
            pass
        #print self.dicsharedDialogState
        
        
        
        #エージェントの行動のトレース
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

        #エージェントがゴールに達したか
        self.isLearnerReachMaximumOutcome=False

        #1ターン前の学習者の状態
        self.previousLeanerDialogState=copy.copy(self.dicsharedDialogState)
                        
    def performAction(self, action):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            SatisticsOfLearnerInExperiment.totalNumberOfAction+=1.0
        
        #対話状態のバックアップ
        self.previousLeanerDialogState=copy.copy(self.dicsharedDialogState)
        
        #エージェントのトレース用
        if ExperimentalConditions.isTraceAgent:
            print"AgentLearners action=" + DialogueProtocols.listSystemActionForLearner[int(action[0])]
        #Offer/Accept Tableの初期化
        for agent1 in DialogueProtocols.listAgentsName:#Offerしたエージェント
            for agent2 in DialogueProtocols.listAgentsName:#Offerされたエージェント
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if item1 != item2:
                            self.dicsharedDialogState["Accept_"+agent1+"_"+agent2+"_"+item1+"_"+item2]=0
                            self.dicsharedDialogState["FO_"+agent1+"_"+agent2+"_"+item1+"_"+item2]=0

        #システムエージェンとの行動
        #-Learnerの場合
        #-ランダム方策を使う場合
        if ExperimentalConditions.isUseRandomPolicy:
            #部分ルールの適応            #TEST Fruit saradの場合は強制的にパス
            numEachItem=[]
            for item in DialogueProtocols.listItems:
                numEachItem.append(self.dicsharedDialogState["NumItem_AgentLearner_"+item])
            min(numEachItem)
            if ExperimentalConditions.isUsePartiallyRulePolicy and (min(numEachItem) > 0.5):
                self._DSupdateDoNothing("AgentLearner")
                self.isLearnerReachMaximumOutcome=True
            else:
                self._RandomValidateActionPolicy("AgentLearner")
        #常にキープだけする場合
        elif ExperimentalConditions.isAlwaysKeeping:
            self._DSupdateDoNothing("AgentLearner")
        #-学習方策を使う場合
        elif (not ExperimentalConditions.isUseHandCraftedPolicy):
#             #TEST
#             action[0]=random.randint(low=0,high=9999)%len(DialogueProtocols.listSystemActionForLearner)
            
            #部分ルールの適応            #TEST Fruit saradの場合は強制的にパス
            numEachItem=[]
            for item in DialogueProtocols.listItems:
                numEachItem.append(self.dicsharedDialogState["NumItem_AgentLearner_"+item])
            min(numEachItem)
            if ExperimentalConditions.isUsePartiallyRulePolicy and (min(numEachItem) > 0.5):
                self._DSupdateDoNothing("AgentLearner")
                self.isLearnerReachMaximumOutcome=True
            #以後学習された方策
            elif re.search("Offer", DialogueProtocols.listSystemActionForLearner[int(action[0])]) != None:
                if re.search("AgentLearner", DialogueProtocols.listSystemActionForLearner[int(action[0])]) == None:
                    pAct=DialogueProtocols.listSystemActionForLearner[int(action[0])].split("_")
                    self._DSupdateExecuteOffer("AgentLearner", pAct[1], pAct[2], pAct[3])
#                     #TEST
#                     DialogueProtocols.totalOffer+=1.0

                else:
                    self._DSupdateDoNothing("AgentLearner")#自分にオファーをしない
                    #TEST Keepをエージェントがしないよう＆学習を高速で終わるようにKeepした瞬間に対話を終了する
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
                    #TEST Keepをエージェントがしないよう＆学習を高速で終わるようにKeepした瞬間に対話を終了する
                    #self.numContKeep+=1000
            elif re.search("Keep", DialogueProtocols.listSystemActionForLearner[int(action[0])]) != None:
                self._DSupdateDoNothing("AgentLearner")
                #TEST Keepをエージェントがしないよう＆学習を高速で終わるようにKeepした瞬間に対話を終了する
                #self.numContKeep+=1000
            else:
                assert False, "System didnt take any action."
        else:
            #self._GreedywisePolicy("AgentLearner")
            self._PlanbasedPolicy("AgentLearner")
            #self._RandomPlanInBetterGoalbasedPolicy("AgentLearner")#TEST
        self.turn+=1
        #print DialogueProtocols.listSystemAction[int(action[0])]
        
        #その後の順序決定&ターン進行
        agentTakingAction=""
        while agentTakingAction !="AgentLearner" and (self.turn < ExperimentalConditions.iMaximumTurn):
            if (self.numContKeep >=len(self.dicAgents.keys())):
                break
            elif self.isLearnerReachMaximumOutcome:
                break
            #順序決定
            #-直前のエージェントのアクションが特定のエージェントに対数Offerの場合、Offerを受けたエージェントが優先的に動ける
            if re.search("Offer_",self.mostPreviousAction) !=None:
                for agent in DialogueProtocols.listAgentsName: 
                    if re.search(agent,self.mostPreviousAction) != None:
                        agentTakingAction=agent
                        #Offerを受けたエージェントの連続行動を規定
                        if not self.listOrgderAgentsTakingAction == None:
                            for elem in self.listOrgderAgentsTakingAction:
                                if agentTakingAction == elem:
                                    self.listOrgderAgentsTakingAction.remove(agentTakingAction)
            #-そうでない場合は、ランダム順序
            else:
                if len(self.listOrgderAgentsTakingAction) ==0: #リストの長さが０なら初期化
                    temp=copy.copy(DialogueProtocols.listAgentsName)
                    while len(temp) > 0:
                        self.listOrgderAgentsTakingAction.append(temp.pop(random.randint(low=0,high=99999)%len(temp)))
                    self.listOrgderAgentsTakingAction.remove(self.mostPreviousAgent)
                    self.listOrgderAgentsTakingAction.append(self.mostPreviousAgent)
                agentTakingAction=self.listOrgderAgentsTakingAction.pop(0)
            assert agentTakingAction != None, "Illegal agent selection"
            #print self.listOrgderAgentsTakingAction
            
            #指定されたエージェントの行動-Learnerでは無い場合
            if agentTakingAction != "AgentLearner":
                #self._GreedywisePolicy(agentTakingAction)
                if self.listOpponentAgentStrategy[agentTakingAction]=="Random":
                    self._RandomValidateActionPolicy(agentTakingAction)
                elif self.listOpponentAgentStrategy[agentTakingAction]=="Handcrafted":
                    self._PlanbasedPolicy(agentTakingAction)
                else:
                    assert False, "Illegal Opponents strategy"
                self.turn+=1

        #追加の対話状態の追加
        #
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            self.dicsharedDialogState["CurrentTurn"]=self.turn
    
    #有効なアクションのみからなるランダム方策
    def _RandomValidateActionPolicy(self,agentTakingAction):
        #有効なOffer集合を調べる
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
                self._DSupdateDoNothing(agentTakingAction)#自分にオファーをしない
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
    
    #ランダム方策
    def _RandomPolicy(self,agentTakingAction):
        action=random.randint(low=0,high=99999)%len(DialogueProtocols.listSystemAction)

        if re.search("Offer", DialogueProtocols.listSystemAction[int(action)]) != None:
            if re.search(agentTakingAction, DialogueProtocols.listSystemAction[int(action)]) == None:
                pAct=DialogueProtocols.listSystemAction[int(action)].split("_")
                self._DSupdateExecuteOffer(agentTakingAction, pAct[1], pAct[2], pAct[3])
            else:
                self._DSupdateDoNothing(agentTakingAction)#自分にオファーをしない
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

    #Hand-crafted policy三号(Proposed by David-san) 2014/1/06
    #-現在の状態から達成可能なゴールに向けてのプランを生成する。
    #-そして、その達成可能なプランからランダムに従うプランを選ぶ
    #-そして、生成される成果が最も良いものを選ぶ。現状より良い成果が期待できない場合は、Keep.
    def _RandomPlanInBetterGoalbasedPolicy(self, agentTakingAction):
        #各エージェントの初期アイテムとPayoffに応じた成果テーブルを作成：
        if self.achievableOutcome == None:
            self.achievableOutcome={}#アイテムＡ＿アイテムＡの個数_アイテムＢ＿アイテムＢの個数_アイテムＣ＿アイテムＣの個数
            self.totalNumofItemsofAgents={}
            #-成果テーブルの生成
            #--各エージェントのアイテムの総数を求める
            dicTotalNumofItemofAgents={}
            for agent in self.dicAgents.keys():
                totalNum= 0
                for item in self.dicAgents[agent]["InitNumItems"].keys():
                    totalNum+=self.dicAgents[agent]["InitNumItems"][item]
                dicTotalNumofItemofAgents[agent]=totalNum
            self.totalNumofItemsofAgents=copy.copy(dicTotalNumofItemofAgents)
            #--それぞれの持ち方の通りに応じた成果を計算する
            #---アイテム数におうじた可能な手札の設定
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
            #---各自のPayoff行列に応じた成果の設定
            for agent in self.dicAgents.keys():
                #各エージェントの手ごとの報酬
                outcomes={}
                for hand in dicAchievableHandofAgents[agent]:
                    rew=0.0
                    numEachItem=[]
                    for item in DialogueProtocols.listItems:
                        rew+=DialogueProtocols.dicPreference[self.dicAgents[agent]["Payoff"][item]]*hand[item]
                        numEachItem.append(hand[item])
                    #フルーツサラダ
                    rew+=500*min(numEachItem)
                    outcomes[hand.keys()[0]+"_"+str(hand[hand.keys()[0]])+"_"+hand.keys()[1]+"_"+str(hand[hand.keys()[1]])+"_"+hand.keys()[2]+"_"+str(hand[hand.keys()[2]])]=rew
                self.achievableOutcome[agent]=outcomes
                
        #現在のエージェントの手持ちの成果よりも悪い成果をテーブル中から除外する
        achievableProfitOutcomATA={}
        #-現在のエージェントの成果を計算
        currentrew=0.0
        numEachItem=[]
        for item in DialogueProtocols.listItems:
            currentrew+=DialogueProtocols.dicPreference[self.dicAgents[agentTakingAction]["Payoff"][item]]*self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item]
            numEachItem.append(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item])
        currentrew+=500*min(numEachItem)
        #-除外の実行
        for outcomes in self.achievableOutcome[agentTakingAction].keys():
            if self.achievableOutcome[agentTakingAction][outcomes] > currentrew:
                achievableProfitOutcomATA[outcomes]=self.achievableOutcome[agentTakingAction][outcomes]
        
        #プランニング（i.e.最短経路探索）
        #TODO<------------------------------------------------------------------
        #-自分の手を含めた各アイテムの総数のカウント (TODO 最初に呼ばれたときにのみ計算するように変更)
        numofEachItems={}
        for item in DialogueProtocols.listItems:
            totalNum=0
            for agent in DialogueProtocols.listAgentsName:
                totalNum+=self.dicAgents[agent]["InitNumItems"][item]
            numofEachItems[item]=totalNum
            
        #-残りのターンで明らかに達成出来ない場合は削除(目標とアイテムのL1距離が残りの手数以上の場合は削除)
        availablePlanatEachHand={}
        for hand in achievableProfitOutcomATA.keys():
            #print hand.split("_")
            initHand=hand.split("_")[0]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[0]]))+"_"
            initHand+=hand.split("_")[2]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[2]]))+"_"
            initHand+=hand.split("_")[4]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[4]]))
            queue=[[initHand]]#アイテム数の状態系列
            correctPlan=[]
            #幅優先探索
            while len(queue) > 0:
                currentPlan=queue.pop(0)
                dist=0
                #---距離の計算
                for i in range(len(hand.split("_"))/2):
                    #print int(hand.split("_")[(i*2)+1])
                    #print int(currentPlan[-1].split("_")[(i*2)+1])
                    #print str(i)
                    #print hand.split("_")
                    #print currentPlan[-1].split("_")
                    dist+=abs(int(float(hand.split("_")[(i*2)+1]))-int(float(currentPlan[-1].split("_")[(i*2)+1])))
                #--ノードがこれ以上展開する必要が無い場合
                if dist == 0: #終端状態の場合
                    correctPlan.append(copy.copy(currentPlan))
                #TESTING NOW
                #elif dist > (self.totalNumofItemsofAgents[agentTakingAction]-len(currentPlan)+1): #既定の手数で到達できない場合
                elif dist > (self.searchDepth[agentTakingAction]-len(currentPlan)+1): #既定の手数で到達できない場合
                    pass
                else: 
                    #ノードを展開する必要がある場合
                    #--自己ノードの追記
                    #--ノードの展開：
                    for itemGive in DialogueProtocols.listItems:
                        for itemGiven in DialogueProtocols.listItems:
                            if itemGive == itemGiven:
                                continue
                            expandedHand=copy.copy(currentPlan[-1])
                            dicExpandedHand={}
                            for i in range((len(expandedHand.split("_"))/2)):
                                dicExpandedHand[expandedHand.split("_")[i*2]]=int(float(expandedHand.split("_")[(i*2)+1]))
                            #ノードの適切性の計算(i.e. 現在の状態でアイテムが交換可能か？)
                            if (dicExpandedHand[itemGive]-1) < 0:
                                continue
                            elif (numofEachItems[itemGiven]-(dicExpandedHand[itemGiven]+1) < 0):
                                continue
                            else :#適切なノードの追加
                                #--条件にあったノードの格納:
                                for item in dicExpandedHand.keys():
                                    dicExpandedHand[item]=int(dicExpandedHand[item])#intにキャスト
                                dicExpandedHand[itemGive]-=1
                                dicExpandedHand[itemGiven]+=1
                                #
                                expandedPlan=copy.copy(currentPlan)
                                codedHand=hand.split("_")[0]+"_"+str(dicExpandedHand[hand.split("_")[0]])+"_"
                                codedHand+=hand.split("_")[2]+"_"+str(dicExpandedHand[hand.split("_")[2]])+"_"
                                codedHand+=hand.split("_")[4]+"_"+str(dicExpandedHand[hand.split("_")[4]])
                                expandedPlan.append(codedHand)
                                queue.append(expandedPlan)
            #-各ゴールごとに到達可能なプラン集合を追加する
            availablePlanatEachHand[hand]=copy.copy(correctPlan)
        #---4Test@
        #if len(availablePlanatEachHand) > 0:
        #    print availablePlanatEachHand
        #else:
        #    print "No valuable plan"
        #プランを適当に選択
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
        #現在の成果とプランニング最良の期待成果を比較して、Keepか(Offer or acceptかを決める)
        if (len(bestPlan)<=1):#Keepする場合
            self._DSupdateDoNothing(agentTakingAction)
        else:#Offer or Acceptする場合
            # TODO-----------------------------
            #-あげるアイテムともらうアイテムを計算する
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
            #-offeredがあり、プランに合致していれば受け入れる
            #----------------------------------------------ここまでデバッグ済み
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
            if not isAccepted:#-そうでない場合はOfferするアイテムを探す
                isOffered=False
                #-Givenを持っているエージェント見る
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


    #Hand-crafted policy二号(Proposed by David-san) 2014/12/03
    #-現在の状態から達成可能なゴールに向けてのプランを生成する。
    #-そして、生成される成果が最も良いものを選ぶ。現状より良い成果が期待できない場合は、Keep.
    def _PlanbasedPolicy(self, agentTakingAction):
        #各エージェントの初期アイテムとPayoffに応じた成果テーブルを作成：
        if self.achievableOutcome == None:
            self.achievableOutcome={}#アイテムＡ＿アイテムＡの個数_アイテムＢ＿アイテムＢの個数_アイテムＣ＿アイテムＣの個数
            self.totalNumofItemsofAgents={}
            #-成果テーブルの生成
            #--各エージェントのアイテムの総数を求める
            dicTotalNumofItemofAgents={}
            for agent in self.dicAgents.keys():
                totalNum= 0
                for item in self.dicAgents[agent]["InitNumItems"].keys():
                    totalNum+=self.dicAgents[agent]["InitNumItems"][item]
                dicTotalNumofItemofAgents[agent]=totalNum
            self.totalNumofItemsofAgents=copy.copy(dicTotalNumofItemofAgents)
            #--それぞれの持ち方の通りに応じた成果を計算する
            #---アイテム数におうじた可能な手札の設定
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
            #---各自のPayoff行列に応じた成果の設定
            for agent in self.dicAgents.keys():
                #各エージェントの手ごとの報酬
                outcomes={}
                for hand in dicAchievableHandofAgents[agent]:
                    rew=0.0
                    numEachItem=[]
                    for item in DialogueProtocols.listItems:
                        rew+=DialogueProtocols.dicPreference[self.dicAgents[agent]["Payoff"][item]]*hand[item]
                        numEachItem.append(hand[item])
                    #フルーツサラダ
                    rew+=500*min(numEachItem)
                    outcomes[hand.keys()[0]+"_"+str(hand[hand.keys()[0]])+"_"+hand.keys()[1]+"_"+str(hand[hand.keys()[1]])+"_"+hand.keys()[2]+"_"+str(hand[hand.keys()[2]])]=rew
                self.achievableOutcome[agent]=outcomes
                
        #現在のエージェントの手持ちの成果よりも悪い成果をテーブル中から除外する
        achievableProfitOutcomATA={}
        #-現在のエージェントの成果を計算
        currentrew=0.0
        numEachItem=[]
        for item in DialogueProtocols.listItems:
            currentrew+=DialogueProtocols.dicPreference[self.dicAgents[agentTakingAction]["Payoff"][item]]*self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item]
            numEachItem.append(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+item])
        currentrew+=500*min(numEachItem)
        #-除外の実行
        for outcomes in self.achievableOutcome[agentTakingAction].keys():
            if self.achievableOutcome[agentTakingAction][outcomes] > currentrew:
                achievableProfitOutcomATA[outcomes]=self.achievableOutcome[agentTakingAction][outcomes]
        
        #プランニング（i.e.最短経路探索）
        #TODO<------------------------------------------------------------------
        #-自分の手を含めた各アイテムの総数のカウント (TODO 最初に呼ばれたときにのみ計算するように変更)
        numofEachItems={}
        for item in DialogueProtocols.listItems:
            totalNum=0
            for agent in DialogueProtocols.listAgentsName:
                totalNum+=self.dicAgents[agent]["InitNumItems"][item]
            numofEachItems[item]=totalNum
            
        #-残りのターンで明らかに達成出来ない場合は削除(目標とアイテムのL1距離が残りの手数以上の場合は削除)
        availablePlanatEachHand={}
        for hand in achievableProfitOutcomATA.keys():
            #print hand.split("_")
            initHand=hand.split("_")[0]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[0]]))+"_"
            initHand+=hand.split("_")[2]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[2]]))+"_"
            initHand+=hand.split("_")[4]+"_"+str(int(self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hand.split("_")[4]]))
            queue=[[initHand]]#アイテム数の状態系列
            correctPlan=[]
            #幅優先探索
            while len(queue) > 0:
                currentPlan=queue.pop(0)
                dist=0
                #---距離の計算
                for i in range(len(hand.split("_"))/2):
                    #print int(hand.split("_")[(i*2)+1])
                    #print int(currentPlan[-1].split("_")[(i*2)+1])
                    #print str(i)
                    #print hand.split("_")
                    #print currentPlan[-1].split("_")
                    dist+=abs(int(float(hand.split("_")[(i*2)+1]))-int(float(currentPlan[-1].split("_")[(i*2)+1])))
                #--ノードがこれ以上展開する必要が無い場合
                if dist == 0: #終端状態の場合
                    correctPlan.append(copy.copy(currentPlan))
                #TESTING NOW
                #elif dist > (self.totalNumofItemsofAgents[agentTakingAction]-len(currentPlan)+1): #既定の手数で到達できない場合
                elif dist > (self.searchDepth[agentTakingAction]-len(currentPlan)+1): #既定の手数で到達できない場合
                    pass
                else: 
                    #ノードを展開する必要がある場合
                    #--自己ノードの追記
                    #--ノードの展開：
                    for itemGive in DialogueProtocols.listItems:
                        for itemGiven in DialogueProtocols.listItems:
                            if itemGive == itemGiven:
                                continue
                            expandedHand=copy.copy(currentPlan[-1])
                            dicExpandedHand={}
                            for i in range((len(expandedHand.split("_"))/2)):
                                dicExpandedHand[expandedHand.split("_")[i*2]]=int(float(expandedHand.split("_")[(i*2)+1]))
                            #ノードの適切性の計算(i.e. 現在の状態でアイテムが交換可能か？)
                            if (dicExpandedHand[itemGive]-1) < 0:
                                continue
                            elif (numofEachItems[itemGiven]-(dicExpandedHand[itemGiven]+1) < 0):
                                continue
                            else :#適切なノードの追加
                                #--条件にあったノードの格納:
                                for item in dicExpandedHand.keys():
                                    dicExpandedHand[item]=int(dicExpandedHand[item])#intにキャスト
                                dicExpandedHand[itemGive]-=1
                                dicExpandedHand[itemGiven]+=1
                                #
                                expandedPlan=copy.copy(currentPlan)
                                codedHand=hand.split("_")[0]+"_"+str(dicExpandedHand[hand.split("_")[0]])+"_"
                                codedHand+=hand.split("_")[2]+"_"+str(dicExpandedHand[hand.split("_")[2]])+"_"
                                codedHand+=hand.split("_")[4]+"_"+str(dicExpandedHand[hand.split("_")[4]])
                                expandedPlan.append(codedHand)
                                queue.append(expandedPlan)
            #-各ゴールごとに到達可能なプラン集合を追加する
            availablePlanatEachHand[hand]=copy.copy(correctPlan)
        #---4Test@
        #if len(availablePlanatEachHand) > 0:
        #    print availablePlanatEachHand
        #else:
        #    print "No valuable plan"
        #最良のプランを選択
        #プランニングで求めたパスの期待値を計算
        bestPlan=[]
        bestExpectedOutcome=-9999
        probSuccessofproceeding=None#各プランを正しく進行出来る確立。
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
        #現在の成果とプランニング最良の期待成果を比較して、Keepか(Offer or acceptかを決める)
        if (len(bestPlan)<=1) or (bestExpectedOutcome < currentrew):#Keepする場合
            self._DSupdateDoNothing(agentTakingAction)
        else:#Offer or Acceptする場合
            # TODO-----------------------------
            #-あげるアイテムともらうアイテムを計算する
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
            #-offeredがあり、プランに合致していれば受け入れる
            #----------------------------------------------ここまでデバッグ済み
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
            if not isAccepted:#-そうでない場合はOfferするアイテムを探す
                isOffered=False
                #-Givenを持っているエージェント見る
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
    
    #Hand-crafted policy一号　２０１４・１２・０１
    #-Greedy wise方策:もしOfferを受けて、そのOfferが有益なもの（スコアをあげる）のであれば、Acceptする    
    #-Ofeerが無い場合で、もしＨａｔｅアイテムをもっている場合、Likeのアイテムを持つ人にランダムにＯｆｆｅｒする。もし、いなければ、Neutralのアイテムを持っている人を探してＯｆｆｅｒする
    #もしＨａｔｅアイテムを持っていなくて、Ｎｅｕｔｒａｌアイテムをもっている場合、Likeのアイテムを持つ人にランダムにＯｆｆｅｒする。
    #それ以外の場合は、Keepする
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
                #--OffeｒであげるアイテムがHateの場合無条件であげる(ただし)
                if self.dicAgents[agentTakingAction]["Payoff"][offerItemtoGive] =="Hate":
                    if self._isAvailableTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven):
                        self._DSupdateExecuteTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven)
                        isAccepted=True
                #--OfferであげるアイテムがNeutralの場合は、くれるアイテムの場合がLikeの場合のみ対処
                elif (self.dicAgents[agentTakingAction]["Payoff"][offerItemtoGive] =="Neutral") and (self.dicAgents[agentTakingAction]["Payoff"][offerItemtoGiven] =="Like"):
                    if self._isAvailableTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven):
                        self._DSupdateExecuteTrade(agentTakingAction, offerAgent, offerItemtoGive, offerItemtoGiven)
                        isAccepted=True
        
        #アクセプトしない場合は、スコアをあげそうな人をランダムに選んで交渉
        isOffered=False
        likeItem=None
        for item in self.dicAgents[agentTakingAction]["Payoff"].keys():
            if self.dicAgents[agentTakingAction]["Payoff"][item] == "Like":
                likeItem=item
        assert likeItem != None, "Illegal access to payoff mat"
        if not isAccepted and not (self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+likeItem] == self.dicAgents[agentTakingAction]["InitNumItems"]):
            #Hateするアイテムを持っている場合
            hateItem=None
            for item in self.dicAgents[agentTakingAction]["Payoff"].keys():
                if self.dicAgents[agentTakingAction]["Payoff"][item] == "Hate":
                    hateItem = item
            if self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+hateItem] > 0.0:
                #-エージェントLikeを持っているエージェント見る
                agentsHavingLikeItem=[]
                for agent in DialogueProtocols.listAgentsName:
                    if (agent != agentTakingAction):
                        for item in DialogueProtocols.listItems:
                            if (self.dicAgents[agentTakingAction]["Payoff"][item]=="Like") and (self.dicsharedDialogState["NumItem_"+agent+"_"+item] > 0.0):
                                agentsHavingLikeItem.append(agent)
                if len(agentsHavingLikeItem) > 0:
                    self._DSupdateExecuteOffer(agentTakingAction, agentsHavingLikeItem[random.randint(0,9999)%len(agentsHavingLikeItem)], hateItem,likeItem)
                    isOffered=True
                else:#だれもライクを持っていない場合はニュートラルをみる
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
            #HateItemがなくて、Neutraアイテムを持っている場合
            if not isOffered:
                neutralItem=None
                for item in self.dicAgents[agentTakingAction]["Payoff"].keys():
                    if self.dicAgents[agentTakingAction]["Payoff"][item] == "Neutral":
                        neutralItem = item
                if self.dicsharedDialogState["NumItem_"+agentTakingAction+"_"+neutralItem] > 0.0:
                    #-エージェントLikeを持っているエージェント見る
                    agentsHavingLikeItem=[]
                    for agent in DialogueProtocols.listAgentsName:
                        if (agent != agentTakingAction):
                            for item in DialogueProtocols.listItems:
                                if (self.dicAgents[agentTakingAction]["Payoff"][item]=="Like") and (self.dicsharedDialogState["NumItem_"+agent+"_"+item] > 0.0):
                                    agentsHavingLikeItem.append(agent)
                    if len(agentsHavingLikeItem) > 0:
                        self._DSupdateExecuteOffer(agentTakingAction, agentsHavingLikeItem[random.randint(0,9999)%len(agentsHavingLikeItem)], neutralItem,likeItem)
                        isOffered=True                            
        #それ以外の場合はＫｅｅｐ
        if (not isAccepted) and (not isOffered):
            self._DSupdateDoNothing(agentTakingAction)

       
    #sourceAgentはitem1を差し出す。TargetAgentはitem2を差し出す 
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
            
        #OfferTableの初期化
        for ds in self.dicsharedDialogState.keys():
            if re.search("Offered",ds) != None:
                self.dicsharedDialogState[ds]=0.0
        #
        self.numContKeep=0
        
        if ExperimentalConditions.isAddAdditionalDStoLearner:
            #print "Full offered/accepted table"
            self.dicsharedDialogState["Accept_"+sourceAgent+"_"+targetAgent+"_"+item1+"_"+item2]=1.0
            self.dicsharedDialogState["HistoryA_"+sourceAgent+"_"+targetAgent+"_"+item1+"_"+item2]+=1.0
        
        #トレース
        if ExperimentalConditions.isTraceAgent:
            print "Turn "+str(self.turn)
            print sourceAgent + " accept " + targetAgent + "'s offer that " + sourceAgent + " give " + item1 + ", and " + targetAgent + " give " + item2
            for agent in DialogueProtocols.listAgentsName:
                print agent + " have: "
                for item in DialogueProtocols.listItems:
                    print item + ":" + str(self.dicsharedDialogState["NumItem_"+agent+"_"+item])
    
    #対話状態の更新誰かはOfferしたとき
    #sourceAgentはitem1を差し出す。TargetAgentはitem2を差し出す 
    def _DSupdateExecuteOffer(self,sourceAgent, targetAgent, item1, item2):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            if sourceAgent == "AgentLearner":
                SatisticsOfLearnerInExperiment.totalNumberOfOffer+=1.0

        #RejectTableのアップデート
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
        #トレース
        if ExperimentalConditions.isTraceAgent:
            print "Turn "+str(self.turn)
            print sourceAgent + " offer to " + targetAgent + " that " + sourceAgent + " give " + item1 + ", and " + targetAgent + " give " + item2
    
    #何もしない i.e. Keep
    def _DSupdateDoNothing(self,sourceAgent, targetAgent=None, item1=None, item2=None):
        #
        if ExperimentalConditions.isCalculateStatisticsOfLearner:
            if sourceAgent == "AgentLearner":
                SatisticsOfLearnerInExperiment.totalNumberOfKeep+=1.0
            
        #RejectTableのアップデート
        if (re.search("Offer_",self.mostPreviousAction) != None):
            spAct=self.mostPreviousAction.split("_")
            self.dicsharedDialogState["Rejected_"+self.mostPreviousAgent+"_"+sourceAgent+"_"+spAct[2]+"_"+spAct[3]]=1.0
        
        self.mostPreviousAction="Keep"
        self.mostPreviousAgent=sourceAgent
        self.numContKeep+=1
        #OfferTableの初期化
        for ds in self.dicsharedDialogState.keys():
            if re.search("Offered",ds) != None:
                self.dicsharedDialogState[ds]=0.0
        #トレース
        if ExperimentalConditions.isTraceAgent:
            print "Turn "+str(self.turn)
            print sourceAgent + " keeping"
            
    
    #トレードが可能かチェックする return boolean
    #高速化 20141215
    def _isAvailableTrade(self,sourceAgent, targetAgent, item1, item2):
        #sourceaAgentの持ち物チェック
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
            
            #対話状態のトレース
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
        
#         #TEST 逐次報酬（サブゴール）
#         numEachItem=[]
#         for item in DialogueProtocols.listItems:
#             rew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item]
#             numEachItem.append(self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item])
#         #フルーツサラダ
#         rew+=500*min(numEachItem)

        #改善率に応じた逐次報酬
        if ExperimentalConditions.isFeedRewardAsImprovement:
            #
            #print "reward"
            
            #1ターン前での報酬計算
            numEachItem=[]
            tempPrevRew=0.0
            for item in DialogueProtocols.listItems:
                tempPrevRew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.previousLeanerDialogState["NumItem_AgentLearner_"+item]
                numEachItem.append(self.testEnv.previousLeanerDialogState["NumItem_AgentLearner_"+item])
            #TEST
            #print "Prev: ",
            #print numEachItem
            tempPrevRew+=500*min(numEachItem)
            
            #現在のターンでの報酬計算
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
            #全ターン終了後にいっせいにあげる            
            if self.isFinished():
                numEachItem=[]
                for item in DialogueProtocols.listItems:
                    rew+=DialogueProtocols.dicPreference[self.testEnv.dicAgents["AgentLearner"]["Payoff"][item]]*self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item]
                    numEachItem.append(self.testEnv.dicsharedDialogState["NumItem_AgentLearner_"+item])
                #フルーツサラダ
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

#--------------------------------main 以下,強化学習用プログラム----------------------------------------------------------
warnings.filterwarnings(action="ignore")    
for combofOpponentsStrategy in ExperimentalConditions.listCombinationofOpponentsStrategy:
    ExperimentalConditions.currentCombinationofOpponentsStrategy=combofOpponentsStrategy
    currennumOpponents=len(combofOpponentsStrategy.split("x"))
    print "Current number of opponents=" + str(len(combofOpponentsStrategy.split("x")))
    print "Combination of Opponents=" + ExperimentalConditions.currentCombinationofOpponentsStrategy
    #実験条件に応じたエージェントの名前・アクション生成.............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................../................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................./////&状態生成
    #-エージェントの名前
    DialogueProtocols.listAgentsName=["AgentLearner"]
    for i in range(currennumOpponents): 
        DialogueProtocols.listAgentsName.append("Agent"+str(i))
    #-エージェントのアクション
    #--Offer以外
    DialogueProtocols.listSystemAction=["Accept","Keep"]
    #--Offer
    for agent in DialogueProtocols.listAgentsName:
        for item1 in DialogueProtocols.listItems: 
            for item2 in DialogueProtocols.listItems:
                if item1 != item2:
                    #agentに自分のitem1を差し出して、agentのitem2を受け取る申し出。
                    DialogueProtocols.listSystemAction.append("Offer_"+agent+"_"+item1+"_"+item2)

    print "Protocol for System Action is generated." 
    print str(len(DialogueProtocols.listSystemAction))+" actions are generated."
    print DialogueProtocols.listSystemAction
    #-エージェントの状態
    DialogueProtocols.listSystemState=[]
    DialogueProtocols.listDialogStateinSpecialCase=[]
    #--OfferedTable
    for agent in DialogueProtocols.listAgentsName:
        for item1 in DialogueProtocols.listItems: 
            for item2 in DialogueProtocols.listItems:
                if item1 != item2:
                    #オファーもとのエージェント（agent）のitem1とオファーされたエージェントのitem2の交換。
                    DialogueProtocols.listSystemState.append("OfferedFrom_"+agent+"_"+item1+"_"+item2)
                    #
                    if agent != "AgentLearner":
                        DialogueProtocols.listDialogStateinSpecialCase.append("OfferedFrom_"+agent+"_"+item1+"_"+item2)
    
    #--各エージェントのアイテムの個数
    for agent in DialogueProtocols.listAgentsName: 
        for item1 in DialogueProtocols.listItems: 
            DialogueProtocols.listSystemState.append("NumItem_"+agent+"_"+item1)
            #
            DialogueProtocols.listDialogStateinSpecialCase.append("NumItem_"+agent+"_"+item1)

    print "Protocol for System State is generated." 
    print str(len(DialogueProtocols.listSystemState))+" states are generated."

    #Bias項　After analysis
    print "Bias always 1" 
    DialogueProtocols.listSystemState.append("Bias")
    #
    DialogueProtocols.listDialogStateinSpecialCase.append("Bias")
    
    
    #追加の対話状態29141205
    if ExperimentalConditions.isAddAdditionalDStoLearner:
        print "Additional dialog state is appended"
        print "1.CurrentTurn"
        DialogueProtocols.listSystemState.append("CurrentTurn")
        #
        #DialogueProtocols.listDialogStateinSpecialCase.append("CurrentTurn")
        
        print "Full offered/accepted table"
        print "Full offered/accepted table"
        for agent1 in DialogueProtocols.listAgentsName:#Offer/Acceptしたエージェント
            for agent2 in DialogueProtocols.listAgentsName:#Offer/Acceptされたエージェント
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if item1 != item2:
                            DialogueProtocols.listSystemState.append("FO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            DialogueProtocols.listSystemState.append("Accept_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #
                            #DialogueProtocols.listDialogStateinSpecialCase.append("FO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #DialogueProtocols.listDialogStateinSpecialCase.append("Accept_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
        #アクセプトのヒストリ
        print "Offer/accept History"
        for agent1 in DialogueProtocols.listAgentsName:#Offer/Acceptしたエージェント
            for agent2 in DialogueProtocols.listAgentsName:#Offer/Acceptされたエージェント
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if item1 != item2:
                            DialogueProtocols.listSystemState.append("HistoryO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            DialogueProtocols.listSystemState.append("HistoryA_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #
                            #DialogueProtocols.listDialogStateinSpecialCase.append("HistoryO_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #DialogueProtocols.listDialogStateinSpecialCase.append("HistoryA_"+agent1+"_"+agent2+"_"+item1+"_"+item2)        
        #ッリジェクトのヒストリ
        print "Reject History"
        for agent1 in DialogueProtocols.listAgentsName:#Rejectされたエージェント
            for agent2 in DialogueProtocols.listAgentsName:#したエージェント
                for item1 in DialogueProtocols.listItems:
                    for item2 in DialogueProtocols.listItems:
                        if (item1 != item2) and (agent1 != agent2):
                            DialogueProtocols.listSystemState.append("Rejected_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            #
                            if agent1 == "AgentLearner":
                                pass
                                #DialogueProtocols.listDialogStateinSpecialCase.append("Rejected_"+agent1+"_"+agent2+"_"+item1+"_"+item2)
                            
    print "OfferedFrom_TargetAgentName_itemGive_itemGiven:直前にどのTargetAgentから物々(itemGiveをあげてitemGivenをもらう)交換のOfferを受けたか。"
    print "NumItem_TargetAgentName_item:各エージェントのItemの個数"
    print "All shared DS"
    print DialogueProtocols.listSystemState
    print "DS for learner"
    print DialogueProtocols.listDialogStateinSpecialCase
    
    #実験条件のアナウンス
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
    
    #AgentLearner用の無駄の無いアクションの追加
    DialogueProtocols.listSystemActionForLearner=copy.copy(DialogueProtocols.listSystemAction)
    for elem in DialogueProtocols.listSystemAction:
        if re.search("Offer_AgentLearner_",elem) != None: 
            DialogueProtocols.listSystemActionForLearner.remove(elem)
    print "Action for learner"
    print DialogueProtocols.listSystemActionForLearner
    
    #ターン
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
        agent.learner.rewardDiscount=0.9#0.99よりも0.9の方が収束がよさげ
        agent.learner._lambda=0.99
        
    experiment = EpisodicExperiment(task, agent)
    
    #実験-学習
    if ExperimentalConditions.isLearning:
        for pol in range(ExperimentalConditions.numSystems):
            #毎回重みを初期化
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
            #記録用ファイル作成
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
                    if elem < (ExperimentalConditions.numBatch*(4.0/4.0)):#全学習を探索に使う
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
                    #num=size(reward)#対話の最後にのみrewardを与える時以外はコメントアウト
                    rew=ave/num
                    print "平均報酬@step"+str(index)+"=" + str(ave/num)
                    #
                    if ExperimentalConditions.isCalculateStatisticsOfLearner:
                        SatisticsOfLearnerInExperiment.totalNumberOfDialogue+=num
                    
                    #学習結果の出力
                    f.write("平均報酬@step"+str(index)+"="+str(ave/num)+"\n")
                    if bestCumReward < ((double)(ave)/(double)(num)) and (index >= ExperimentalConditions.numBatchStartLogging):
                        bestCumReward =((double)(ave)/(double)(num))
                        if not ExperimentalConditions.isUseLinearQ:
                            bestPolicyW=copy.deepcopy(agent.learner.module.network.params)
                        else:
                            bestPolicyW=copy.deepcopy(agent.learner._theta)
#                     #TEST
#                     print str(DialogueProtocols.totalAccept/DialogueProtocols.totalOffer)

                    #収束が悪い場合は学習打ち切り 20141208
                    if (index >= 2) and (rew < ExperimentalConditions.dBadConversion):
                        print "Learning was stopped because of bad conversion"
                        break
                    if not ExperimentalConditions.isUseLinearQ:
                        agent.learn()
                        agent.reset()#各ステップごとに初期化はすること（学習後は方策が変わるため、これまでのデータは使えない）
                        #追加　学習ごとの探索の現象　２０１４1208
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
    
    
    #実験-最適方策評価
    if ExperimentalConditions.isTest:
        #エージェントのアクションをargmax選択にする
        if not ExperimentalConditions.isUseLinearQ:
            agent.learner._explorer.epsilon=0.0
        else:
            agent.learner._behaviorPolicy=agent.learner._greedyAction
            agent.learner.batchMode=True
            agent.learning=False
        #weight.txtから重みを読み込む(ファイル形式はcsv) 
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
                print "平均報酬@step" +str(index)+"=" + str(ave/num)
                f.write("平均報酬@step"+str(index)+"="+str(ave/num)+"\n")
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
    
    
    
    