# -*- coding: utf-8 -*-

import sys
import random
import numpy as np

from logger import Logger

sys.path.append('./..')

from TsutsuiNN.v191004.chain    import Chain
from TsutsuiNN.v191004.lstm     import LSTM
from TsutsuiNN.v191004.linear   import Linear
import TsutsuiNN.v191004.optimizers     as optimizers


class MyLSTM( object ) :
    
    # ログ取り用
    class LSTMLogger( Logger ) :
        def __init__(self,model=None,enable=True) :
            super(MyLSTM.LSTMLogger,self).__init__(enable)
            self.__model = model
        
        def WriteBreak( self, note ) :
            if not self.IsEnable : return
            txt = "==========" + str(note) + "=========="
            self.Logging(txt)
        
        def WriteModelInput( self, input ) :
            if not self.IsEnable : return
            input = input[0]
            txt = "model_input"
            for i in input :
                txt += ",%.08f"%(i)
            self.Logging(txt)
        
        def WriteL1Activation( self ) :
            if not self.IsEnable : return
            activation = self.__model.l1.Z[0]
            txt = "l1activation"
            for a in activation :
                txt += ",%.08f"%(a)
            self.Logging(txt)
        
        def WriteL2Activation( self ) :
            if not self.IsEnable : return
            activation = self.__model.l2.Z[0]
            txt = "l2activation"
            for a in activation :
                txt += ",%.08f"%(a)
            self.Logging(txt)
        
        def WriteL1UpwardWeight( self ) :
            if not self.IsEnable : return
            for i,weight in enumerate(self.__model.l1.upward.W.T) :
                txt = "l1upwardweight_%d"%(i)
                for w in weight :
                    txt += ",%.08f"%(w)
                self.Logging(txt)
        
        def WriteL1LateralWeight( self ) :
            if not self.IsEnable : return
            for i,weight in enumerate(self.__model.l1.lateral.W.T) :
                txt = "l1lateralweight_%d"%(i)
                for w in weight :
                    txt += ",%.08f"%(w)
                self.Logging(txt)
        
        def WriteL1Bias( self ) :
            if not self.IsEnable : return
            for i,bias in enumerate(self.__model.l1.upward.b.T):
                txt = "l1bias_%d"%(i)
                for b in bias :
                    txt += ",%.08f"%(b)
                self.Logging(txt)
        
        def WriteL2Bias( self ) :
            if not self.IsEnable : return
            for i,bias in enumerate(self.__model.l2.b.T):
                txt = "l2bias_%d"%(i)
                for b in bias :
                    txt += ",%.08f"%(b)
                self.Logging(txt)
        
        def WriteL1InputState( self ) :
            if not self.IsEnable : return
            for i,input in enumerate(self.__model.l1.a()[-1]):
                txt = "input_state_%d"%(i)
                for i in input :
                    txt += ",%.08f"%(i)
                self.Logging(txt)
        
        def WriteL1InputGateState( self ) :
            if not self.IsEnable : return
            for i,input_gate in enumerate(self.__model.l1.i()[-1]):
                txt = "input_gate_state_%d"%(i)
                for i in input_gate :
                    txt += ",%.08f"%(i)
                self.Logging(txt)
        
        def WriteL1ForgetGateState( self ) :
            if not self.IsEnable : return
            for i,forget_gate in enumerate(self.__model.l1.f()[-1]):
                txt = "forget_gate_state_%d"%(i)
                for f in forget_gate :
                    txt += ",%.08f"%(f)
                self.Logging(txt)
        
        def WriteL1OutputGateState( self ) :
            if not self.IsEnable : return
            for i,output_gate in enumerate(self.__model.l1.o()[-1]):
                txt = "output_gate_state_%d"%(i)
                for o in output_gate :
                    txt += ",%.08f"%(o)
                self.Logging(txt)
        
        def WriteL1CellState( self ) :
            if not self.IsEnable : return
            for i,cell in enumerate(self.__model.l1.c):
                txt = "memory_cell_state_%d"%(i)
                for s in cell :
                    txt += ",%.08f"%(s)
                self.Logging(txt)
        
        def WriteL2Weight( self ) :
            if not self.IsEnable : return
            for i,weight in enumerate(self.__model.l2.W.T) :
                txt = "l2weight_%d"%(i)
                for w in weight :
                    txt += ",%.08f"%(w)
                self.Logging(txt)
    
    
    def __init__( self, in_units=1, hidden_units=5, out_units=1, seed=100, log_enable=True ) :
        self.__model        = Chain( "mean_squared_error", l1=LSTM(in_units, hidden_units, "sigmoid"), l2=Linear(hidden_units, out_units, "sigmoid") )
        self.__logger       = self.LSTMLogger(self.__model,log_enable)
        self.__seed         = seed
        self.__optimizer    = optimizers.Adam()     # 最適化アルゴリズムはAdam固定
        self.__optimizer.setup( self.__model )
        
        self.InitializeParams()
    
    
    @property
    def Logger(self) :
        return self.__logger
    
    
    def InitializeParams( self ) :
        np.random.seed( self.__seed )
        
        for param in self.__model.params() :
            param[:] = np.random.uniform( -0.35, 0.35, param.shape )
    
    
    def Update( self, x_t ) :
        x = np.array(x_t, dtype=np.float32).reshape(1,len(x_t))
        y = self.__model.forward(x)
        self.Logger.WriteModelInput(x)
        self.Logger.WriteL1InputState()
        self.Logger.WriteL1InputGateState()
        self.Logger.WriteL1ForgetGateState()
        self.Logger.WriteL1OutputGateState()
        self.Logger.WriteL1CellState()
        self.Logger.WriteL1Activation()
        self.Logger.WriteL2Activation()
        return y
    
    
    def Train( self, series, answers, log_note="train" ) :
        if len(series) == 0 :
            print "trainig series is empty @ func Train"
            return None
        
        self.__model.Clear()
        self.Logger.WriteBreak(log_note)
        
        losses      = None
        for i,(x_t, a_t) in enumerate( zip(series, answers) ) :
            self.Update(x_t)
            a = np.array(a_t, dtype=np.float32).reshape(1,len(a_t))
            if losses is None :
                losses  = self.__model.loss(a)
            else :
                losses += self.__model.loss(a)
        self.__LearnFromLoss()
        
        self.Logger.WriteL1UpwardWeight()
        self.Logger.WriteL1LateralWeight()
        self.Logger.WriteL2Weight()
        self.Logger.WriteL1Bias()
        self.Logger.WriteL2Bias()
        
        return losses / len(series)
    
    
    def __LearnFromLoss( self ) :
        self.__model.backward()
        self.__optimizer.update()
    
    
    def Predict( self, series, log_note="predict" ) :
        if len(series) == 0 :
            print "input series is empty @ func Predict"
            return None
        
        self.__model.Clear()
        self.Logger.WriteBreak(log_note)
        
        ret = []
        for x_t in series :
                ret.append( self.Update(x_t).reshape(-1,) )
        return ret


class BL_LSTM(MyLSTM) :
    def __init__(self, in_units=2, hidden_units=10, out_units=2, seed=100):
        super(BL_LSTM,self).__init__( in_units, hidden_units, out_units, seed)
    
    
    def Train(self, training_series,epoch, output_directory_path) :
        total_loss  = 0.0
        for i,tr_series in enumerate(training_series) :
            series, answers = tr_series.Get()
            total_loss += super(BL_LSTM,self).Train( series,answers,"series%03d"%(i) )
        self.Logger.Output( output_directory_path+"/%s_epoch.csv"%epoch )
        self.Logger.Clear()
        return total_loss / len(training_series)
    
    
    def Predict(self, predict_series, output_file_path) :
        ret = dict( result=[], predicted=[], answers=[] )
        for i,pr_series in enumerate(predict_series) :
            series, answers = pr_series.Get()
            predicted       = super(BL_LSTM,self).Predict( series,"series%03d"%(i) )
            ret["result"].append( [i,predicted[-1],answers[-1]] )
            ret["predicted"].append( predicted )
            ret["answers"].append( answers )
        self.Logger.Output( output_file_path )
        self.Logger.Clear()
        return ret
