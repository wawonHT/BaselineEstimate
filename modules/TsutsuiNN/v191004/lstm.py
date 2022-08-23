# -*- coding: utf-8 -*-

import sys
import numpy as np

from function       import Activation
from linear         import Linear


class LSTM( object ) :
    def __init__( self, in_units, out_units, activation_function="identity" ) :
        self.__in_units         = in_units
        self.__out_units        = out_units
        self.__sigmoid          = Activation( "sigmoid" )
        self.__tanh             = Activation( "tanh" )
        self.__activation_func  = Activation( activation_function )
        self.upward             = Linear( in_units, out_units*4 )
        self.lateral            = Linear( out_units, out_units*4, nobias=True )
        self.c                  = None
        self.h                  = None
        self.Z                  = None
        
        # セーブ用配列
        # a:input, i:input gate, f:forget gate, o:output gate
        # 頭にUをつけたものはユニットの入力値を表す
        # dAはinput，inputgate，forgetgate，outputgateのデルタを順に横方向に連結した行列
        self.__Ua       = []
        self.__Ui       = []
        self.__Uf       = []
        self.__Uo       = []
        self.__a        = []
        self.__i        = []
        self.__f        = []
        self.__o        = []
        self.__c        = []
        self.__h        = []
        self.__dA       = []
    
    
    def params( self ) :
        return self.upward.params() + self.lateral.params()
    
    
    # メモリーセルの値と隠れ状態を初期化
    def InitializeState( self, N ) :
        self.c = np.zeros(( N, self.__out_units ))
        self.h = np.zeros(( N, self.__out_units ))
    
    
    def forward( self, inputs ) :
        if not isinstance( inputs, np.ndarray ) :
            print "inputs type is not np.ndarray @LSTM"
            print "inputs : ", inputs
            exit()
        
        ACTIVATION  = self.__activation_func.Get()
        sigmoid     = self.__sigmoid.Get()
        tanh        = self.__tanh.Get()
        
        if self.c is None and self.h is None :
            self.InitializeState( inputs.shape[0] )
            self.__c.append( self.c )
        
        
        # ユニットの入力値を計算
        U = self.upward.forward( inputs ) + self.lateral.forward( self.h )
        for i in range( 0, self.__out_units*4, 4 ) :
            if i == 0 :
                Ua = U[:,i:i+1]
                Ui = U[:,i+1:i+2]
                Uf = U[:,i+2:i+3]
                Uo = U[:,i+3:i+4]
            else :
                Ua = np.hstack(( Ua, U[:,i:i+1] ))
                Ui = np.hstack(( Ui, U[:,i+1:i+2] ))
                Uf = np.hstack(( Uf, U[:,i+2:i+3] ))
                Uo = np.hstack(( Uo, U[:,i+3:i+4] ))
        
        
        self.__Ua.append( Ua )
        self.__Ui.append( Ui )
        self.__Uf.append( Uf )
        self.__Uo.append( Uo )
        
        
        # ユニットの出力値を計算
        a = tanh( Ua )
        i = sigmoid( Ui )
        f = sigmoid( Uf )
        o = sigmoid( Uo )
        
        
        
        self.__a.append( a )
        self.__i.append( i )
        self.__f.append( f )
        self.__o.append( o )
        
        
        # メモリーセルの値と現在の隠れ状態を計算
        self.c  = f * self.c + a * i
        self.h  = o * tanh( self.c )
        
        
        self.__c.append( self.c )
        self.__h.append( self.h )
        
        self.Z = ACTIVATION( self.h )    # ノードの出力値を計算
        return self.Z
    
    
    def backward( self, W_upper, delta_upper, ERROR_DIFF_VALUE=None ) :
        ACTIVATION_DIFF = self.__activation_func.Differentiate().Get()
        sigmoid         = self.__sigmoid.Get()
        sigmoid_diff    = self.__sigmoid.Differentiate().Get()
        tanh            = self.__tanh.Get()
        tanh_diff       = self.__tanh.Differentiate().Get()
        
        N   = delta_upper[0].shape[0]               # サンプル数
        dA  = np.zeros(( N, self.__out_units*4 ))   
        dc  = np.zeros(( N, self.__out_units ))
        
        # 活性化関数のデルタを計算
        if W_upper is None and delta_upper is None :
            delta_upper = map( lambda value, U: value * ACTIVATION_DIFF(U), ERROR_DIFF_VALUE, self.__h )
        else :
            delta_upper = map( lambda U, d: ACTIVATION_DIFF(U) * np.dot( d, W_upper.T ), self.__h, delta_upper )
        # 各ユニットのデルタを計算
        for i in range( len(delta_upper)-1, -1, -1 ) :
            epsilon = ( delta_upper[i] + np.dot(dA, self.lateral.W.T) )
            do      = sigmoid_diff( self.__Uo[i] ) * tanh( self.__c[i+1] ) * epsilon
            if i == len(delta_upper)-1 :
                dc  = dc + tanh_diff( self.__c[i+1] ) * self.__o[i] * epsilon
            else :
                dc  = self.__o[i+1] * dc + tanh_diff( self.__c[i+1] ) * self.__o[i] * epsilon
            da  = tanh_diff( self.__Ua[i] ) * self.__i[i] * dc
            di  = sigmoid_diff( self.__Ui[i] ) * self.__a[i] * dc
            df  = sigmoid_diff( self.__Uf[i] ) * self.__c[i] * dc
            
            for j in range( self.__out_units ) :
                if j == 0 :
                    dA = np.hstack( (da[:,j:j+1], di[:,j:j+1], df[:,j:j+1], do[:,j:j+1]) )
                else :
                    dA = np.hstack( (dA, da[:,j:j+1], di[:,j:j+1], df[:,j:j+1], do[:,j:j+1]) )
            
            self.__dA.append( dA )
        
        self.__dA   = self.__dA[::-1]
        return [ self.upward.W, self.__dA ]
    
    
    def grads( self ) :
        self.upward.delta   = self.__dA
        self.lateral.delta  = self.__dA
        return self.upward.grads() + self.lateral.grads()
    
    
    def Ua( self ) :
        return self.__Ua
    
    def Ui( self ) :
        return self.__Ui
    
    def Uf( self ) :
        return self.__Uf
    
    def Uo( self ) :
        return self.__Uo
    
    def a( self ) :
        return self.__a
    
    def i( self ) :
        return self.__i
    
    def f( self ) :
        return self.__f
    
    def o( self ) :
        return self.__o
    
    
    def Clear( self ) :
        self.c      = None
        self.h      = None
        self.__Ua   = []
        self.__Ui   = []
        self.__Uf   = []
        self.__Uo   = []
        self.__a    = []
        self.__i    = []
        self.__f    = []
        self.__o    = []
        self.__c    = []
        self.__h    = []
        self.__dA   = []
        
        self.upward.Clear()
        self.lateral.Clear()