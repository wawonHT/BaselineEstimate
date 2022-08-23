# -*- coding: utf-8 -*-

import sys
import numpy as np

from function       import Activation


class Linear( object ) :
    def __init__( self, in_units, out_units, activation_function="identity", nobias=False ) :
        self.__in_units         = in_units
        self.__out_units        = out_units
        self.__activation_func  = Activation(activation_function)
        self.__nobias           = nobias
        self.Z                  = None
        
        # セーブ用配列
        self.__U        = []
        self.__Z_lower  = []
        self.delta      = []
        
        self.CreateParams()
    
    
    def CreateParams( self ) :
        self.W = np.empty(( self.__in_units, self.__out_units ))
        if not self.__nobias :
            self.b = np.empty(( 1, self.__out_units ))
        else :
            self.b = None
    
    
    def params( self ) :
        if not self.__nobias :
            return [ self.W, self.b ]
        else :
            return [ self.W ]
    
    
    def forward( self, inputs ) :
        if not isinstance( inputs, np.ndarray ) :
            print "inputs type is not np.ndarray @Linear"
            print "inputs : ", inputs
            exit()
        
        Z_lower     = inputs
        ACTIVATION  = self.__activation_func.Get()
        
        
        # ノードの入力値を計算
        if not self.__nobias :
            ones    = np.ones(( 1, Z_lower.shape[0] ))
            try :
                U   = np.dot( Z_lower, self.W ) + np.dot( ones.T, self.b )
            except :
                print "lower_Z : ", Z_lower
                print "W : ", self.W
                print "ones.T : ", ones.T
                print "b : ", self.b
                exit()
        else :
            try :
                U    = np.dot(  Z_lower, self.W )
            except :
                print "lower_Z : ", Z_lower
                print "W : ", self.W
                exit()
        
        
        self.__U.append( U )
        self.__Z_lower.append( Z_lower )
        
        self.Z = ACTIVATION( U )     # ノードの出力値を計算
        return self.Z
    
    
    def backward( self, W_upper, delta_upper, ERROR_DIFF_VALUE=None ) :
        ACTIVATION_DIFF = self.__activation_func.Differentiate().Get()
        if W_upper is None and delta_upper is None :
            self.delta  = map( lambda value, U: value * ACTIVATION_DIFF(U), ERROR_DIFF_VALUE, self.__U )
        else :
            self.delta  = map( lambda U, d: ACTIVATION_DIFF(U) * np.dot( d, W_upper.T ), self.__U, delta_upper )
        return [ self.W, self.delta ]
    
    
    def grads( self ) :
        Ns  = self.delta[0].shape[0]    # サンプル数
        Nt  = len(self.delta)           # フレーム数
        
        # 重みの勾配を計算
        dW  = map( lambda Z_lower, d: np.dot( Z_lower.T, d ) / Ns, self.__Z_lower, self.delta )
        dW  = sum(dW) / Nt
        
        # バイアスの勾配を計算
        if not self.__nobias :
            ones    = np.ones(( 1, Ns ))
            db      = map( lambda d: np.dot( ones, d ) / Ns, self.delta )
            db      = sum(db) / Nt
            return [ dW, db ]
        else :
            return [ dW ]
    
    
    def U( self ) :
        return self.__U
    
    
    def Clear( self ) :
        self.__U        = []
        self.__Z_lower  = []
        self.delta      = []