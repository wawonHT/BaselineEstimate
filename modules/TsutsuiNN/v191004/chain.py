# -*- coding: utf-8 -*-

import sys
import numpy as np

from linear     import Linear
from lstm       import LSTM
from function   import Error


class Chain( object ) :
    def __init__( self, error_function=None, **links ) :
        self.__outputs  = None
        
        # セーブ用配列
        self.__y        = []
        self.__targets  = []
        
        if error_function is not None :
            self.__error_func   = Error( error_function )
        else :
            self.__error_func   = None
        
        self.linknames  = []
        for name,value in links.items() :
            self.__AddLink( name, value )
            self.linknames.append( name )
        self.linknames = sorted( self.linknames )
    
    
    def __AddLink( self, name, value ) :
        if name in self.__dict__ :
            raise AttributeError( "cannot register a new link %s: attribute exists" %name )
        if not isinstance( value, ( Linear, LSTM, Chain ) ) :
            raise TypeError( "can register Linear object or LSTM object or Chain object" )
        
        setattr( self, name, value )
    
    
    # ニューラルネットワークの重みとバイアスを返す
    def params( self ) :
        ret = []
        for name in self.linknames :
            ret += self.__dict__[name].params()
        return ret
    
    
    def forward( self, inputs ) :
        if not isinstance( inputs, np.ndarray ) :
            print "inputs type is not np.ndarray @Chain"
            print "inputs : ", inputs
            exit()
        
        for name in self.linknames :
            inputs = self.__dict__[name].forward( inputs )
        self.__outputs  = inputs
        self.__y.append( self.__outputs )
        return self.__outputs
    
    
    def loss( self, targets ) :
        ERROR   = self.__error_func.Get()
        loss    = ERROR( self.__outputs, targets )
        self.__targets.append( targets )
        return loss
    
    
    def backward( self, W_upper=None, delta_upper=None, value=None ) :
        if self.__error_func is not None :
            ERROR_DIFF  = self.__error_func.Differentiate().Get()
            value       = map( lambda y,targets: ERROR_DIFF( y, targets ), self.__y,self.__targets )
        
        for name in self.linknames[::-1] :
            if W_upper is None and delta_upper is None :
                W_upper, delta_upper = self.__dict__[name].backward( W_upper, delta_upper, value )
            else :
                W_upper, delta_upper = self.__dict__[name].backward( W_upper, delta_upper )
        return W_upper, delta_upper
    
    
    # ニューラルネットワークの重みの勾配とバイアスの勾配を返す
    def grads( self ) :
        ret = []
        for name in self.linknames :
            ret += self.__dict__[name].grads()
        return ret
    
    
    def Clear( self ) :
        self.__y        = []
        self.__targets  = []
        for name in self.linknames :
            self.__dict__[name].Clear()