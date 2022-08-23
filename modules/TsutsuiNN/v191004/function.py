# -*- coding: utf-8 -*-

import inspect
import sys
import numpy as np


class GetFunction( object ) :
    def __init__( self, function_name ) :
        self.name = function_name
    
    def Get( self ) :
        for name,value in inspect.getmembers( self, inspect.ismethod ) :
            if name == self.name :
                return value
        
        raise AttributeError( "%s doesnot exist. select from %s" %( self.name, inspect.getmembers( self, inspect.ismethod ) ) )
        exit()


class Activation( GetFunction ) :
    def __init__( self, function_name ) :
        super( Activation, self ).__init__( function_name )
    
    def Differentiate( self ) :
        return Differential( self.name )
    
    def identity( self, inputs ) :
        return inputs
    
    def sigmoid( self, inputs ) :
        ret = 1.0 / ( 1.0+np.exp(-inputs) )
        return ret
    
    def tanh( self, inputs ) :
        return np.tanh( inputs )
    
    def relu( self, inputs ) :
        return np.where( inputs<0, 0, inputs )


class Error( GetFunction ) :
    def __init__( self, function_name ) :
        super( Error, self ).__init__( function_name )
    
    def Differentiate( self ) :
        return Differential( self.name )
    
    def squared_error( self, inputs, targets ) :
        ret = np.sum( (inputs - targets)**2 )
        return ret
    
    def mean_squared_error( self, inputs, targets ) :
        ret = np.average( (inputs - targets)**2 )
        return ret


class Differential( GetFunction ) :
    def __init__( self, function_name ) :
        super( Differential, self ).__init__( function_name )
    
    def identity( self, inputs ) :
        ret = np.ones(inputs.shape)
        return ret
    
    def sigmoid( self, inputs ) :
        f   = Activation( "sigmoid" ).Get()( inputs )
        ret = f * ( 1-f ) 
        return ret
    
    def tanh( self, inputs ) :
        f   = Activation( "tanh" ).Get()( inputs )
        ret = 1 - f * f
        return ret
    
    def relu( self, inputs ) :
        return np.where( inputs<0, 0, 1 )
    
    def squared_error( self, inputs, targets ) :
        ret = 2 * ( inputs - targets )
        return ret
    
    def mean_squared_error( self, inputs, targets ) :
        ret = ( 2 / inputs.size ) * ( inputs - targets )
        return ret