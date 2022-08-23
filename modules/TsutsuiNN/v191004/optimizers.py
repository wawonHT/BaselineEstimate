
import sys
import numpy as np


class SetModel( object ) :
    def __init__( self ) :
        self.model  = None
    
    def setup( self, model ) :
        self.model  = model


class SGD( SetModel ) :
    def __init__( self, epsilon=0.01 ) :
        self.epsilon    = epsilon
    
    def update( self ) :
        for name in self.model.linknames :
            link = self.model.__dict__[name]
            
            for param, grad in zip( link.params(), link.grads() ) :
                param[:] = param - self.epsilon * grad


class Adam( SetModel ) :
    def __init__( self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10**(-8) ) :
        self.alpha      = alpha
        self.beta1      = beta1
        self.beta2      = beta2
        self.epsilon    = epsilon
        self.t          = 0
        
        self.save_moments = []
    
    def update( self ) :
        self.t += 1
        
        if self.t == 1 :
            for i in range( len(self.model.params()) ) :
                self.save_moments.append( ApplyAdam() )
        
        for i, (param, grad) in enumerate( zip(self.model.params(), self.model.grads()) ) :
            param[:] = self.save_moments[i]( param, grad )


class ApplyAdam( object ) :
    def __init__( self, alpha=Adam().alpha, beta1=Adam().beta1, beta2=Adam().beta2, epsilon=Adam().epsilon ) :
        self.alpha      = alpha
        self.beta1      = beta1
        self.beta2      = beta2
        self.epsilon    = epsilon
        self.t          = 0
        self.m          = 0
        self.v          = 0
    
    def __call__( self, param, grad ) :
        self.t += 1
        self.m  = self.beta1 * self.m + ( 1-self.beta1 ) * grad
        self.v  = self.beta2 * self.v + ( 1-self.beta2 ) * grad**2
        hat_m   = self.m / ( 1-self.beta1**self.t )
        hat_v   = self.v / ( 1-self.beta2**self.t )
        param   = param - self.alpha * hat_m / ( hat_v**0.5+self.epsilon )
        
        return param