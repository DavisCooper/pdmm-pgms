import autograd.numpy as np
from autograd.numpy import dot
from autograd.numpy.random import normal

class ChainNode:
    """
    This class is the implementation of a single node in a Kalman Filter chain for PDMM
    It can calculate it's local loss and propogate messages to adjacent nodes
    """
    def __init__(self,N,E,a,y,i):

    	self.x = normal(0,1,[N,1])
        self.E = E
        self.a = a
        self.y = y
        self.Neighbours = []

    def update(self):
    	# perform gradient descent on x
    	# update new messages

	def finalise(self):
		# push new messages to Neighbours		


        
    