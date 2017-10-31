import autograd.numpy as np     
from autograd.numpy import dot
from autograd.numpy.random import normal

class Neighbour:
	"""
	This class contains the information for a neighbouring node to a chain node
	i - index
	A_i|j - adjacent matrix
	m_i->j - message
	c_ij - constant
	P_ij - weight matrix
	"""

	def __init__(self,node,j,A_ij,c_ij,P_ij,N):
		self.node = node
		self.j = j
		self.A_ij = A_ij
		self.c_ij = c_ij
		self.P_ij = P_ij
		self.m_ij = normal(0,1,[N,1])
		self.new_m_ij = normal(0,1,[N,1])	



