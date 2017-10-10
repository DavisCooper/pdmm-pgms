from autograd import grad
import autograd.numpy as np     
from autograd.numpy import dot
from autograd.numpy.random import normal
from autograd.numpy.linalg import norm

class ChainNode:
	"""
	This class is the implementation of a single node in a Kalman Filter chain for PDMM
	It can calculate it's local loss and propogate messages to adjacent nodes
	"""

	def __init__(self,i,N,E,a):

		self.i = i
		self.x = normal(0,1,[N,1])
		self.E = E
		self.a = a
		self.Neighbours = []
		self.M = {}

		self.d_T = 1e-10
		self.p = 1e-3
		self.N_max = 1000

	def update(self):

		find_minimum() 
		update_messages()

	def finalise(self):
		for neighbour in self.Neighbours:
			neighbour.m_ij = neighbour.new_m_ij	

	def get_message(self,j):
		for neighbour in self.Neighbours:
			if(neighbour.j==j):
				return neighbour.m_ij


	def find_minimum(self):
		#perform gradient descent on objective
		obj_grad = grad(objective)
		d = 1
		i = 0
		while(d>d_T and i < N_max):
			g = p*obj_grad(self.x)
			x = x - g
			d = norm(g)
			i = i+1

	def update_messages(self):
		#push new messages to Neighbours
		for neighbour in self.Neighbours:
			m_ji = neighbour.node.get_message(i)
			neighbour.new_m_i = m_ji  + (neighbour.c_ij - 2*dot(neighbour.A_ij,self.x))

	def objective(self,x):
		obj = 0.5*dot(x.T,dot(self.E,x)) - dot(self.a.T,x)
		
		for neighbour in self.Neighbours:
			m_ji = neighbour.node.get_message(i)
			A_ij = neighbour.A_ij
			P_ij = neighbour.P_ij

			obj_int = dot(A_ij,x) - m_ji
			obj = obj + 0.5*dot(f_int.T,dot(P_ij,f_int)) 
		
		return obj

		
	
