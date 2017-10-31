from autograd import grad
import autograd.numpy as np
from autograd.numpy import dot
from autograd.numpy.random import normal
from autograd.numpy.linalg import norm


class Node:
    """
	This class is the implementation of a single node in a Kalman Filter chain for PDMM
	It can calculate it's local loss and propogate messages to adjacent nodes
	"""

    def __init__(self, i, N, f, p, d):

        self.i = i
        self.x = normal(0, 1, [N, 1])
        self.f = f
        self.Neighbours = []
        self.M = {}

        self.d_T = d
        self.p = p
        self.N_max = 5000

    def finalise(self):
        for neighbour in self.Neighbours:
            neighbour.m_ij = neighbour.new_m_ij

    def get_message(self, j):
        for neighbour in self.Neighbours:
            if neighbour.j == j:
                return neighbour.m_ij

    def update_messages(self):
        # push new messages to Neighbours
        for neighbour in self.Neighbours:
            m_ji = neighbour.node.get_message(self.i)
            neighbour.new_m_ij = m_ji + (neighbour.c_ij - 2 * dot(neighbour.A_ij, self.x))

    def objective(self, x):
        obj = self.f(x)

        for neighbour in self.Neighbours:
            m_ji = neighbour.node.get_message(self.i)
            A_ij = neighbour.A_ij
            P_ij = neighbour.P_ij

            obj_int = dot(A_ij, x) - m_ji
            obj = obj + 0.5 * dot(obj_int.T, dot(P_ij, obj_int))

        return obj

    def find_minimum(self):
        # perform gradient descent on objective
        obj_grad = grad(self.objective)
        d = 1
        i = 0
        while d > self.d_T and i < self.N_max:
            g = self.p * obj_grad(self.x)
            self.x = self.x - g
            d = norm(g)
            i = i + 1
        print(i, d)

    def update(self):

        self.find_minimum()
        self.update_messages()
