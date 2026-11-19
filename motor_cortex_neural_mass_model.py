import numpy as np
import networkx as nx
import itertools
import random


""" ------- Description -------- """
""" This code base is for a neural mass model (Jansen and Rit-based architecture)
    representing the motor cortex. 
"""

class JansenRitModel(nx.DiGraph):
    # Initialize the Jansen-Rit model.
    def __init__(self):
        # Initialize DiGraph.
        super().__init__(self)
        # Create an all-to-all connected network between the three neural populations- pyramidal cells, excitatory cells, and inhibitory cells.
        self.add_edges_from(itertools.permutations(['pyramidal', 'excitatory', 'inhibitory'], r=2))

class JansenRitNeuralPopulation():
    # Each neural population consists of two blocks: 
    # 1- Postsynaptic membrane potential (PSP) block. \\ Transforms the avg. pulse density of incoming action potentials into an avg. postsynaptic membrane potential.
    # 2- Sigmoid block. \\ Transforms avg. membrane potential of a neural population into an avg. pulse density.
    def __init__(self, sigm:object):
        self.psp_block = None
        self.sigm = sigm


# Create an instance of a sigmoid function.
def make_sigmoid_func(max_fr, r, v0):
    """
    :Parameters:
        max_fr : maximum firing rate. 
        r      : steepness of sigmoidal transmission.
        v0     : the PSP yeilding 50% firing rate.
    """
    def sigmoid(v):
        """
        :Parameters:
            v : avg. membrane potential of neural population. 
        """
        return 2*max_fr/(1+ np.exp(r*(v0 - v)))
    return sigmoid


def forward_euler_old(T, dt, dts, vs, xs, t0, out_buffer):
    '''
    :Parameters:
    T: time window to integrate over.
    dt: step size
    dts: a set of steps (So, dts >> dt ?)
    vs:
    xs:
    t0: initial time value. 

    '''
    # Define parameters for (fixed step-size) integration.
    num_time_steps = int(np.round(T / dt))
    # Initialize foward state matrix. \\ There is a state for each time step, ie x0, x1, x2, ...
    x_foward_states = np.zeros((2, num_time_steps)) # \\ The columns are the state vectors at time t.
    # Initialize the time step index. \\ Note that it is not necessarily t0=0. 
    t_idx = 0 # 

    # Solve IVP for forward Euler method.
    x_foward_states[:,t_idx] = x0 # \\ This is the initial state x0 at initial time t0 (also referred to as t_idx =0).
    # We iterate forward by each time step.
    for time_step in range(t0, num_time_steps+t0):

        
def forward_euler(F:object, t0, x0, T, dt):
    '''
    Solves the ode dx/dt = f(t,x) using the Explicit (Foward) Euler Method.

    :Parameters:
    F: Function representing the ODE // f(t,x)
    t0: Initial time.
    x0: Initial value of x at t0.
    T: time window to integrate over.
    dt: timestep size.
    '''

    # Define parameters for (fixed step-size) integration.
    num_time_steps = int(np.round(T / dt))
    # Define the dimension of state sapce to be 6 = (2) * (3) \\ (DEs per PSP block ) \times ( # of neural populations)
    state_space_dim = 2 * 3 
    # Initialize foward state matrix. \\ There is a state for each time step, ie x0, x1, x2, ...
    X = np.zeros((state_space_dim, num_time_steps)) # \\ The columns are the state vectors at time t.

    # Solve IVP for forward Euler method.
    # Initialize the time step index. \\ Note that it is not necessarily t0=0. 
    n = 0 
    X[:,n] = x0 # \\ This is the initial state x0 at initial time t0 (also referred to as n =0).
    # We iterate forward by each time step.
    for t_step in np.arange(start=t0, stop=t0+ dt*num_time_steps, step=dt):
        # Next state vector equals current state vector x_i+1 plus timestep \times derivative at x_i
        # X[:, t_idx+1] = (np.eye(state_space_dim) + dt*A) @ X[:, t_idx] \\ alternative formulation using matrix multiplication. "A" is the system matrix.
        X[:, n+1] = X[:, n] + dt*F(t_step, X[:, n])


def F(X_t, sigm:object):
    # Decompose the state vector into its components.
    x0, x1, x2, x3, x4, x5 = X_t

    # Compute the sigmoidal outputs of each neural population.
    b1 = sigm(x1 - x2)
    b2 = sigm(c1*x0)
    b3 = sigm(c3*x0)

    # Compute the derivatives of the state vector components.
    dx0dt = x3
    dx1dt = x4
    dx2dt = x5
    dx3dt = A*a*b1 - 2*a*x3 - a**2*x0
    dx4dt = A*a*(p_t + c2*b2) - 2*a*x4 - a**2*x1
    dx5dt = B*b*c4*b3 - 2*b*x5 - b**2*x2
    
    return np.array([dx0dt, dx1dt, dx2dt, dx3dt, dx4dt, dx5dt])





if __name__ == '__main__':

    # Initiate our neural populations.
    # -Pyramidal cells.
    snn1 = JansenRitNeuralPopulation(sigm=make_sigmoid_func(max_fr_1, r_1, v0_1))
    # -Excitatory cells.
    snn2 = JansenRitNeuralPopulation(sigm=make_sigmoid_func(max_fr_2, r_2, v0_2))
    # -Inhibitory cells.
    snn3 = JansenRitNeuralPopulation(sigm=make_sigmoid_func(max_fr_3, r_3, v0_3))

    # Intitiate our Jansen-Rit Model.
    snn = JansenRitModel()
    print(list(snn.nodes))
    print(list(snn.edges))

    # Assign connectivity constants.
    [c1, c2, c3, c4] = np.random.uniform(0.0, 1.0, 4)
    print([c1, c2, c3, c4])
    snn['pyramidal']['excitatory']['connectivity'] = c1
    snn['pyramidal']['inhibitory']['connectivity'] = c3
    snn['excitatory']['pyramidal']['connectivity'] = c2
    snn['inhibitory']['pyramidal']['connectivity'] = c4