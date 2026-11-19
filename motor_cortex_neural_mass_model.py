import numpy as np
import networkx as nx
import itertools
import random
import pandas as pd
import matplotlib.pyplot as plt

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
    def __init__(self, max_fr, r, v0):
        self.max_fr = max_fr
        self.r = r
        self.v0 = v0

    def set_sigmoid(self):
        self.sigm = make_sigmoid_func(self.max_fr, self.r, self.v0)


# Create an instance of a sigmoid function.
def make_sigmoid_func(max_fr, r, v0):
    """:Parameters:
        max_fr : maximum firing rate. 
        r      : steepness of sigmoidal transmission.
        v0     : the PSP yeilding 50% firing rate.
    """
    def sigmoid(v):
        """:Parameters:
            v: avg. membrane potential of neural population. 
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
        pass

def make_integrator(T, dt):
    """
    Makes forward euler integrator.
    :Parameters:
    T: time window to integrate over.
    dt: timestep size.
    """
    # Define parameters for (fixed step-size) integration.
    num_time_steps = int(np.round(T / dt))
        
    def forward_euler(F:object, t0, x0):
        '''
        Solves the ode dx/dt = f(t,x) using the Explicit (Foward) Euler Method.

        :Parameters:
        F: Function representing the ODE // f(t,x)
        t0: Initial time.
        x0: Initial value of x at t0.

        :Output: 
        X : matrix of state vectors for each timestep.
        '''
        # Define the dimension of state sapce to be 6 = (2) * (3) \\ (DEs per PSP block ) \times ( # of neural populations)
        state_space_dim = 2 * 3 
        # Initialize foward state matrix. \\ There is a state for each time step, ie x0, x1, x2, ...
        X = np.zeros((state_space_dim, num_time_steps)) # \\ The columns are the state vectors at time t.

        # Solve IVP for forward Euler method.
        # Initialize the time step index. \\ Note that it is not necessarily t0=0. 
        n = 0 
        X[:,n] = x0 # \\ This is the initial state x0 at initial time t0 (also referred to as n =0).
        # We iterate forward by each time step.
        for t_step in np.arange(start=t0, stop=t0+ dt*(num_time_steps-1), step=dt):
            # Next state vector equals current state vector x_i+1 plus timestep \times derivative at x_i
            # X[:, t_idx+1] = (np.eye(state_space_dim) + dt*A) @ X[:, t_idx] \\ alternative formulation using matrix multiplication. "A" is the system matrix.
            X[:, n+1] = X[:, n] + dt*F(t_step, X[:, n])
            # Increment the time step index.
            n += 1; print(f"timestep: {n} / {num_time_steps} completed.")
        
        return X

    return forward_euler


def make_ode_func(sigm0, sigm1, sigm2, A, B, a, b, c1, c2, c3, c4):
    """
    Makes a function with hyperparams selected.

    ::hyperparameters::
    sigm0/1/2: sigmoid function for the three populations PC, EiP, IiP
    A: maximum amplitude of excitatory PSP (EPSP).
    B: maximum amplitude of inhibitory PSP (IPSP).
    a: for EPSP, lumped representation of the sum of the reciprocal of the time constant of passive membrande and all other spatially distributed delays in the dendritic network.
    b: for IPSP, ...
    c1/2/3/4: connectivity constants.
    """

    def F(t_step, Y_t, p_t=0):
        """ 
        Evaluates the odes function given a state vector.

        ::Parameters::
        t_step : actual time t.
        Y_t : state vector at time step t.
        p_t : (?) external stimuli.
        """

        # Decompose the state vector Y_t into its components y0, y1, ... .
        # y0, y1, y2 represent the outputs (membrane potential) of the three PSP blocks - pyramidal, EiP, IiP, respectively.
        # y3, y4, y5 represent the change of these outputs.
        y0, y1, y2, y3, y4, y5 = Y_t

        # Compute the sigmoidal outputs of each neural population.
        # x0, x1, x2 represent the inputs (firing rates) into the three PSP blocks.
        x0 = sigm0(y1 - y2)
        x1 = sigm1(c1*y0)
        x2 = sigm2(c3*y0)

        # Compute the derivatives of the state vector components.
        dy0dt = y3
        dy1dt = y4
        dy2dt = y5
        dy3dt = A*a*x0 - 2*a*y3 - a**2*y0
        dy4dt = A*a*(p_t + c2*x1) - 2*a*y4 - a**2*y1
        dy5dt = B*b*c4*x2 - 2*b*y5 - b**2*y2
        
        return np.array([dy0dt, dy1dt, dy2dt, dy3dt, dy4dt, dy5dt])

    return F





if __name__ == '__main__':

    # Assign the hyperparameter values ...
    # for the pyramidal cells.
    max_fr_PC, r_PC, v0_PC = np.random.uniform(0.0, 1.0, 3) # gives three random values between 0 and 1.
    # for Excitatory interneurons.
    max_fr_EiP, r_EiP, v0_EiP = np.random.uniform(0.0, 1.0, 3)
    # for Inhibitory interneurons.
    max_fr_IiP, r_IiP, v0_IiP = np.random.uniform(0.0, 1.0, 3)

    # Initiate our neural populations and assign their respective sigmoid functions.
    # -Pyramidal cells.
    PC = JansenRitNeuralPopulation(max_fr=max_fr_PC, r=r_PC, v0=v0_PC)
    PC.set_sigmoid()
    # -Excitatory interneurons.
    EiP = JansenRitNeuralPopulation(max_fr=max_fr_EiP, r=r_EiP, v0=v0_EiP)
    EiP.set_sigmoid()
    # -Inhibitory interneurons.
    IiP = JansenRitNeuralPopulation(max_fr=max_fr_IiP, r=r_IiP, v0=v0_IiP)
    IiP.set_sigmoid()

    # Intitiate our Jansen-Rit Model.
    snn = JansenRitModel()
    print(list(snn.nodes))
    print(list(snn.edges))
    state_space_dim = 6

    # Assign connectivity constants.
    [c1, c2, c3, c4] = np.random.uniform(0.1, 1.0, 4)
    snn['pyramidal']['excitatory']['connectivity'] = c1
    snn['pyramidal']['inhibitory']['connectivity'] = c3
    snn['excitatory']['pyramidal']['connectivity'] = c2
    snn['inhibitory']['pyramidal']['connectivity'] = c4

    # Assign the max ampltitudes and (?) time delays.
    A, B, a, b = np.random.uniform(0.1, 1.0, 4)
    # Formulate the ode.
    F = make_ode_func(sigm0=PC.sigm, sigm1=EiP.sigm, sigm2=IiP.sigm, A=A, B=B, a=a, b=b, c1=c1, c2=c2, c3=c3, c4=c4)
    
    # Assign hyperparams for integrator.
    T = 10 # Entire time window to integrate over.
    dt = 0.1 # size of time steps.
    # Instantiate the forward euler method.
    forward_euler = make_integrator(T, dt)

    # Run the forward euler method and return state space trajectory.
    t0 = 0 # initial time.
    Y = forward_euler(F, t0, x0=np.zeros(state_space_dim,))
    print(pd.DataFrame(Y))


    """ ----- Plotting State Space Trajectory ----- """
    # Plot the state space trajectory.
    # To make stuff easier, define variables.
    num_time_steps = int(np.round(T / dt))
    t = np.arange(start=t0, stop=t0+ dt*(num_time_steps), step=dt)

    plt.figure(figsize=(12, 8))

    # Phase portrait: x1 vs x2 (pyramidal population)
    plt.subplot(2, 3, 1)
    plt.plot(Y[0, :], Y[1, :], 'b-', linewidth=1)
    plt.xlabel('y0 (Pyramidal Potential)')
    plt.ylabel('dy0/dt')
    plt.title('Phase Portrait: y0 vs dy0/dt')

    # Phase portrait: x3 vs x4 (excitatory interneuron)
    plt.subplot(2, 3, 2)
    plt.plot(Y[2, :], Y[3, :], 'g-', linewidth=1)
    plt.xlabel('y1 (Excitatory Interneuron)')
    plt.ylabel('dy1/dt')
    plt.title('Phase Portrait: y1 vs dy1/dt')

    # Phase portrait: x5 vs x6 (inhibitory interneuron)
    plt.subplot(2, 3, 3)
    plt.plot(Y[4, :], Y[5, :], 'r-', linewidth=1)
    plt.xlabel('y2 (Inhibitory Interneuron)')
    plt.ylabel('dy2/dt')
    plt.title('Phase Portrait: y2 vs dy2/dt')

    # Time series of all states
    plt.subplot(2, 3, 4)
    plt.plot(t, Y[0, :], 'b-', label='y0 (Pyramidal)')
    plt.plot(t, Y[2, :], 'g-', label='y1 (Excitatory)')
    plt.plot(t, Y[4, :], 'r-', label='y2 (Inhibitory)')
    plt.xlabel('Time (s)')
    plt.ylabel('Potential')
    plt.title('Time Series')
    plt.legend()

    # 3D state space trajectory (x1, x3, x5)
    ax = plt.subplot(2, 3, 5, projection='3d')
    ax.plot(Y[0, :], Y[2, :], Y[4, :], 'k-', linewidth=0.8)
    ax.set_xlabel('y0')
    ax.set_ylabel('y1')
    ax.set_zlabel('y2')
    ax.set_title('3D State Space Trajectory')

    # Optional: Add a scatter of initial point
    ax.scatter(Y[0, 0], Y[1, 0], Y[2, 0], color='red', s=50)

    plt.tight_layout()
    plt.show()