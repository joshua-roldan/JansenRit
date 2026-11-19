import matplotlib.pyplot as plt

def plot_state_space_trajectory(t, Y):
    """ ----- Plotting State Space Trajectory ----- """
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