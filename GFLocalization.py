from Localization import Localization
from GaussianFilter import GaussianFilter
import matplotlib.pyplot as plt
from GetEllipse import GetEllipse
import numpy as np

from Pose import Pose3D
class GFLocalization(Localization,GaussianFilter):
    """
    Map-less localization using a Gaussian filter.
    """
    def __init__(self, index, kSteps, robot, x0, P0,  *args):
        """
        Constructor.

        :param x0: initial state
        :param P0: initial covariance
        :param index: Named tuple used to relate the state vector, the simulation and the observation vectors (:class:`IndexStruct.IndexStruct`)
        :param kSteps: simulation time steps
        :param robot: Simulated Robot object
        :param args: arguments to be passed to the parent constructor
        """
        self.robot = robot  # robot object
        self.index = index  # list of index structures
        self.kSteps = kSteps
        self.k = 0  # initialize log time step

        x_state_exists = False
        y_state_exists = False
        self.plot_xy_estimation = False
        for s in range(len(index)):
            if index[s].state == 'x': x_state_exists = True
            if index[s].state == 'y': y_state_exists = True
        self.plot_xy_estimation = x_state_exists & y_state_exists

        self.plt_robot_ellipse, = plt.plot([], [], 'b')

        self.xTraj = []
        self.yTraj = []
        self.trajectory = plt.plot([], [], marker='.', color='blue', markersize=1)

        super().__init__(index, kSteps, robot, x0, P0, *args)  # call parent constructor

    def GetInput(self):  # get the input from the robot. To be overidden by the child class
        """
        Get the input from the robot. Relates to the motion model as follows:

        .. math::
            x_k &= f(x_{k-1},u_k,w_k) \\\\
            w_k &= N(0,Q_k)
            :label: eq-f-GFLocalization


        **To be overidden by the child class** .

        :return uk, Qk: input and covariance of the motion model
        """
        pass

    def GetMeasurements(self):  # get the measurements from the robot To be overidden by the child class
        """
        Get the measurements from the robot. Corresponds to the observation model:

        .. math::
            z_k &= h(x_{k},v_k) \\\\
            v_k &= N(0,R_k)
            :label: eq-h


        **To be overidden by the child class** .

        :return: zk, Rk, Hk, Vk: observation vector and covariance of the observation noise. Hk is the Observation matrix and Vk is the noise observation matrix.
        """
        pass

    def Localize(self, xk_1, Pk_1):
        """
        Localization iteration. Reads the input of the motion model, performs the prediction step, reads the measurements, performs the update step and logs the results.
        The method also plots the uncertainty ellipse of the robot pose.

        :param xk_1: previous state vector
        :param Pk_1: previous covariance matrix
        :return xk, Pk: updated state vector and covariance matrix
        """

        # TODO: To be implemented by the student
        uk, Qk = self.GetInput()
        if uk is None:
            xk_bar = xk_1
            Pk_bar = Pk_1
            self.xk_bar = xk_bar  # store predicted state for logging
        else:
            xk_bar, Pk_bar = self.Prediction(uk, Qk, xk_1, Pk_1)
            self.xk_bar = xk_bar  # store predicted state for logging
        
        zk, Rk, Hk, Vk = self.GetMeasurements()
        if zk is None:
            xk = xk_bar
            Pk = Pk_bar
        else:
            xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)
        
        self.PlotUncertainty(xk,Pk)

        return xk, Pk

    def LocalizationLoop(self, x0, P0, usk):
        """
        Localization loop. During *self.kSteps* it calls the :meth:`Localize` method for each time step.

        :param x0: initial state vector
        :param P0: initial covariance matrix
        """

        xk_1 = x0
        Pk_1 = P0
        xk_bar = x0
        zk = None
        zf, Rf = self.GetFeatures()  
        xk_1, Pk_1 = self.AddNewFeatures(xk_1,Pk_1, zf, Rf)  # initialize the map with the first observation

        xsk_1 = self.robot.xsk_1
        # self.kSteps +=1  # to include the last step in the loop
        for self.k in range(0, self.kSteps):
            xsk = self.robot.fs(xsk_1, usk)  # Simulate the robot motion
            xk, Pk = self.Localize(xk_1, Pk_1)  # Localize the robot

            xsk_1 = xsk  # current state becomes previous state for next iteration
            xk_1 = xk
            Pk_1 = Pk
            xk_bar = self.xk_bar
        
        self.PlotXY(estimation=True)  # plot the robot trajectory
        self.PlotState()  # plot the state estimation results
        plt.show()

    def Log(self, xsk, xk, Pk, xk_bar, zk):

        xk_dim = len(xk_bar)
        xk_bar_dim = len(xk_bar)

        # initialize the log arrays if they don't exist
        if not hasattr(self, 'log_x'): self.log_x = np.zeros((xk.shape[0], self.kSteps))
        if not hasattr(self, 'log_xs'): self.log_xs = np.zeros((xsk.shape[0], self.kSteps))
        # if not hasattr(self, 'log_z'): self.log_z = np.zeros((zk.shape[0], self.kSteps))
        if not hasattr(self, 'log_x_bar'): self.log_x_bar = np.zeros((xk_bar.shape[0], self.kSteps))
        if not hasattr(self, 'log_sigma'): self.log_sigma = np.zeros((xk.shape[0], self.kSteps))

        self.log_xs[0:xsk.shape[0], self.k] = xsk.T
        self.log_x[0:xk.shape[0], self.k] = xk.T
        self.log_sigma[0:xk.shape[0], self.k] = np.sqrt(np.diag(Pk)).T
        self.log_x_bar[0:xk_bar.shape[0], self.k] = xk_bar.T
        # self.log_z[0:self.zk.shape[0], self.k] = zk.T
        self.k += 1

    def PlotState(self):
        '''Plot the results of the localization
           For each state DOF s
           -si[s] is the corresponding simulated stated
           -x1[s] is the corresponding observation '''

        fig, axs = plt.subplots(self.log_x.shape[0], 3)
        axs = np.array([axs]) if self.log_x.shape[0] == 1 else axs
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        for s in range(self.log_x.shape[0]):  # for each estimated state
            axs[s, 0].set_title('x_'+str(s), fontstyle='italic')
            #axs[s, 0].plot(self.log_x_bar[s, 0:self.kSteps], ls='-', c='orange')
            axs[s, 0].plot(self.log_x[s, 0:self.kSteps], ls='-', c='blue')
            axs[s, 0].plot(
                self.log_x[s, 0:self.kSteps] + 3 * self.log_sigma[s, 0:self.kSteps],
                ls='-',
                c='green')
            axs[s, 0].plot(
                self.log_x[s, 0:self.kSteps] - 3 * self.log_sigma[s, 0:self.kSteps],
                ls='-',
                c='green')

            # if self.index[s].observation is not None:  # there is a corresponding observed state
            #     axs[s, 0].plot(self.log_z[self.index[s].observation, 0:self.kSteps], ls='-', c='pink')

            if self.index[s].simulation is not None and s<self.log_xs.shape[0]:  # there is a corresponding simulated state
                axs[s, 0].plot(self.log_xs[self.index[s].simulation, 0:self.kSteps], ls='-', c='red')
                axs[s, 1].set_title('error', fontstyle='italic')
                e = self.log_xs[self.index[s].simulation, 0:self.kSteps] - self.log_x[s,
                                                                           0:self.kSteps]  # error = simulated - estimated
                axs[s, 1].plot(e, ls='-', c='blue')
                axs[s, 1].plot(+3 * self.log_sigma[s, 0:self.kSteps], ls='-', c='green')
                axs[s, 1].plot(-3 * self.log_sigma[s, 0:self.kSteps], ls='-', c='green')

                axs[s, 2].set_title('error histogram', fontstyle='italic')
                axs[s, 2].hist(e, density=True, facecolor='b', alpha=0.75)
                min_ylim, max_ylim = plt.ylim()
                e_mean = e.mean()
                axs[s, 2].axvline(e_mean, color='k', linestyle='dashed', linewidth=1)
                axs[s, 2].text(e_mean * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(e_mean))

    def PlotXY(self, estimation=True):
        ''' Plot the x-y trajectory of the robot
        simulation: True if the simulated XY robot trajectory is available
        '''
        fig, axs = plt.figure(), plt.axes()
        axs.plot(self.log_xs[0, 0:self.kSteps], self.log_xs[1, 0:self.kSteps], ls='-', c='blue', label='Ground Truth')
        if self.plot_xy_estimation:
            axs.plot(self.log_x[0, 0:self.kSteps], self.log_x[1, 0:self.kSteps], ls='-', c='red', label='Estimated')
        plt.title("Robot XY Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    def PlotRobotUncertainty(self,xk,Pk):  # plots the robot trajectory and its uncertainty ellipse
        """
        Plots the robot trajectory and its uncertainty ellipse.
        :param xk: state vector
        :param Pk: covariance matrix of the state vector
        """
        if xk.size>1:
            # Plot Robot Ellipse
            robot_ellipse = GetEllipse(xk, Pk)
            self.plt_robot_ellipse.set_data(robot_ellipse[0], robot_ellipse[1])  # update it

            # Plot Robot Trajectory
            self.xTraj.append(xk[0, 0])
            self.yTraj.append(xk[1, 0])
            self.trajectory.pop(0).remove()
            self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='blue', markersize=1)

    def PlotUncertainty(self,xk,Pk):
        """
        Plots the uncertainty ellipse of the robot pose.
        :param xk: state vector
        :param Pk: covariance matrix of the state vector
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty(xk,Pk)
