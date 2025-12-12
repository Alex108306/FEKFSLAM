from MapFeature import *
from blockarray import *
class FEKFSLAMFeature(MapFeature):
    """
    This class extends the :class:`MapFeature` class to implement the Feature EKF SLAM algorithm.
    The  :class:``MapFeature`` class is a base class providing support to localize the robot using a map of point features.
    The main difference between FEKMBL and FEAKFSLAM is that the former uses the robot pose as a state variable,
    while the latter uses the robot pose and the feature map as state variables. This means that few methods provided by
    class need to be overridden to gather the information from state vector instead that from the deterministic map.
    """
    def hfj(self, xk_bar, Fj):  # Observation function for zf_i and x_Fj
        """
        This method implements the direct observation model for a single feature observation  :math:`z_{f_i}` , so it implements its related
        observation function (see eq. :eq:`eq-FEKFSLAM-hfj`). For a single feature observation :math:`z_{f_i}` of the feature :math:`^Nx_{F_j}` the method computes its
        expected observation from the current robot pose :math:`^Nx_B`.
        This function uses a generic implementation through the following equation:

        .. math::
            z_{f_i}=h_{Fj}(x_k,v_k)=s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_j}) + v_{fi_k}
            :label: eq-FEKFSLAM-hfj

        Where :math:`^Nx_B` is the robot pose and :math:`^Nx_{F_j}` are both included within the state vector:

        .. math::
            x_k=[^Nx_B^T~\cdots~^Nx_{F_j}~\cdots~^Nx_{F_{nf}}]^T
            :label: eq-FEKFSLAM-xk

        and :meth:`s2o` is a conversion function from the store representation to the observation representation.

        The method is called by :meth:`FEKFSLAM.hf` to compute the expected observation for each feature
        observation contained in the observation vector :math:`z_f=[z_{f_1}^T~\\cdots~z_{f_i}^T~\\cdots~z_{f_{n_zf}}^T]^T`.

        :param xk_bar: mean of the predicted state vector
        :param Fj: map index of the observed feature.
        :return: expected observation of the feature :math:`^Nx_{F_j}`
        """
        
        ## To be completed by the student
        xk_bar_robot = self.GetRobotPose(xk_bar)
        x_Fj = np.array([xk_bar[(self.xB_dim + Fj*self.zfi_dim): (self.xB_dim + (Fj+1)*self.zfi_dim), 0]]).T
        x_Fj = CartesianFeature(x_Fj)
        hfj = self.s2o(x_Fj).boxplus(xk_bar_robot.ominus())
        return hfj
        #return ...

    def Jhfjx(self, xk, Fj):  # Observation function for zf_i and x_Fj
        """
        Jacobian of the single feature direct observation model :meth:`hfj` (eq. :eq:`eq-FEKFSLAM-hfj`)  with respect to the state vector :math:`\\bar{x}_k`:

        .. math::
            x_k&=[^Nx_B^T~\cdots~^Nx_{F_j}~\cdots~^Nx_{F_{nf}}]^T\\\\
            J_{hfjx}&=\\frac{\\partial h_{f_{zfi}}({x}_k, v_k)}{\\partial {x}_k}=
            \\frac{\\partial s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_j})+v_{fi_k}}{\\partial {x}_k}\\\\
            &=
            \\begin{bmatrix}
            \\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{B_k}}} & \\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{F_1}}} & \\cdots &\\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{F_j}}} & \\cdots & \\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{F_n}} } \\\\
            \\end{bmatrix} \\\\
            &=
            \\begin{bmatrix}
            J_{s2o}{J_{1\\boxplus} J_\\ominus} & {0} & \\cdots & J_{s2o}{J_{2\\boxplus}} & \\cdots &{0}\\\\
            \\end{bmatrix}\\\\
            :label: eq-FEKFSLAM-Jhfjx

        where we have used the abreviature:

        .. math::
            J_{s2o} &\equiv J_{s2o}(\\ominus ^Nx_B \\boxplus^Nx_{F_j})\\\\
            J_{1\\boxplus} &\equiv J_{1\\boxplus}(\\ominus ^Nx_B,^Nx_{F_j} )\\\\
            J_{2\\boxplus} &\equiv J_{2\\boxplus}(\\ominus ^Nx_B,^Nx_{F_j} )\\\\

        :param xk: state vector mean
        :param Fj: map index of the observed feature
        :return: Jacobian matrix defined in eq. :eq:`eq-Jhfjx`        """

        ## To be completed by the student
        num_features = (xk.shape[0] - self.xB_dim) // self.zfi_dim
        xk_bar_robot = self.GetRobotPose(xk)
        x_Fj = np.array([xk[(self.xB_dim + Fj*self.zfi_dim): (self.xB_dim + (Fj+1)*self.zfi_dim), 0]]).T
        x_Fj = CartesianFeature(x_Fj)
        a = self.J_s2o((x_Fj).boxplus(xk_bar_robot.ominus())) @ ((x_Fj).J_1boxplus(xk_bar_robot.ominus())) @ xk_bar_robot.J_ominus()
        b = self.J_s2o((x_Fj).boxplus(xk_bar_robot.ominus())) @ ((x_Fj).J_2boxplus(xk_bar_robot.ominus()))
        J_hjfx = np.zeros((self.zfi_dim, self.xB_dim + num_features*self.zfi_dim))
        J_hjfx[0:self.zfi_dim, 0:self.xB_dim] = a
        J_hjfx[0:self.zfi_dim, self.xB_dim + Fj*self.zfi_dim : self.xB_dim + (Fj+1)*self.zfi_dim] = b
        return J_hjfx
        #return ...

class FEKFSLAM2DCartesianFeature(FEKFSLAMFeature, Cartesian2DMapFeature):
    """
    Class to inherit from both :class:`FEKFSLAMFeature` and :class:`Cartesian2DMapFeature` classes.
    Nothing else to do here (if using s2o & o2s), only needs to be defined.
    """
    pass


