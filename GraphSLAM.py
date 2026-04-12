from MapFeature import *
from FEKFMBL import *

def WrapAngle(angle):
    """
    Wraps an angle to the range [-pi, pi].

    :param angle: Angle to be wrapped (in radians)
    :return: Wrapped angle in the range [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class GraphSLAM(FEKFMBL):
    """
    Class implementing the Feature-based Extended Kalman Filter for Simultaneous Localization and Mapping (FEKFSLAM).
    It inherits from the FEKFMBL class, which implements the Feature Map based Localization (MBL) algorithm using an EKF.
    :class:`FEKFSLAM` extends :class:`FEKFMBL` by adding the capability to map features previously unknown to the robot.
    """
 
    def __init__(self,  *args):

        super().__init__(*args)

        # self.xk_1 # state vector mean at time step k-1 inherited from FEKFMBL
        self.i = 0
        self.initialize = False
        # self.odom_cov = np.zeros((self.xB_dim, self.xB_dim))  # covariance of the odometry readings
        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))

        self.nzm = 0  # number of measurements observed
        self.nzf = 0  # number of features observed

        self.H = None  # Data Association Hypothesis
        # self.nf = 0  # number of features in the state vector
        self.nf = len(self.robot.M)  # number of features

        self.compass_update = False  # whether the compass measurement
        self.feature_update = False  # whether the feature measurement is used for the update step

        self.plt_MappedFeaturesEllipses = []

        return

 
    def AddNewFeatures(self, xk, Pk, znp, Rnp):
        """
        This method adds new features to the map. Given:

        * The SLAM state vector mean and covariance:

        .. math::
            {{}^Nx_{k}} & \\approx {\\mathcal{N}({}^N\\hat x_{k},{}^NP_{k})}\\\\
            {{}^N\\hat x_{k}} &=   \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k}}&=
            \\begin{bmatrix}
            {{}^NP_{B}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance

        * And the vector of non-paired feature observations (feature which have not been associated with any feature in the map), and their covariance matrix:

            .. math::
                {z_{np}} &=   \\left[ {}^Bz_{F_1} ~  \\cdots ~ {}^Bz_{F_{n_{zf}}}  \\right]^T \\\\
                {R_{np}}&= \\begin{bmatrix}
                {}^BR_{F_1} &  \\cdots & 0  \\\\
                \\vdots &  \\ddots & \\vdots \\\\
                0 & \\cdots & {}^BR_{F_{n_{zf}}}
                \\end{bmatrix}
                :label: FEKFSLAM-non-paire-feature-observations

        this method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
        Therefore, for each new feature :math:`{}^Bz_{F_i}`, included in the vector :math:`z_{np}`, and its corresponding feature observation noise :math:`{}^B R_{F_i}`, the state vector mean and covariance are updated as follows:

            .. math::
                {{}^Nx_{k}^+} & \\approx {\\mathcal{N}({}^N\\hat x_{k}^+,{}^NP_{k}^+)}\\\\
                {{}^N x_{k}^+} &=
                \\left[ {{}^N x_{B_k}^T} ~ {{}^N x_{F_1}^T} ~ \\cdots ~{{}^N x_{F_n}^T}~ |~\\left({{}^N x_{B_k} \\boxplus ({}^Bz_{F_i} }+v_k)\\right)^T \\right]^T \\\\
                {{}^N\\hat x_{k}^+} &=
                \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~ \\cdots ~{{}^N\\hat x_{F_n}^T}~ |~{{}^N\\hat x_{B_k} \\boxplus {}^Bz_{F_i}^T } \\right]^T \\\\
                {P_{k}^+}&= \\begin{bmatrix}
                {{}^NP_{B_k}}  &  {{}^NP_{B_kF_1}}   &  \\cdots   &  {{}^NP_{B_kF_n}} & | & {{}^NP_{B_k} J_{1 \\boxplus}^T}\\\\
                {{}^NP_{F_1B_k}}  &  {{}^NP_{F_1}}   &  \\cdots   &  {{}^NP_{F_1F_n}} & | & {{}^NP_{F_1B_k} J_{1 \\boxplus}^T}\\\\
                \\vdots  & \\vdots & \\ddots  & \\vdots & | &  \\vdots \\\\
                {{}^NP_{F_nB_k}}  &  {{}^NP_{F_nF_1}}   &  \\cdots   &  {{}^NP_{F_n}}  & | & {{}^NP_{F_nB_k} J_{1 \\boxplus}^T}\\\\
                \\hline
                {J_{1 \\boxplus} {}^NP_{B_k}}  &  {J_{1 \\boxplus} {}^NP_{B_kF_1}}   &  \\cdots   &  {J_{1 \\boxplus} {}^NP_{B_kF_n}}  & | &  {J_{1 \\boxplus} {}^NP_R J_{1 \\boxplus} ^T} + {J_{2\\boxplus}} {{}^BR_{F_i}} {J_{2\\boxplus}^T}\\\\
                \\end{bmatrix}
                :label: FEKFSLAM-add-a-new-feature

        :param xk: state vector mean
        :param Pk: state vector covariance
        :param znp: vector of non-paired feature observations (they have not been associated with any feature in the map)
        :param Rnp: Matrix of non-paired feature observation covariances
        :return: [xk_plus, Pk_plus] state vector mean and covariance after adding the new features
        """

        assert len(znp) > 0, "AddNewFeatures: znp is empty"
        
        ## To be completed by the student
        xk_robot = Pose3D(np.array([xk[0:self.xB_dim, 0]]).T)
        xk_plus = xk.copy()
        nf_curr = int((len(xk_plus) - self.xB_dim)/self.xF_dim)
        Pk_plus = Pk.copy()
        nf = len(znp)  # number of new features to be added
        for i in range(int(nf)):
            num_curr_features = int((len(xk_plus) - self.xB_dim)/self.xF_dim)
            zfi = znp[i]
            Rfi = Rnp[i]

            G1_k = block_diag(*[np.eye(self.xB_dim), *([np.eye(self.xF_dim)] * num_curr_features)])
            Jgx = self.Jgx(xk_robot, zfi)
            Jgx = np.pad(Jgx, ((0, 0), (0, G1_k.shape[1]-Jgx.shape[1])), 'constant')
            G1_k = np.vstack((G1_k, Jgx))
            
            G2_k = np.zeros((len(xk_plus), self.zfi_dim))
            Jgv = self.Jgv(xk_robot, zfi)
            G2_k = np.vstack((G2_k, Jgv))

            xk_plus = np.vstack((xk_plus, self.g(xk_robot,zfi)))
            Pk_plus = G1_k @ Pk_plus @ G1_k.T + G2_k @ Rfi @ G2_k.T
        
        for i in range(nf_curr, nf_curr + nf):
            landmark_key = gtsam.symbol('L', self.nzf)
            pos_feature = xk_plus[self.xB_dim + self.nzf*self.xF_dim: self.xB_dim + (self.nzf+1)*self.xF_dim, 0]
            cov_feature = Pk_plus[self.xB_dim + self.nzf*self.xF_dim: self.xB_dim + (self.nzf+1)*self.xF_dim, self.xB_dim + self.nzf*self.xF_dim: self.xB_dim + (self.nzf+1)*self.xF_dim]
            noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_feature)
            self.initial.insert(landmark_key, gtsam.Point2(pos_feature[0], pos_feature[1]))
            self.graph.add(gtsam.PriorFactorPoint2(landmark_key, gtsam.Point2(pos_feature[0], pos_feature[1]), noise_model))
            self.nzf += 1

        return xk_plus, Pk_plus

    def Prediction(self, uk, Qk, xk_1, Pk_1):
        """
        This method implements the prediction step of the FEKFSLAM algorithm. It predicts the state vector mean and
        covariance at the next time step. Given state vector mean and covariance at time step k-1:

        .. math::
            {}^Nx_{k-1} & \\approx {\\mathcal{N}({}^N\\hat x_{k-1},{}^NP_{k-1})}\\\\
            {{}^N\\hat x_{k-1}} &=   \\left[ {{}^N\\hat x_{B_{k-1}}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k-1}}&=
            \\begin{bmatrix}
            {{}^NP_{B_{k-1}}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance-k-1

        the control input and its covariance :math:`u_k` and :math:`Q_k`, the method computes the state vector mean and covariance at time step k:

        .. math::
            {{}^N\\hat{\\bar x}_{k}} &=   \\left[ {f} \\left( {{}^N\\hat{x}_{B_{k-1}}}, {u_{k}}  \\right)  ~  { {}^N\\hat x_{F_1}^T} \\cdots { {}^N\\hat x_{F_n}^T}\\right]^T\\\\
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}
            :label: FEKFSLAM-prediction-step

        where

        .. math::
            {F_{1_k}} &= \\left.\\frac{\\partial {f_S({}^Nx_{k-1},u_k,w_k)}}{\\partial {{}^Nx_{k-1}}}\\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}} \\\\
             &=
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{B_{k-1}}}} &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{Fn}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}} \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_x}} & {0} & \\cdots & {0} \\\\
            {0}   & {I} & \\cdots & {0} \\\\
            \\vdots& \\vdots  & \\ddots & \\vdots  \\\\
            {0}   & {0} & \\cdots & {I} \\\\
            \\end{bmatrix}
            \\\\{F_{2_k}} &= \\left. \\frac{\\partial {f({}^Nx_{k-1},u_k,w_k)}}{\\partial {w_{k}}} \\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}}
            =
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {w_{k}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {w_{k}}}\\\\
            \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {w_{k}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_w}}\\\\
            {0}\\\\
            \\vdots\\\\
            {0}\\\\
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-Jacobian

        obtaining the following covariance matrix:
        
        .. math::
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}{{}^N\\bar P_{k}}
             &=
            \\begin{bmatrix}
            {J_{f_x}P_{B_{k-1}} J_{f_x}^T} + {J_{f_w}Q J_{f_w}^T}  & |  &  {J_{f_x}P_{B_kF_1}} & \\cdots & {J_{f_x}P_{B_kF_n}}\\\\
            \\hline
            {{}^NP_{F_1B_k} J_{f_x}^T} & |  &  {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_n}}\\\\
            \\vdots & | & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_nB_k} J_{f_x}^T} & | &  {{}^NP_{F_nF_1}} & \\cdots & {{}^NP_{F_n}}
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-covariance

        The method returns the predicted state vector mean (:math:`{}^N\\hat{\\bar x}_k`) and covariance (:math:`{{}^N\\bar P_{k}}`).

        :param uk: Control input
        :param Qk: Covariance of the Motion Model noise
        :param xk_1: State vector mean at time step k-1
        :param Pk_1: Covariance of the state vector at time step k-1
        :return: [xk_bar, Pk_bar] predicted state vector mean and covariance at time step k
        """

       ## To be completed by the student
        self.uk = uk
        self.Qk = Qk

        number_of_robot_states = self.xB_dim
        number_of_feature_states = int((len(xk_1) - number_of_robot_states)/self.xF_dim)
        xk_bar = xk_1.copy()

        x_robot_1 = self.GetRobotPose(xk_1)
        x_robot_pred = self.f(x_robot_1, uk)
        
        xk_bar[0:number_of_robot_states, 0] = x_robot_pred[0:number_of_robot_states, 0]
        Jfx = self.Jfx(x_robot_1)
        Jfw = self.Jfw(x_robot_1)
        F1k = block_diag(*[Jfx, *[np.eye(self.xF_dim)] * number_of_feature_states])
        F2k = np.vstack((Jfw, *[np.zeros((self.xF_dim, number_of_robot_states))] * number_of_feature_states))
        Pk_bar = F1k @ Pk_1 @ F1k.T + F2k @ Qk @ F2k.T

        rel_pose = Pose3D(self.rel_disp)
        J1_rel = rel_pose.J_1oplus(uk)
        J2_rel = rel_pose.J_2oplus()
        self.rel_cov = J1_rel @ self.rel_cov @ J1_rel.T + J2_rel @ Qk @ J2_rel.T
        self.rel_disp = np.array(rel_pose.oplus(uk)).reshape(3, 1)
        
        return xk_bar, Pk_bar
    
    def Update(self, zk, Rk, xk_bar):
        """
        Update step of the graph-SLAM. It calls the observation model and its Jacobians to update the state vector and its covariance matrix.

        :param zk: observation vector
        :param Rk: covariance matrix of the noise vector
        :param xk_bar: predicted mean state vector.
        :param Pk_bar: covariance matrix of the predicted state vector.
        :param Hk: Jacobian of the observation model with respect to the state vector.
        :param Vk: Jacobian of the observation model with respect to the noise vector.
        :return xk,Pk: updated mean state vector and its covariance matrix. Also updated in the class attributes.
        """
        # logging for plotting
        self.xk_bar = xk_bar
        self.zk = zk
        self.nz = zk.shape[0]  # store dimensionality of the observation
        self.Rk = Rk
        # KF equations begin here
        self.i += 1

        # relative_pose = gtsam.Pose2(self.xk_prev[0,0], self.xk_prev[1,0], self.xk_prev[2,0]).between(gtsam.Pose2(xk_bar[0,0], xk_bar[1,0], xk_bar[2,0]))

        # OdometryNoise = gtsam.noiseModel.Gaussian.Covariance(self.odom_cov)
        relative_pose = gtsam.Pose2(self.rel_disp[0,0], self.rel_disp[1,0], self.rel_disp[2,0])
        OdometryNoise = gtsam.noiseModel.Gaussian.Covariance(self.rel_cov + np.eye(3) * 1e-6)

        sigma_heading = float(np.sqrt(Rk[0, 0]))

        self.graph.add(gtsam.BetweenFactorPose2(self.i-1, self.i, relative_pose, OdometryNoise))

        if self.compass_update == True:
            self.graph.add(gtsam.PoseRotationPrior2D(self.i, gtsam.Rot2(zk[0,0]), gtsam.noiseModel.Isotropic.Sigma(1, sigma_heading)))

        self.initial.insert(self.i, gtsam.Pose2(xk_bar[0,0], xk_bar[1,0], xk_bar[2,0]))

        k = 0
        if self.feature_update == True:
            for j in range(len(self.H)):
                if self.H[j] != None:
                    feature_key = gtsam.symbol('L', self.H[j])
                    if self.compass_update == True:
                        angle = WrapAngle(np.arctan2(zk[1+self.zfi_dim*k+1, 0], zk[1+self.zfi_dim*k, 0]))
                        _range = np.linalg.norm(zk[1+self.zfi_dim*k:1+self.zfi_dim*k+2, 0])
                        J_r = np.array([[zk[1+self.zfi_dim*k, 0]/ _range, zk[1+self.zfi_dim*k+1, 0]/ _range]])
                        J_g = np.array([[-zk[1+self.zfi_dim*k+1, 0] / (_range**2), zk[1+self.zfi_dim*k, 0] / (_range**2)]])
                        J_full = np.vstack((J_g, J_r))
                        R_cart = Rk[1+self.zfi_dim*k:1+self.zfi_dim*k+2, 1+self.zfi_dim*k:1+self.zfi_dim*k+2]
                        model_noise = gtsam.noiseModel.Gaussian.Covariance(J_full @ R_cart @ J_full.T + np.eye(2) * 1e-6)
                    else:
                        angle = WrapAngle(np.arctan2(zk[self.zfi_dim*k+1, 0], zk[self.zfi_dim*k, 0]))
                        _range = np.linalg.norm(zk[self.zfi_dim*k:self.zfi_dim*k+2, 0])
                        J_r = np.array([[zk[self.zfi_dim*k, 0]/ _range, zk[self.zfi_dim*k+1, 0]/ _range]])
                        J_g = np.array([[-zk[self.zfi_dim*k+1, 0] / (_range**2), zk[self.zfi_dim*k, 0] / (_range**2)]])
                        J_full = np.vstack((J_g, J_r))
                        R_cart = Rk[self.zfi_dim*k:self.zfi_dim*k+2, self.zfi_dim*k:self.zfi_dim*k+2]
                        model_noise = gtsam.noiseModel.Gaussian.Covariance(J_full @ R_cart @ J_full.T + np.eye(2) * 1e-6)
                    k += 1
                    self.graph.add(gtsam.BearingRangeFactor2D(self.i, feature_key, gtsam.Rot2(angle), _range, model_noise))

        if self.initialize == False:
                optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
                self.initial = optimizer.optimize()
                self.initialize = True

        self.isam2.update(self.graph, self.initial)

        # MUST clear graph+initial after update — iSAM2 stores them internally
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        results = self.isam2.calculateEstimate()
        pose = results.atPose2(self.i)
        self.xk[0,0] = pose.x()
        self.xk[1,0] = pose.y()
        self.xk[2,0] = WrapAngle(pose.theta())

        full_graph = self.isam2.getFactorsUnsafe()
        marginals = gtsam.Marginals(full_graph, results)

        # Build key list matching EKF state vector order
        all_keys = gtsam.KeyVector()
        all_keys.append(self.i)  # current pose
        for j in range(self.nzf):
            all_keys.append(gtsam.symbol('L', j))

        joint_cov = marginals.jointMarginalCovariance(all_keys).fullMatrix()  # Have to use jointMarginalCovariance to get the full covariance matrix for the current pose and all features, as the marginal covariance only gives the covariance for a single variable

        for j in range(self.nzf):
            feature_key = gtsam.symbol('L', j)
            if results.exists(feature_key):
                feature_pos = results.atPoint2(feature_key)
                self.xk[self.xB_dim + j*self.xF_dim: self.xB_dim + (j+1)*self.xF_dim] = np.array([[feature_pos[0]], [feature_pos[1]]])
            
        state_dim = self.xB_dim + self.nzf * self.xF_dim
        self.Pk[0:state_dim, 0:state_dim] = joint_cov

        self.initial = gtsam.Values()  # reset the initial values for the next iteration

        # self.odom_cov = np.zeros((self.xB_dim, self.xB_dim))  # reset the odometry covariance after each update
        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))
        self.compass_update = False
        self.feature_update = False
        
        self.xk_prev = self.xk.copy()

        return self.xk, self.Pk

    def Localize(self, xk_1, Pk_1):
        """
        This method implements the FEKFSLAM algorithm. It localizes the robot and maps the features in the environment.
        It implements a single interation of the SLAM algorithm, given the current state vector mean and covariance.
        The unique difference it has with respect to its ancestor :meth:`FEKFMBL.Localize` is that it calls the method
        :meth:`AddNewFeatures` to add new non-paired features to the map.

        :param xk_1: state vector mean at time step k-1
        :param Pk_1: covariance of the state vector at time step k-1
        :return: [xk, Pk] state vector mean and covariance at time step k
        """

        ## To be completed by the student
        self.nf = int((len(xk_1) - self.xB_dim)/self.zfi_dim)

        uk, Qk = self.GetInput()
        xk_bar, Pk_bar = self.Prediction(uk, Qk, xk_1, Pk_1)
        zm, Rm, Hm, Vm = self.GetMeasurements()
        if zm is None:
            zm = []
            Rm = []
            Hm = []
            Vm = []
        self.zm=zm
        if len(Hm) != 0:
            Hm_temp = np.zeros((1, self.xB_dim + self.nf*self.zfi_dim))
            Hm_temp[0, 0:self.xB_dim] = Hm
            Hm = Hm_temp

        if len(zm) != 0:
            self.compass_update = True

        zf, Rf = self.GetFeatures()
        if len(zf) != 0:
            self.feature_update = True

        self.H = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)
        zk, Rk, Hk, Vk, znp, Rnp = self.StackMeasurementsAndFeatures(xk_bar,zm, Rm, Hm, Vm, zf, Rf, self.H)

        if len(zk) == 0:
            xk = xk_bar
            Pk = Pk_bar
        else:
            xk, Pk = self.Update(zk, Rk, xk_bar)
        
        if len(znp) > 0:
            xk, Pk = self.AddNewFeatures(xk, Pk, znp, Rnp)
        
        self.xk = xk
        self.Pk = Pk

        # Use the variable names zm, zf, Rf, znp, Rnp so that the plotting functions work
        self.Log(self.robot.xsk, self.GetRobotPose(self.xk), self.GetRobotPoseCovariance(self.Pk),
                 self.GetRobotPose(self.xk_bar), zm)  # log the results for plotting

        self.PlotUncertainty(zf, Rf, znp, Rnp)
        return self.xk, self.Pk

    def PlotMappedFeaturesUncertainty(self):
        """
        This method plots the uncertainty of the mapped features. It plots the uncertainty ellipses of the mapped
        features in the environment. It is called at each Localization iteration.
        """
        # remove previous ellipses
        for i in range(len(self.plt_MappedFeaturesEllipses)):
            self.plt_MappedFeaturesEllipses[i].remove()
        self.plt_MappedFeaturesEllipses = []

        self.xk=BlockArray(self.xk,self.xF_dim, self.xB_dim)
        self.Pk=BlockArray(self.Pk,self.xF_dim, self.xB_dim)

        # draw new ellipses
        for Fj in range(self.nf):
            feature_ellipse = GetEllipse(self.xk[[Fj]],
                                         self.Pk[[Fj,Fj]])  # get the ellipse of the feature (x_Fj,P_Fj)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], 'r')  # plot it
            self.plt_MappedFeaturesEllipses.append(plt_ellipse)  # and add it to the list

    def PlotUncertainty(self, zf, Rf, znp, Rnp):
        """
        This method plots the uncertainty of the robot (blue), the mapped features (red), the expected feature observations (black) and the feature observations.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param znp: vector of non-paired feature observations
        :param Rnp: covariance matrix of non-paired feature observations
        :return:
        """
        if self.k % self.robot.visualizationInterval == 0:
            # print('Plotting Uncertainty at step ', self.k)
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(znp, Rnp,'b')
            self.PlotFeatureObservationUncertainty(zf, Rf,'g')
            self.PlotExpectedFeaturesObservationsUncertainty()
            self.PlotMappedFeaturesUncertainty()
