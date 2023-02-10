from casadi import *
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import scipy.linalg
import os
from ament_index_python.packages import get_package_share_directory


def ev_track_ode_model(mass, camera_transform, desired_depth, desired_area):

    model_name = 'ev_drone_track'

    # constants
    m = mass
    g = 9.81 # gravity constant [m/s^2]
    cRb = camera_transform  # fixed transformation between drone's body frame to camera frame
    
    # constants related to image features
    z = desired_depth
    a = desired_area

    # set up states & controls
    xn = SX.sym('xn')      # visual features (chosen from image moments)
    yn = SX.sym('yn')
    an = SX.sym('an')
    qw = SX.sym('qw')      # quaternion
    qx = SX.sym('qx')
    qy = SX.sym('qy')
    qz = SX.sym('qz')
    vx = SX.sym('vx')      # body-frame linear velocity
    vy = SX.sym('vy')
    vz = SX.sym('vz')
 
    x = vertcat(xn, yn, an, qw, qx, qy, qz, vx, vy, vz)

    # controls
    f      = SX.sym('f')       # thrust
    omegax = SX.sym('omegax')  # angular rates
    omegay = SX.sym('omegay')
    omegaz = SX.sym('omegaz')

    u = vertcat(f, omegax, omegay, omegaz)
    
    # xdot
    xn_dot = SX.sym('xn_dot')
    yn_dot = SX.sym('yn_dot')
    an_dot = SX.sym('an_dot')
    qw_dot = SX.sym('qw_dot')
    qx_dot = SX.sym('qx_dot')
    qy_dot = SX.sym('qy_dot')
    qz_dot = SX.sym('qz_dot')
    vx_dot = SX.sym('vx_dot')
    vy_dot = SX.sym('vy_dot')
    vz_dot = SX.sym('vz_dot')

    xdot = vertcat(xn_dot, yn_dot, an_dot, qw_dot, qx_dot, qy_dot, qz_dot, vx_dot, vy_dot, vz_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []
    
    # dynamics
    f_expl = vertcat(
        -1*(cRb[0,0]*vx + cRb[0,1]*vy + cRb[0,2]*vz),
        -1*(cRb[1,0]*vx + cRb[1,1]*vy + cRb[1,2]*vz),
        -1*(cRb[2,0]*vx + cRb[2,1]*vy + cRb[2,2]*vz),
        0.5*(-qx*omegax - qy*omegay - qz*omegaz ), # quaternion derivatives: q_dot = 1/2*q[*]omega, [*]: quaternion multiplication 
        0.5*( qw*omegax + qy*omegaz - qz*omegay ),
        0.5*( qw*omegay + qz*omegax - qx*omegaz ),
        0.5*( qw*omegaz + qx*omegay - qy*omegax ),
        -2*(qx*qz - qw*qy)*g,                       # acceleration: a = f/m - R^T*[0; 0; g]
        -2*(qy*qz + qw*qx)*g,
        f/m - (1 - 2*(qx*qx + qy*qy))*g
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model


def visual_tracking_ocp_settings(mass, camera_transform, desired_depth, desired_area, intrinsic_param, Tf, N, Q, R, x0, max_thrust, max_omega_xy, max_omega_z, max_v):

    # create ocp object
    ocp = AcadosOcp() 
    
    # set model
    model = ev_track_ode_model(mass, camera_transform, desired_depth, desired_area) 
    ocp.model = model

    # set dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    cost_Q = np.diag(Q)
    cost_R = np.diag(R)
    ocp.cost.W = scipy.linalg.block_diag(cost_Q, cost_R)
    ocp.cost.W_e = cost_Q

    ocp.cost.Vx = np.zeros((ny, nx))   # state
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))   # controls
    ocp.cost.Vu[-4:,-4:] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(nx)         # terminal state

    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    # set constraints
    ocp.constraints.x0 = x0  # initial constraints

    ocp.constraints.lbu = np.array([0.1*max_thrust, -max_omega_xy, -max_omega_xy, -max_omega_z])
    ocp.constraints.ubu = np.array([0.9*max_thrust, max_omega_xy, max_omega_xy, max_omega_z])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3]) # control constraints
    
    ocp.constraints.lbx = np.array([0, 0, -max_v, -max_v, -max_v])
    ocp.constraints.ubx = np.array([intrinsic_param.x_lim, intrinsic_param.y_lim, max_v, max_v, max_v])
    ocp.constraints.idxbx = np.array([0, 1, 7, 8, 9]) # state constraints

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-4

    # set acados export directory & model file
    ocp.code_export_directory = os.path.join(get_package_share_directory('visual_mpc'), 'acados_code')
    model_file = os.path.join(
        os.path.relpath(get_package_share_directory('visual_mpc'), os.curdir),
        model.name + "_" + "acados_ocp.json"
    )
    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file=model_file)

    return acados_solver