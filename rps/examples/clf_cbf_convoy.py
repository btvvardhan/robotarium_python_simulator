import numpy as np
from cvxopt import matrix, solvers

from rps.robotarium import Robotarium

# Quiet CVXOPT output a bit
solvers.options['show_progress'] = False
solvers.options['reltol'] = 1e-4
solvers.options['feastol'] = 1e-4


def si_to_uni(u_si, poses, linear_gain=1.0, angular_gain=np.pi):
    """
    Map single-integrator velocities (2xN) to unicycle velocities (2xN).

    u_si:  2xN array of [u_x; u_y] in world frame
    poses: 3xN array of [x; y; theta]
    """
    assert u_si.shape[0] == 2
    assert poses.shape[0] == 3
    N = u_si.shape[1]

    dxu = np.zeros((2, N))

    for i in range(N):
        vx, vy = u_si[:, i]
        theta = poses[2, i]

        # Forward component along heading
        forward_vec = np.array([np.cos(theta), np.sin(theta)])
        perp_vec = np.array([-np.sin(theta), np.cos(theta)])

        v_forward = forward_vec.dot(np.array([vx, vy]))
        v_perp = perp_vec.dot(np.array([vx, vy]))

        v = linear_gain * v_forward

        if abs(v_forward) < 1e-6 and abs(v_perp) < 1e-6:
            omega = 0.0
        else:
            # Map angle between velocity vector and heading to angular rate
            omega = angular_gain * np.arctan2(v_perp, v_forward) / (np.pi / 2.0)

        dxu[0, i] = v
        dxu[1, i] = omega

    return dxu


def state_dependent_safe_distance(x_i, obstacles, d0, k_p, alpha):
    """
    Compute d_safe_eff for a robot at position x_i (2x1) given obstacle centers.
    """
    p = 0.0
    for o in obstacles:
        diff = x_i - o  # 2x1
        dist2 = float(diff.T @ diff)
        p += np.exp(-alpha * dist2)

    # Smooth saturating function sigma(p)
    sigma = 1.0 / (1.0 + np.exp(-p))
    return d0 + k_p * sigma


def solve_clf_cbf(i, x_si, u_nom, leader_pos, formation_offsets, obstacles, params):
    """
    Solve the CLF-CBF QP for follower i.

    i: follower index (int, >= 1)
    x_si: 2xN array of SI positions
    u_nom: 2xN array of nominal SI controls
    leader_pos: 2x1 array (leader position)
    formation_offsets: list of 2x1 arrays, one per follower
    obstacles: list of 2x1 arrays
    params: dict with c, gamma, d0, k_p, alpha, u_max
    """
    c = params["c"]
    gamma = params["gamma"]
    d0 = params["d0"]
    k_p = params["k_p"]
    alpha = params["alpha"]
    u_max = params["u_max"]

    N = x_si.shape[1]

    # State of follower i
    x_i = x_si[:, [i]]  # 2x1
    x_ref = leader_pos + formation_offsets[i - 1]
    e_i = x_i - x_ref

    u_des = u_nom[:, [i]]

    # If very close to reference, we can skip QP and just use u_des
    if float(e_i.T @ e_i) < 1e-6:
        return u_des

    # Effective safety distance based on obstacles
    d_safe = state_dependent_safe_distance(x_i, obstacles, d0, k_p, alpha)
    # For obstacle clearance, you can use same or slightly larger distance
    d_obs_eff = d_safe

    # Objective: minimize ||u - u_des||^2
    P = 2.0 * np.eye(2)
    q = -2.0 * u_des.reshape(2, 1)

    G_rows = []
    h_vals = []

    # 1) CLF: 2 e^T u <= -c ||e||^2
    G_clf = 2.0 * e_i.T  # 1x2
    h_clf = -c * float(e_i.T @ e_i)
    G_rows.append(G_clf)
    h_vals.append(h_clf)

    # 2) Robot-robot CBF constraints
    for j in range(N):
        if j == i:
            continue

        x_j = x_si[:, [j]]
        diff = x_i - x_j  # 2x1
        dist2 = float(diff.T @ diff)
        h_ij = dist2 - d_safe ** 2

        # Only enforce if we're within some "interaction" band
        if h_ij < 0.05:  # you can tune this
            a_ij = 2.0 * diff  # 2x1

            u_j = u_nom[:, [j]]
            rhs = -gamma * h_ij + float((2.0 * diff).T @ u_j)

            # a^T u >= rhs  -> -a^T u <= -rhs
            G_ij = -a_ij.T  # 1x2
            h_ij_qp = -rhs

            G_rows.append(G_ij)
            h_vals.append(h_ij_qp)

    # 3) Robot-obstacle CBF constraints
    for o in obstacles:
        diff = x_i - o  # 2x1
        dist2 = float(diff.T @ diff)
        h_ik = dist2 - d_obs_eff ** 2

        if h_ik < 0.05:
            a_ik = 2.0 * diff
            rhs = -gamma * h_ik

            G_ik = -a_ik.T
            h_ik_qp = -rhs

            G_rows.append(G_ik)
            h_vals.append(h_ik_qp)

    # 4) Speed limit approx: box constraints on u_x, u_y
    G_rows.append(np.array([[1.0, 0.0]]))
    h_vals.append(u_max)
    G_rows.append(np.array([[-1.0, 0.0]]))
    h_vals.append(u_max)
    G_rows.append(np.array([[0.0, 1.0]]))
    h_vals.append(u_max)
    G_rows.append(np.array([[0.0, -1.0]]))
    h_vals.append(u_max)

    # Stack constraints
    G = np.vstack(G_rows)  # m x 2
    h = np.array(h_vals).reshape(-1, 1)  # m x 1

    # Convert to CVXOPT matrices
    P_c = matrix(P, tc='d')
    q_c = matrix(q, tc='d')
    G_c = matrix(G, tc='d')
    h_c = matrix(h, tc='d')

    try:
        sol = solvers.qp(P_c, q_c, G_c, h_c)
        if sol['status'] != 'optimal':
            # Fallback to nominal if solver fails
            return u_des
        u_opt = np.array(sol['x']).reshape((2, 1))
        return u_opt
    except Exception:
        # If anything goes wrong, just use nominal
        return u_des


def main():
    # ---------- High-level parameters ----------
    N = 5                      # 1 leader + 4 followers
    num_iterations = 8000

    # Leader and follower gains
    k_L = 0.5
    k_f = 1.0

    # CLF/CBF parameters
    c = 0.5
    gamma = 1.0
    u_max = 0.15

    # State-dependent safety parameters
    d0 = 0.12     # base safety distance
    k_p = 0.08    # scaling for environment crowding
    alpha = 5.0   # how fast proximity grows near obstacles

    params = dict(c=c, gamma=gamma, d0=d0, k_p=k_p, alpha=alpha, u_max=u_max)

    # Formation: offsets for followers (in leader frame)
    formation_offsets = [
        np.array([[-0.15], [-0.15]]),  # robot 1
        np.array([[0.15], [-0.15]]),   # robot 2
        np.array([[-0.30], [-0.30]]),  # robot 3
        np.array([[0.30], [-0.30]]),   # robot 4
    ]

    # Virtual obstacles ("shelves") as centers in the arena
    obstacles = [
        np.array([[0.0], [0.3]]),
        np.array([[0.0], [-0.3]]),
    ]

    # Leader waypoints to create a "corridor" traversal
    waypoints = np.array([
        [-0.7,  0.7,  0.7, -0.7],   # x
        [ 0.0,  0.0,  0.2,  0.0],   # y
    ])
    state = 0
    close_enough = 0.05

    # ---------- Create Robotarium instance ----------
    r = Robotarium(number_of_robots=N,
                   show_figure=True,
                   sim_in_real_time=True)

    for k in range(num_iterations):
        # Get current poses (3xN) and extract SI positions (2xN)
        x = r.get_poses()
        x_si = x[0:2, :]

        # Nominal SI controls
        u_si_nom = np.zeros((2, N))

        # ----- Leader (index 0) -----
        leader_pos = x_si[:, [0]]
        wp = waypoints[:, [state]]

        u_L_des = -k_L * (leader_pos - wp)

        # Clip leader speed
        norm_L = np.linalg.norm(u_L_des)
        if norm_L > u_max:
            u_L_des *= u_max / norm_L

        u_si_nom[:, [0]] = u_L_des

        # Switch waypoint if close
        if np.linalg.norm(leader_pos - wp) < close_enough:
            state = (state + 1) % waypoints.shape[1]

        # ----- Followers: nominal CLF controls -----
        for i in range(1, N):
            x_i = x_si[:, [i]]
            x_ref = leader_pos + formation_offsets[i - 1]
            e_i = x_i - x_ref

            u_i_des = -k_f * e_i
            norm_i = np.linalg.norm(u_i_des)
            if norm_i > u_max:
                u_i_des *= u_max / norm_i

            u_si_nom[:, [i]] = u_i_des

        # ----- Followers: CLF-CBF QP -----
        u_si = np.copy(u_si_nom)
        for i in range(1, N):
            u_si[:, [i]] = solve_clf_cbf(
                i=i,
                x_si=x_si,
                u_nom=u_si_nom,
                leader_pos=leader_pos,
                formation_offsets=formation_offsets,
                obstacles=obstacles,
                params=params
            )

        # Optionally you could also apply CBF to leader, but here we just
        # focus on follower safety.

        # ----- Map SI velocities to unicycle commands and step -----
        dxu = si_to_uni(u_si, x)

        r.set_velocities(np.arange(N), dxu)
        r.step()

    r.call_at_scripts_end()


if __name__ == "__main__":
    main()
