# Based upon the iLQR implementation at https://github.com/anassinator/ilqr
"""Controllers."""

import six
import abc
import warnings
import math
import numpy as np
import tensorflow as tf

from scipy.linalg import block_diag


@six.add_metaclass(abc.ABCMeta)
class BaseController():

    """Base trajectory optimizer controller."""

    @abc.abstractmethod
    def fit(self, x0, us_init, *args, **kwargs):
        """Computes the optimal controls.
        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            *args, **kwargs: Additional positional and key-word arguments.
        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        raise NotImplementedError


class iLQR(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, args, net, sess, cost, max_reg=1e10, worst_case=False):
        """Constructs an iLQR solver.
        Args:
            args: Various arguments and specifications
            net: Neural network dynamics model
            sess: TensorFlow session
            cost: Cost function.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            worst_case: whether to optimize for worst-case cost, otherwise will 
                optimize for expected cost
        """
        self.net = net
        self.sess = sess
        self.cost = cost
        self.H = args.mpc_horizon
        self.seq_length = args.seq_length
        self.num_models = args.num_models
        self.worst_case = worst_case
        if self.num_models == 1: self.worst_case = False
        self.state_dim = args.latent_dim
        self.full_state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.action_max = args.action_max

        # Regularization terms
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = np.zeros((self.H, self.action_dim))
        if self.worst_case:
            self._K = np.zeros((self.H, self.action_dim, self.state_dim))
        else:
            self._K = np.zeros((self.H, self.action_dim, self.state_dim*self.num_models))

        super(iLQR, self).__init__()

    def fit(self, x0, us_init, A, B, n_iterations=100, tol=1e-6, on_iteration=None):
        """Computes the optimal controls.
        Args:
            x0: Initial state [num_models*state_dim].
            us_init: Initial control path [H, action_dim].
            A: dynamics A-matrix [state_dim, latent_dim]
            B: dynamics B-matrix [state_dim, action_dim]
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.
        Returns:
            Tuple of
                xs: optimal state path [H+1, num_models*state_dim].
                us: optimal control path [H, action_dim].
                L_opt: optimal predicted cost
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        k = self._k
        K = self._K
        states = None

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, A_stack, B_stack, L, L_x, L_u, L_xx, L_ux, L_uu) = self._forward_rollout(x0, us, A, B)
                L_opt = L
                J_opt = L.sum()
                changed = False

            try:
                # Backward pass (only use first model if optimizing for worst-case cost)
                if self.worst_case:
                    k, K = self._backward_pass(A_stack[:self.state_dim, :self.state_dim], B_stack[:self.state_dim],\
                                 L_x[:, :self.state_dim], L_u, L_xx[:, :self.state_dim, :self.state_dim],\
                                 L_ux[:, :, :self.state_dim], L_uu, us)
                else:
                    k, K = self._backward_pass(A_stack, B_stack, L_x, L_u, L_xx, L_ux, L_uu, us)

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, A_stack, B_stack, k, K, alpha)
                    L_new = self._trajectory_cost(xs_new, us_new)
                    J_new = sum(L_new)

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        L_opt = L_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us, L_opt

    def _control(self, xs, us, A_stack, B_stack, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.
        Args:
            xs: Nominal state path [H+1, num_models*state_dim].
            us: Nominal control path [H, action_dim].
            A_stack: A-matrix for augmented dynamics [num_models*state_dim, num_models*state_dim]
            B_stack: B-matrix for augmented dynamics [num_models*state_dim, action_dim]
            k: Feedforward gains [H, action_dim].
            K: Feedback gains [H, action_size, state_dim].
            alpha: Line search coefficient.
        Returns:
            Tuple of
                xs: state path [H+1, num_models*state_dim].
                us: control path [H, action_dim].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        # Find new action and state trajectory
        for i in range(self.H):
            if self.worst_case:
                us_new[i] = us[i] + alpha * (k[i] + K[i].dot(xs_new[i, :self.state_dim] - xs[i, :self.state_dim]))
            else:
                us_new[i] = us[i] + alpha * (k[i] + K[i].dot(xs_new[i] - xs[i]))

            # Eq (8c).
            xs_new[i + 1] = np.dot(A_stack, xs_new[i]) + np.dot(B_stack, self.action_max*np.tanh(us_new[i]))

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.
        Args:
            xs: State path [H+1, state_dim].
            us: Control path [H, action_dim].
        Returns:
            Trajectory's total cost.
        """
        # Reshape matrix containing state trajectory
        xs = xs.reshape(self.H+1, -1, self.state_dim).transpose(1, 0, 2)

        # Find number of passes to be made through network
        num_models = len(xs)
        n_passes = int(math.ceil(num_models/float(self.batch_size)))

        # Initialize arrays for network input and output
        states = np.zeros((self.batch_size*n_passes, self.H+1, self.full_state_dim))
        x_in = np.zeros((self.batch_size*n_passes, self.seq_length, self.state_dim))
        
        # Feed in latent states and get out full state values
        x_in[:num_models, :self.H] = xs[:, :self.H]
        xT_in = np.zeros((self.batch_size*n_passes, self.state_dim))
        xT_in[:num_models] = xs[:, -1]
        for n in range(n_passes):
            feed_in = {}
            feed_in[self.net.z_vals_reshape] = x_in[n*self.batch_size:(n+1)*self.batch_size]
            feed_in[self.net.z1] = xT_in[n*self.batch_size:(n+1)*self.batch_size]
            feed_out = [self.net.x_pred_init, self.net.rec_state]
            x_rec, xT_rec = self.sess.run(feed_out, feed_in)
            states[self.batch_size*n:(self.batch_size*(n+1)), :self.H] = x_rec[:, :self.H]
            states[self.batch_size*n:(self.batch_size*(n+1)), self.H] = xT_rec
        if self.worst_case:
            states = states[:1]
        else:
            states = states[:num_models]

        # Use external function to calculate cost
        cost = self.cost(states, us, self.gamma)
        return cost

    def _forward_rollout(self, x0, us, A, B):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.
        Args:
            x0: Initial state [state_dim].
            us: Control path [H, action_dim].
            A: Set of A-matrices [num_models, state_dim, state_dim].
            B: Set of B-matrices [num_models, state_dim, action_dim].
        Returns:
            Tuple of:
            xs, A_stack, B_stack, L, L_x, L_u, L_xx, L_ux, L_uu
                xs: State path [H+1, state_dim].
                A_stack: A-matrix describing augmented dynamics
                    [num_models*state_dim, num_models*state_dim].
                B_stack: B-matrix describing augmented dynamics
                    [num_models*state_dim, action_dim].
                L: Cost path [H+1].
                L_x: Jacobian of cost path w.r.t. x [H+1, num_models*state_dim].
                L_u: Jacobian of cost path w.r.t. u [H, action_dim].
                L_xx: Hessian of cost path w.r.t. x, x
                    [H+1, num_models*state_dim, num_models*state_dim].
                L_ux: Hessian of cost path w.r.t. u, x
                    [H, action_dim, num_models*state_dim].
                L_uu: Hessian of cost path w.r.t. u, u
                    [H, action_dim, action_dim].
        """
        # Find number of times to do pass through network
        n_passes = int(math.ceil(self.num_models/float(self.batch_size)))

        # Pad us with zeros
        us = np.concatenate((us, np.zeros((self.seq_length - self.H, self.action_dim))), axis=0)

        # Construct arrays of A-matrices and B-matrices to feed into network
        A_mats = np.zeros((self.batch_size*n_passes, self.state_dim, self.state_dim))
        A_mats[:self.num_models] = A
        B_mats = np.zeros((self.batch_size*n_passes, self.action_dim, self.state_dim))
        B_mats[:self.num_models] = B
        x0_vals = np.zeros((self.batch_size*n_passes, self.state_dim))
        x0_vals[:self.num_models] = x0
        u_vals = np.zeros((self.batch_size*n_passes, self.H, self.action_dim))
        u_vals[:] = us

        # Initialize arrays to hold all loss arrays and states
        L_arr = np.zeros((self.batch_size*n_passes, self.H+1))
        L_x_arr = np.zeros((self.batch_size*n_passes, self.H+1, self.state_dim))
        L_u_arr = np.zeros((self.batch_size*n_passes, self.H, self.action_dim))
        L_xx_arr = np.zeros((self.batch_size*n_passes, self.H+1, self.state_dim, self.state_dim))
        L_ux_arr = np.zeros((self.batch_size*n_passes, self.H, self.action_dim, self.state_dim))
        L_uu_arr = np.zeros((self.batch_size*n_passes, self.H, self.action_dim, self.action_dim))
        xs_arr = np.zeros((self.batch_size*n_passes, self.H+1, self.state_dim))

        # Perform necessary number of passes through network
        for n in range(n_passes):
            feed_in = {}
            feed_in[self.net.A] = A_mats[n*self.batch_size:(n+1)*self.batch_size]
            feed_in[self.net.B] = B_mats[n*self.batch_size:(n+1)*self.batch_size]
            feed_in[self.net.z1] = x0_vals[n*self.batch_size:(n+1)*self.batch_size]
            feed_in[self.net.u_ilqr] = u_vals[n*self.batch_size:(n+1)*self.batch_size]
            feed_out = [self.net.L, self.net.L_x, self.net.L_u, self.net.L_xx, self.net.L_ux, self.net.L_uu, self.net.xs]
            L, L_x, L_u, L_xx, L_ux, L_uu, xs = self.sess.run(feed_out, feed_in)

            L_arr[self.batch_size*n:(self.batch_size*(n+1))] = L
            L_x_arr[self.batch_size*n:(self.batch_size*(n+1))] = L_x
            L_u_arr[self.batch_size*n:(self.batch_size*(n+1))] = L_u
            L_xx_arr[self.batch_size*n:(self.batch_size*(n+1))] = L_xx
            L_ux_arr[self.batch_size*n:(self.batch_size*(n+1))] = L_ux
            L_uu_arr[self.batch_size*n:(self.batch_size*(n+1))] = L_uu
            xs_arr[self.batch_size*n:(self.batch_size*(n+1))] = xs
            
        # Extract desired number sets for each array
        L_arr = L_arr[:self.num_models]
        L_x_arr = L_x_arr[:self.num_models]
        L_u_arr = L_u_arr[:self.num_models]
        L_xx_arr = L_xx_arr[:self.num_models]
        L_ux_arr = L_ux_arr[:self.num_models]
        L_uu_arr = L_uu_arr[:self.num_models]
        xs_arr = xs_arr[:self.num_models]

        # Identify outliers (any model that predicts a cost greater than twice the median)
        sum_cost = np.sum(L_arr, axis=1)
        cost_median = np.median(sum_cost)
        candidates = sum_cost * (sum_cost <= 2.0*cost_median)

        # If optimizing for worst case, replace all models with model with highest predicted cost (excluding outliers)
        if self.worst_case:
            max_idx = np.argmax(candidates)
            L_arr[:] = L_arr[max_idx]
            L_x_arr[:] = L_x_arr[max_idx]
            L_u_arr[:] = L_u_arr[max_idx]
            L_xx_arr[:] = L_xx_arr[max_idx]
            L_ux_arr[:] = L_ux_arr[max_idx]
            L_uu_arr[:] = L_uu_arr[max_idx]
            xs_arr[:] = xs_arr[max_idx]
            A[:] = A[max_idx]
            B[:] = B[max_idx]
        else:
            # For expected cost, use all non-outlier models
            L_arr = L_arr[candidates > 0]
            L_x_arr = L_x_arr[candidates > 0]
            L_u_arr = L_u_arr[candidates > 0]
            L_xx_arr = L_xx_arr[candidates > 0]
            L_ux_arr = L_ux_arr[candidates > 0]
            L_uu_arr = L_uu_arr[candidates > 0]
            xs_arr = xs_arr[candidates > 0]
            A = A[candidates > 0]
            B = B[candidates > 0]

        # Find number of remaining models
        num_models = len(L_arr)

        # Initialize and fill in arrays to hold cost quantities
        L = np.mean(L_arr, axis=0)
        L_x = np.zeros((self.H+1, num_models*self.state_dim))
        L_xx = np.zeros((self.H+1, num_models*self.state_dim, num_models*self.state_dim))
        L_ux = np.zeros((self.H, self.action_dim, num_models*self.state_dim))
        L_u = L_u_arr[0]
        L_uu = L_uu_arr[0]
        xs = np.zeros((self.H+1, num_models*self.state_dim))
        for t in range(self.H+1):
            L_x[t] = L_x_arr[:, t].reshape(num_models*self.state_dim)/num_models
            L_xx[t] = block_diag(*L_xx_arr[:, t])/num_models
            xs[t] = xs_arr[:, t].reshape(num_models*self.state_dim)
            if t < self.H:
                L_ux[t] = L_ux_arr[:, t].transpose(1, 0, 2).reshape(self.action_dim, num_models*self.state_dim)/num_models

        # Reshape A and B to have dimensionality appropriate for augmented state
        A_stack = block_diag(*A.transpose(0, 2, 1))
        B_stack = B.transpose(0, 2, 1).reshape(num_models*self.state_dim, self.action_dim)

        return xs, A_stack, B_stack, L, L_x, L_u, L_xx, L_ux, L_uu

    def _backward_pass(self,
                       A_stack,
                       B_stack,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       us):
        """Computes the feedforward and feedback gains k and K.
        Args:
            A_stack: A-matrix describing augmented dynamics
                    [num_models*state_dim, num_models*state_dim].
            B_stack: B-matrix describing augmented dynamics
                    [num_models*state_dim, action_dim].
            L_x: Jacobian of cost path w.r.t. x [H+1, num_models*state_dim].
            L_u: Jacobian of cost path w.r.t. u [H, action_dim].
            L_xx: Hessian of cost path w.r.t. x, x
                [H+1, num_models*state_dim, num_models*state_dim].
            L_ux: Hessian of cost path w.r.t. u, x [H, action_dim, num_models*state_dim].
            L_uu: Hessian of cost path w.r.t. u, u
                [H, action_dim, action_dim].
        Returns:
            Tuple of
                k: feedforward gains [H, action_dim].
                K: feedback gains [H, action_dim, state_dim].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.zeros((self.H, self.action_dim, len(A_stack)))

        for i in range(self.H - 1, -1, -1):
            f_u = self.action_max*B_stack*(1 - np.tanh(us[i])**2)
            f_uu = -2*np.tanh(us[i])*f_u
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(A_stack, f_u, f_uu, L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx)

            # Eq (6).
            try:
                k[i] = -np.linalg.solve(Q_uu, Q_u)
                K[i] = -np.linalg.solve(Q_uu, Q_ux)
            except:
                pdb.set_trace()

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           A_stack,
           f_u,
           f_uu,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx):
        """Computes second order expansion.
        Args:
            A_stack: A-matrix describing augmented dynamics
                    [num_models*state_dim, num_models*state_dim].
            f_u: Jacobian of state w.r.t. u [num_models*state_dim, action_dim].
            f_uu: 2nd deriv of state w.r.t. u [num_models*state_dim, action_dim].
            l_x: Jacobian of cost path w.r.t. x [num_models*state_dim].
            l_u: Jacobian of cost path w.r.t. u [action_dim].
            l_xx: Hessian of cost path w.r.t. x, x
                [num_models*state_dim, num_models*state_dim].
            l_ux: Hessian of cost path w.r.t. u, x [action_dim, num_models*state_dim].
            l_uu: Hessian of cost path w.r.t. u, u
                [action_dim, action_dim].
            V_x: Jacobian of the value function at the next time step
                [num_models*state_dim].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [num_models*state_dim, num_models*state_dim].
        Returns:
            Tuple of
                Q_x: [num_models*state_dim].
                Q_u: [action_dim].
                Q_xx: [num_models*state_dim, num_models*state_dim].
                Q_ux: [action_dim, num_models*state_dim].
                Q_uu: [action_dim, action_dim].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + A_stack.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + A_stack.T.dot(V_xx).dot(A_stack)

        # Eqs (11b) and (11c).
        if self.worst_case:
            reg = self._mu * np.eye(self.state_dim)
        else:
            reg = self._mu * np.eye(len(A_stack))
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(A_stack)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
