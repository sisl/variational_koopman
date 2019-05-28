import numpy as np
import tensorflow as tf
import pdb

class VariationalKoopman():
    def __init__(self, args):
        """Constructs Deep Variational Koopman Model 
        Args:
            args: Various arguments and specifications
        """

        # Placeholder for states and control inputs
        self.x = tf.Variable(np.zeros((2*args.batch_size*args.seq_length, args.state_dim), dtype=np.float32), trainable=False, name="state_values")
        self.u = tf.Variable(np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32), trainable=False, name="action_values")
        
        # Placeholders for values needed for ilqr
        self.u_ilqr = tf.Variable(np.zeros((args.batch_size, args.seq_length, args.action_dim), dtype=np.float32), trainable=False, name="action_values_ilqr")

        # Parameters to be set externally
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        self.kl_weight = tf.Variable(0.0, trainable=False, name="kl_weight")

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(args.state_dim), trainable=False, name="state_shift", dtype=tf.float32)
        self.scale = tf.Variable(np.zeros(args.state_dim), trainable=False, name="state_scale", dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_scale", dtype=tf.float32)
        
        # Create the computational graph
        self._create_feature_extractor_params(args)
        self._create_feature_extractor(args)
        self._create_temporal_encoder(args)
        self._create_inference_network_params(args)
        self._infer_observations(args)
        self._create_prior_network(args)
        self._propagate_solution(args)
        self._create_decoder_params(args)
        self._generate_predictions(args)
        if args.ilqr:
            self._find_ilqr_params(args)
        self._create_optimizer(args)

    def _create_feature_extractor_params(self, args):
        """Create parameters to comprise feature extractor
        Args:
            args: Various arguments and specifications
        """
        self.extractor_w = []
        self.extractor_b = []

        # Loop through elements of feature extractor and define parameters
        for i in range(len(args.extractor_size)):
            if i == 0:
                prev_size = args.state_dim
            else:
                prev_size = args.extractor_size[i-1]
            self.extractor_w.append(tf.get_variable("extractor_w"+str(i), [prev_size, args.extractor_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.extractor_b.append(tf.get_variable("extractor_b"+str(i), [args.extractor_size[i]]))

        # Last set of weights to map to output
        self.extractor_w.append(tf.get_variable("extractor_w_end", [args.extractor_size[-1], args.latent_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.extractor_b.append(tf.get_variable("extractor_b_end", [args.latent_dim]))

    def _get_extractor_output(self, args, states):
        """Function to run inputs through extractor
        Args:
            args: Various arguments and specifications
            states: states to feed into extractor [2*batch_size*seq_length, state_dim]
        Returns:
            Extracted features
        """
        extractor_input = states
        for i in range(len(args.extractor_size)):
            extractor_input = tf.nn.relu(tf.nn.xw_plus_b(extractor_input, self.extractor_w[i], self.extractor_b[i]))
        output = tf.nn.xw_plus_b(extractor_input, self.extractor_w[-1], self.extractor_b[-1])
        return output

    def _create_feature_extractor(self, args):
        """Create feature extractor (maps state -> features, assumes feature same dimensionality as latent states)
        Args:
            args: Various arguments and specifications
        """
        features = self._get_extractor_output(args, self.x)
        self.features = tf.reshape(features, [args.batch_size, 2*args.seq_length, args.latent_dim])

    def _create_temporal_encoder(self, args):
        """Bidirectional LSTM to generate temporal encoding (also generate distribution over g1 here)
        Args:
            args: Various arguments and specifications
        """
        # Define forward and backward layers
        fwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        bwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())

        # Construct input -- concatenate sequence of states with sequence of actions
        padded_u = tf.concat([tf.zeros([args.batch_size, 1, args.action_dim]), self.u[:, :(args.seq_length-1)]], axis=1)
        rnn_input = tf.concat([self.features[:, :args.seq_length], padded_u], axis=2)
        
        # Get outputs from rnn and concatenate
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, rnn_input, dtype=tf.float32)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw[:, -1], output_bw[:, -1]], axis=1)

        # Single transformation and affine layer into temporal encoding
        hidden = tf.layers.dense(output, 
                                units=args.transform_size, 
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.temporal_encoding = tf.layers.dense(hidden, 
                                    units=args.latent_dim, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

        # Now construct distribution over g1 through transformation with single hidden layer
        g_input = tf.concat([self.temporal_encoding, self.features[:, 0]], axis=1)
        hidden = tf.layers.dense(g_input, 
                                    units=args.transform_size, 
                                    activation=tf.nn.relu, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.g1_dist = tf.layers.dense(hidden, 
                                        units=2*args.latent_dim,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    def _create_inference_network_params(self, args):
        """Create parameters to comprise inference network
        Args:
            args: Various arguments and specifications
        """
        self.inference_w = []
        self.inference_b = []

        # Loop through elements of inference network and define parameters
        for i in range(len(args.inference_size)):
            if i == 0:
                prev_size = 3*args.latent_dim + args.action_dim
            else:
                prev_size = args.inference_size[i-1]
            self.inference_w.append(tf.get_variable("inference_w"+str(i), [prev_size, args.inference_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.inference_b.append(tf.get_variable("inference_b"+str(i), [args.inference_size[i]]))

        # Last set of weights to map to output
        self.inference_w.append(tf.get_variable("inference_w_end", [args.inference_size[-1], 2*args.latent_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.inference_b.append(tf.get_variable("inference_b_end", [2*args.latent_dim]))
 
    def _get_inference_distribution(self, args, features, u, g_enc):
        """Function to infer distribution over g
        Args:
            args: Various arguments and specifications
            features: Extracted features [batch_size, latent_dim]
            u: Control input [batch_size, action_dim]
            g_enc: Temporal encoding of previous g-values [batch_size, latent_dim]
        Returns:
            Next g-value
        """
        inference_input = tf.concat([features, u, self.temporal_encoding, g_enc], axis=1)
        for i in range(len(args.inference_size)):
            inference_input = tf.nn.relu(tf.nn.xw_plus_b(inference_input, self.inference_w[i], self.inference_b[i]))
        g_dist = tf.nn.xw_plus_b(inference_input, self.inference_w[-1], self.inference_b[-1])
        return g_dist

    def _gen_sample(self, args, dist_params):
        """Function to generate samples given distribution parameters
        Args:
            args: Various arguments and specifications
            dist_params: Mean and logstd of distribution [batch_size, 2*latent_dim]
        Returns:
            g: Sampled g-value [batch_size, latent_dim]
        """
        g_mean, g_logstd = tf.split(dist_params, [args.latent_dim, args.latent_dim], axis=1)

        # Make standard deviation estimates better conditioned, otherwise could be problem early in training
        g_std = tf.minimum(tf.exp(g_logstd) + 1e-6, 10.0) 
        samples = tf.random_normal([args.batch_size, args.latent_dim], seed=args.seed)
        g = samples*g_std + g_mean
        return g

    def _infer_observations(self, args):
        """Step through time and determine g_t distributions and values
        Args:
            args: Various arguments and specifications
        """
        # Sample value for initial observation from distribution
        self.g_t = self._gen_sample(args, self.g1_dist)

        # Start list of g-distributions and sampled values
        self.g_vals = [tf.expand_dims(self.g_t, axis=1)]
        self.g_dists = [tf.expand_dims(self.g1_dist, axis=1)]

        # Create parameters for transformation to be performed at output of GRU in observation encoder
        W_g_out = tf.get_variable("w_g_out", [args.rnn_size, args.transform_size], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        b_g_out = tf.get_variable("b_g_out", [args.transform_size])
        W_to_g_enc = tf.get_variable("w_to_g_enc", [args.transform_size, args.latent_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        b_to_g_enc = tf.get_variable("b_to_g_enc", [args.latent_dim])

        # Initialize single-layer GRU network to create observation encodings
        cell = tf.nn.rnn_cell.GRUCell(args.rnn_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.rnn_state = cell.zero_state(args.batch_size, tf.float32)
        g_t = self.g_t

        for t in range(1, args.seq_length):
            # Generate temporal encoding
            self.rnn_output, self.rnn_state = cell(g_t, self.rnn_state)
            hidden = tf.nn.relu(tf.nn.xw_plus_b(self.rnn_output, W_g_out, b_g_out))
            g_enc = tf.nn.xw_plus_b(hidden, W_to_g_enc, b_to_g_enc)

            # Now get distribution over g_t and sample value
            g_dist = self._get_inference_distribution(args, self.features[:, t], self.u[:, t-1], g_enc)
            g_t = self._gen_sample(args, g_dist)

            # Append values to list
            self.g_vals.append(tf.expand_dims(g_t, axis=1))
            self.g_dists.append(tf.expand_dims(g_dist, axis=1))

        # Finally, stack inferred observations
        self.g_vals = tf.reshape(tf.stack(self.g_vals, axis=1), [args.batch_size, args.seq_length, args.latent_dim])
        self.g_dists = tf.reshape(tf.stack(self.g_dists, axis=1), [args.batch_size*args.seq_length, 2*args.latent_dim])

    def _create_prior_network(self, args):
        """Construct network and generate paramaters for conditional prior distributions
        Args:
            args: Various arguments and specifications
        """
        gvals_reshape = tf.reshape(self.g_vals[:, :-1], [args.batch_size*(args.seq_length-1), args.latent_dim])
        u_reshape = tf.reshape(self.u[:, :(args.seq_length-1)], [args.batch_size*(args.seq_length-1), args.action_dim])
        prior_input = tf.concat([gvals_reshape, u_reshape], axis=1)

        # Construct layers of prior network
        for ps in args.prior_size:
            prior_input = tf.layers.dense(prior_input, 
                                            units=ps, 
                                            activation=tf.nn.relu, 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

        # Final affine transform to dist params
        prior_params = tf.layers.dense(prior_input, 
                                            units=2*args.latent_dim,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        prior_params = tf.reshape(prior_params, [args.batch_size, args.seq_length-1, 2*args.latent_dim])

        # Construct diagonal unit Gaussian prior params for g1
        g1_prior = tf.concat([tf.zeros([args.batch_size, 1, args.latent_dim]), tf.ones([args.batch_size, 1, args.latent_dim])], axis=2)

        # Combine and reshape to get full set of prior distribution parameter values
        g_prior = tf.concat([g1_prior, prior_params], axis=1)

        # Combine and reshape to get full set of prior distribution parameter values
        self.g_prior = tf.reshape(g_prior, [args.batch_size*args.seq_length, 2*args.latent_dim])

    def _propagate_solution(self, args):
        """Perform least squares to get A- and B-matrices and propagate forward
        Args:
            args: Various arguments and specifications
        """
        # Define X- and Y-matrices
        X = tf.concat([self.g_vals[:, :-1], self.u[:, :(args.seq_length-1)]], axis=2)
        Y = self.g_vals[:, 1:]

        # Solve for A and B using least-squares
        self.K = tf.matrix_solve_ls(X, Y, l2_regularizer=args.l2_regularizer)
        self.A = self.K[:, :args.latent_dim]
        self.B = self.K[:, args.latent_dim:]

        # Perform least squares to find A-inverse
        self.A_inv = tf.matrix_solve_ls(Y - tf.matmul(self.u[:, :(args.seq_length-1)], self.B), self.g_vals[:, :-1], l2_regularizer=args.l2_regularizer)
        
        # Get predicted code at final time step
        self.z_t = self.g_vals[:, -1]

        # Create recursive predictions for z
        z_t = tf.expand_dims(self.z_t, axis=1)
        z_vals = [z_t]
        for t in range(args.seq_length-2, -1, -1):
            u = self.u[:, t]
            u = tf.expand_dims(u, axis=1)
            z_t = tf.matmul(z_t - tf.matmul(u, self.B), self.A_inv)
            z_vals.append(z_t) 
        self.z_vals_reshape = tf.stack(z_vals, axis=1)

        # Flip order
        self.z_vals_reshape = tf.squeeze(tf.reverse(self.z_vals_reshape, [1]))

        # Reshape predicted z-values
        self.z_vals = tf.reshape(self.z_vals_reshape, [args.batch_size*args.seq_length, args.latent_dim])

    def _create_decoder_params(self, args):
        """Create parameters to comprise decoder network
        Args:
            args: Various arguments and specifications
        """
        self.decoder_w = []
        self.decoder_b = []

        # Loop through elements of decoder network and define parameters
        for i in range(len(args.extractor_size)-1, -1, -1):
            if i == len(args.extractor_size)-1:
                prev_size = args.latent_dim
            else:
                prev_size = args.extractor_size[i+1]
            self.decoder_w.append(tf.get_variable("decoder_w"+str(len(args.extractor_size)-i), [prev_size, args.extractor_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.decoder_b.append(tf.get_variable("decoder_b"+str(len(args.extractor_size)-i), [args.extractor_size[i]]))

        # Last set of weights to map to output
        self.decoder_w.append(tf.get_variable("decoder_w_end", [args.extractor_size[0], args.state_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.decoder_b.append(tf.get_variable("decoder_b_end", [args.state_dim]))

    def _get_decoder_output(self, args, encodings):
        """Function to run inputs through decoder
        Args:
            args: Various arguments and specifications
            encodings: Input to decoder [2*batch_size*seq_length, latent_dim]
        Returns:
            output: Reconstructed states [2*batch_size*seq_length, state_dim]
        """
        decoder_input = encodings
        for i in range(len(args.extractor_size)):
            decoder_input = tf.nn.elu(tf.nn.xw_plus_b(decoder_input, self.decoder_w[i], self.decoder_b[i]))
        output = tf.nn.xw_plus_b(decoder_input, self.decoder_w[-1], self.decoder_b[-1])
        return output

    def _generate_predictions(self, args):
        """Generate predictions for how system will evolve given z1, A, and B (used for control, not during training)
        Args:
            args: Various arguments and specifications
        """
        self.z1 = tf.squeeze(self.z_vals_reshape[:, -1])
        z_t = tf.expand_dims(self.z1, axis=1)
        z_pred = [z_t]
        for t in range(args.seq_length, 2*args.seq_length):
            u = self.u[:, t-1]
            u = tf.expand_dims(u, axis=1)
            z_t = tf.matmul(z_t, self.A) + tf.matmul(u, self.B)
            z_pred.append(z_t) 
        z_pred = tf.stack(z_pred, axis=1)

        # Reshape predicted z-values
        self.z_pred_reshape = z_pred[:, 1:]
        z_pred = tf.reshape(z_pred[:, 1:], [args.batch_size*args.seq_length, args.latent_dim]) 
        self.x_future_norm = tf.reshape(self._get_decoder_output(args, z_pred), [args.batch_size, args.seq_length, args.state_dim])
        self.x_future = self.x_future_norm*self.scale + self.shift

    def _get_cost(self, args, z_u_t):
        """Get cost associated with a set of states and actions
        Args:
            args: Various arguments and specifications
            z_u_t: Latent state and control input at given time step [batch_size, state_dim+action_dim]
        Returns:
            Cost [batch_size]
        """
        z_t = z_u_t[:, :args.latent_dim]
        u_t = z_u_t[:, args.latent_dim:]
        states = self._get_decoder_output(args, z_t)*self.scale + self.shift
        if args.domain_name == 'Pendulum-v0':
            return tf.square(tf.atan2(states[:, 1], states[:, 0])) + 0.1*tf.square(states[:, 2]) + 0.001*tf.square(tf.squeeze(u_t))
        else:
            raise NotImplementedError

    def _find_ilqr_params(self, args):
        """Find necessary params to perform iLQR
        Args:
            args: Various arguments and specifications
        """
        # Initialize state
        z_t = self.z1

        # Initialize lists to hold quantities
        L = []
        L_x = []
        L_u = []
        L_xx = []
        L_ux = []
        L_uu = [] 
        z_vals = [z_t]

        # Loop through time
        for t in range(args.mpc_horizon):
            # Find cost for current state
            z_u_t = tf.concat([z_t, self.u_ilqr[:, t]], axis=1)
            l_t = args.gamma**t*self._get_cost(args, z_u_t)

            # Find gradients and Hessians (think you need to compute Hessians this way because it handles 3d tensors weirdly)
            grads = tf.gradients(l_t, z_u_t)[0]
            hessians = tf.reduce_sum(tf.hessians(l_t, z_u_t)[0], axis=2)              

            # Separate into individual components
            l_x = grads[:, :args.latent_dim]
            l_u = grads[:, args.latent_dim:]
            l_xx = hessians[:, :args.latent_dim, :args.latent_dim]
            l_ux = hessians[:, args.latent_dim:, :args.latent_dim]
            l_uu = hessians[:, args.latent_dim:, args.latent_dim:]

            # Append to lists
            L.append(l_t)
            L_x.append(l_x)
            L_u.append(l_u)
            L_xx.append(l_xx)
            L_ux.append(l_ux)
            L_uu.append(l_uu)

            # Find action by passing it through tanh
            u_t = args.action_max*tf.nn.tanh(self.u_ilqr[:, t])
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.squeeze(tf.matmul(tf.expand_dims(z_t, axis=1), self.A) + tf.matmul(u_t, self.B))
            z_vals.append(z_t)

        # Find cost and gradients at last time step
        z_u_t = tf.concat([z_t, tf.zeros_like(self.u_ilqr[:, -1])], axis=1)
        l_T = args.gamma**args.seq_length*self._get_cost(args, z_u_t)
        grads = tf.gradients(l_T, z_u_t)[0]
        hessians = tf.reduce_sum(tf.hessians(l_T, z_u_t)[0], axis=2)  
        L.append(l_T)
        L_x.append(grads[:, :args.latent_dim])
        L_xx.append(hessians[:, :args.latent_dim, :args.latent_dim])

        # Finally stack into tensors
        self.L = tf.stack(L, axis=1)
        self.L_x = tf.stack(L_x, axis=1)
        self.L_u = tf.stack(L_u, axis=1)
        self.L_xx = tf.stack(L_xx, axis=1)
        self.L_ux = tf.stack(L_ux, axis=1)
        self.L_uu = tf.stack(L_uu, axis=1)
        self.xs = tf.stack(z_vals, axis=1)
        states_pred = self._get_decoder_output(args, tf.reshape(self.xs, [-1, args.latent_dim]))*self.scale + self.shift
        self.states_pred = tf.reshape(states_pred, [args.batch_size, -1, args.state_dim])

    def _create_optimizer(self, args):
        """Create optimizer to minimize loss
        Args:
            args: Various arguments and specifications
        """
        # First extract mean and std for prior dists, dist over g, and dist over x
        g_prior_mean, g_prior_logstd = tf.split(self.g_prior, [args.latent_dim, args.latent_dim], axis=1)
        g_prior_std = tf.exp(g_prior_logstd) + 1e-6
        g_mean, g_logstd = tf.split(self.g_dists, [args.latent_dim, args.latent_dim], axis=1)
        g_std = tf.exp(g_logstd) + 1e-6

        # Get predictions for x and reconstructions
        self.x_pred_norm = self._get_decoder_output(args, self.z_vals)
        self.x_pred = self.x_pred_norm*self.scale + self.shift

        # First component of loss: NLL of observed states
        x_reshape = tf.reshape(self.x, [args.batch_size, 2*args.seq_length, args.state_dim])
        x_pred_reshape = tf.reshape(self.x_pred_norm, [args.batch_size, args.seq_length, args.state_dim])
        self.x_pred_init = x_pred_reshape*self.scale + self.shift # needed for ilqr

        # Add in predictions for how system will evolve
        self.x_pred_reshape = tf.concat([x_pred_reshape, self.x_future_norm], axis=1)
        self.x_pred_reshape_unnorm = self.x_pred_reshape*self.scale + self.shift

        # Prediction loss
        self.pred_loss = tf.reduce_sum(tf.square(x_reshape - self.x_pred_reshape))
    
        # Weight loss at t = T more heavily
        self.pred_loss += 20.0*tf.reduce_sum(tf.square(x_reshape[:, args.seq_length-1]\
                                                             - x_pred_reshape[:, args.seq_length-1]))

        # Define reconstructed state needed for ilqr
        self.rec_state = self._get_decoder_output(args, self.z1)*self.scale + self.shift

        # Second component of loss: KLD between approximate posterior and prior
        g_prior_dist = tf.distributions.Normal(loc=g_prior_mean, scale=g_prior_std)
        g_dist = tf.distributions.Normal(loc=g_mean, scale=g_std)
        self.kl_loss = tf.reduce_sum(tf.distributions.kl_divergence(g_dist, g_prior_dist))

        # Sum with regularization losses to form total cost
        self.cost = self.pred_loss + self.kl_weight*self.kl_loss + tf.reduce_sum(tf.losses.get_regularization_losses())  

        # Perform parameter update
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = [v for v in tf.trainable_variables()]
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))

