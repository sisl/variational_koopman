import math
import numpy as np
import random
import progressbar

# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, shift, scale, shift_u, scale_u, env, net, sess, predict_evolution=False):
        """Constructs object to hold and update training/validation data.
        Args:
            args: Various arguments and specifications
            shift: Shift of state values for normalization
            scale: Scaling of state values for normalization
            shift_u: Shift of action values for normalization
            scale_u: Scaling of action values for normalization
            env: Simulation environment
            net: Neural network dynamics model
            sess: TensorFlow session
            predict_evolution: Whether to predict how system will evolve in time
        """
        self.batch_size = args.batch_size
        self.seq_length = 2*args.seq_length if predict_evolution else args.seq_length
        self.shift_x = shift
        self.scale_x = scale
        self.shift_u = shift_u
        self.scale_u = scale_u
        self.env = env
        self.net = net
        self.sess = sess

        print('validation fraction: ', args.val_frac)

        print("generating data...")
        self._generate_data(args)
        self._process_data(args)

        print('creating splits...')
        self._create_split(args)

        print('shifting/scaling data...')
        self._shift_scale(args)

    def _generate_data(self, args):
        """Load data from environment
        Args:
            args: Various arguments and specifications
        """
        # Initialize array to hold states and actions
        x = np.zeros((args.n_trials, args.n_subseq, self.seq_length, args.state_dim), dtype=np.float32)
        u = np.zeros((args.n_trials, args.n_subseq, self.seq_length-1, args.action_dim), dtype=np.float32)

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args.n_trials).start()

        # Define array for dividing trials into subsequences
        stagger = (args.trial_len - self.seq_length)/args.n_subseq
        self.start_idxs = np.linspace(0, stagger*args.n_subseq, args.n_subseq)

        # Loop through episodes
        for i in range(args.n_trials):
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args.trial_len, args.state_dim), dtype=np.float32)
            u_trial = np.zeros((args.trial_len-1, args.action_dim), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            for t in range(1, args.trial_len):
                action = self.env.action_space.sample()  
                u_trial[t-1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])

            # Divide into subsequences
            for j in range(args.n_subseq):
                x[i, j] = x_trial[int(self.start_idxs[j]):(int(self.start_idxs[j])+self.seq_length)]
                u[i, j] = u_trial[int(self.start_idxs[j]):(int(self.start_idxs[j])+self.seq_length-1)]
            bar.update(i)
        bar.finish()

        # Generate test scenario that is double the length of standard sequences
        self.x_test = np.zeros((2*args.seq_length, args.state_dim), dtype=np.float32)
        self.u_test = np.zeros((2*args.seq_length-1, args.action_dim), dtype=np.float32)
        self.x_test[0] = self.env.reset()
        for t in range(1, 2*args.seq_length):
            action = self.env.action_space.sample()
            self.u_test[t-1] = action
            step_info = self.env.step(action)
            self.x_test[t] = np.squeeze(step_info[0])

        # Reshape and trim data sets
        self.x = x.reshape(-1, self.seq_length, args.state_dim)
        self.u = u.reshape(-1, self.seq_length-1, args.action_dim)
        len_x = int(np.floor(len(self.x)/args.batch_size)*args.batch_size)
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

    def _process_data(self, args):
        """Create batch dicts and shuffle data
        Args:
            args: Various arguments and specifications
        """
        # Create batch_dict
        self.batch_dict = {}

        # Print tensor shapes
        print('states: ', self.x.shape)
        print('inputs: ', self.u.shape)
            
        self.batch_dict['states'] = np.zeros((args.batch_size, self.seq_length, args.state_dim))
        self.batch_dict['inputs'] = np.zeros((args.batch_size, self.seq_length-1, args.action_dim))

        # Shuffle data before splitting into train/val
        print('shuffling...')
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.u = self.u[p]

    def _create_split(self, args):
        """Divide data into training/validation sets
        Args:
            args: Various arguments and specifications
        """
        # Compute number of batches
        self.n_batches = len(self.x)//args.batch_size
        self.n_batches_val = int(math.floor(args.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        # Divide into train and validation datasets
        self.x_val = self.x[self.n_batches_train*args.batch_size:]
        self.u_val = self.u[self.n_batches_train*args.batch_size:]
        self.x = self.x[:self.n_batches_train*args.batch_size]
        self.u = self.u[:self.n_batches_train*args.batch_size]

        # Set batch pointer for training and validation sets
        self.reset_batchptr_train()
        self.reset_batchptr_val()

    def _shift_scale(self, args):
        """Shift and scale data to be zero-mean, unit variance
        Args:
            args: Various arguments and specifications
        """
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1))
            self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1))
            self.shift_u = np.mean(self.u[:self.n_batches_train], axis=(0, 1))
            self.scale_u = np.std(self.u[:self.n_batches_train], axis=(0, 1))

            # Remove very small scale values
            self.scale_x[self.scale_x < 1e-6] = 1.0

            # Set u norm params to be 0, 1 for pendulum environment
            if args.domain_name == 'Pendulum-v0':
                self.shift_u = np.zeros_like(self.shift_u)
                self.scale_u = np.ones_like(self.scale_u)

        # Shift and scale values for test sequence
        self.x_test = (self.x_test - self.shift_x)/self.scale_x
        self.u_test = (self.u_test - self.shift_u)/self.scale_u

    def update_data(self, x_new, u_new, val_frac):
        """Update training/validation data
        Args:
            x_new: New state values
            u_new: New control inputs
            val_frac: Fraction of new data to include in validation set
        """
        # First permute data
        p = np.random.permutation(len(x_new))
        x_new = x_new[p]
        u_new = u_new[p]

        # Divide new data into training and validation components
        n_seq_val = max(int(math.floor(val_frac * len(x_new))), 1)
        n_seq_train = len(x_new) - n_seq_val
        x_new_val = x_new[n_seq_train:]
        u_new_val = u_new[n_seq_train:]
        x_new = x_new[:n_seq_train]
        u_new = u_new[:n_seq_train]

        # Now update training and validation data
        self.x = np.concatenate((x_new, self.x), axis=0)
        self.u = np.concatenate((u_new, self.u), axis=0)
        self.x_val = np.concatenate((x_new_val, self.x_val), axis=0)
        self.u_val = np.concatenate((u_new_val, self.u_val), axis=0)

        # Update sizes of train and val sets
        self.n_batches_train = len(self.x)//self.batch_size
        self.n_batches_val = len(self.x_val)//self.batch_size

    def next_batch_train(self):
        """Sample a new batch from training data
        Args:
            None
        Returns:
            batch_dict: Batch of training data
        """
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train*self.batch_size:(self.batchptr_train+1)*self.batch_size]
        self.batch_dict['states'] = (self.x[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u[batch_index] - self.shift_u)/self.scale_u

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    def reset_batchptr_train(self):
        """Reset pointer to first batch in training set
        Args:
            None
        """
        self.batch_permuation_train = np.random.permutation(len(self.x))
        self.batchptr_train = 0

    def next_batch_val(self):
        """Sample a new batch from validation data
        Args:
            None
        Returns:
            batch_dict: Batch of validation data
        """
        # Extract next validation batch
        batch_index = range(self.batchptr_val*self.batch_size,(self.batchptr_val+1)*self.batch_size)
        self.batch_dict['states'] = (self.x_val[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u_val[batch_index] - self.shift_u)/self.scale_u

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    def reset_batchptr_val(self):
        """Reset pointer to first batch in validation set
        Args:
            None
        """
        self.batchptr_val = 0

