import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import numpy as np
import progressbar
import random
import tensorflow as tf
import time

from replay_memory import ReplayMemory
from variational_koopman_model import VariationalKoopman
from utils import visualize_predictions, perform_rollouts

def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',       type=str,   default='./checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',       type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',      type= str,  default='',         help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name',      type= str,  default='koopman_model', help='name of checkpoint files for saving')
    parser.add_argument('--domain_name',    type= str,  default='Pendulum-v0', help='environment name')

    parser.add_argument('--seq_length',     type=int,   default= 32,        help='sequence length for training')
    parser.add_argument('--mpc_horizon',    type=int,   default= 16,        help='horizon to consider for MPC')
    parser.add_argument('--batch_size',     type=int,   default= 16,        help='minibatch size')
    parser.add_argument('--latent_dim',     type=int,   default= 4,         help='dimensionality of code')

    parser.add_argument('--num_epochs',     type=int,   default= 100,       help='number of epochs')
    parser.add_argument('--learning_rate',  type=float, default= 0.0005,    help='learning rate')
    parser.add_argument('--decay_rate',     type=float, default= 0.5,       help='decay rate for learning rate')
    parser.add_argument('--l2_regularizer', type=float, default= 1.0,       help='regularization for least squares')
    parser.add_argument('--grad_clip',      type=float, default= 5.0,       help='clip gradients at this value')

    parser.add_argument('--n_trials',       type=int,   default= 1000,       help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len',      type=int,   default= 256,       help='number of steps in each trial')
    parser.add_argument('--n_subseq',       type=int,   default= 12,        help='number of subsequences to divide each sequence into')
    parser.add_argument('--kl_weight',      type=float, default= 1.0,       help='weight applied to kl-divergence loss')

    ######################################
    #          Network Params            #
    ######################################
    parser.add_argument('--extractor_size', nargs='+', type=int, default=[32],    help='hidden layer sizes in feature extractor/decoder')
    parser.add_argument('--inference_size', nargs='+', type=int, default=[32],    help='hidden layer sizes in feature inference network')
    parser.add_argument('--prior_size',     nargs='+', type=int, default=[32],    help='hidden layer sizes in prior network')
    parser.add_argument('--rnn_size',       type=int,   default= 64,        help='size of RNN layers')
    parser.add_argument('--transform_size', type=int,   default= 64,        help='size of transform layers')
    parser.add_argument('--reg_weight',     type=float, default= 1e-4,      help='weight applied to regularization losses')
    parser.add_argument('--seed',           type=int,   default= 1,         help='random seed for sampling operations')

    #####################################
    #       Addtitional Options         #
    #####################################
    parser.add_argument('--ilqr',           type=bool,  default= False,     help='whether to perform ilqr with the trained model')
    parser.add_argument('--evaluate',       type=bool,  default= False,     help='whether to evaluate trained network')
    parser.add_argument('--perform_mpc',    type=bool,  default= False,     help='whether to perform MPC instead of training')
    parser.add_argument('--worst_case',     type=bool,  default= False,     help='whether to optimize for worst-case cost')
    parser.add_argument('--gamma',          type=float, default= 1.0,       help='discount factor')
    parser.add_argument('--num_models',     type=int,   default= 5,         help='number of models to use in MPC')

    args = parser.parse_args()

    # Set random seed
    random.seed(1)

    # Create environment
    env = gym.make(args.domain_name)

    # Find state and action dimensionality from environment
    args.state_dim = env.observation_space.shape[0]
    if args.domain_name == 'CartPole-v1': args.state_dim += 1 # taking sine and cosine of theta
    args.action_dim = env.action_space.shape[0]
    args.action_max = env.action_space.high[0]

    # Construct model
    net = VariationalKoopman(args)

    # Begin training
    train(args, net, env)

# Train network
def train(args, net, env):

    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Generate data
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        shift_u = sess.run(net.shift_u)
        scale_u = sess.run(net.scale_u)

        # Generate training data
        replay_memory = ReplayMemory(args, shift, scale, shift_u, scale_u, env, net, sess, predict_evolution=True)

        #Function to evaluate loss on validation set
        def val_loss(kl_weight):
            replay_memory.reset_batchptr_val()
            loss = 0.0
            for b in range(replay_memory.n_batches_val):
                # Get inputs
                batch_dict = replay_memory.next_batch_val()
                x = batch_dict['states']
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
                feed_in[net.u] = u
                feed_in[net.kl_weight] = kl_weight

                # Find loss
                feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss/replay_memory.n_batches_val

        # Specify number of times to loop through training procedure
        n_loops = 10 if args.ilqr else 1
        for n in range(n_loops):
            # Re-initialize network parameters if after the first loop (in place of fine-tuning since training is cheap)
            if n >= 1:
                tf.global_variables_initializer().run()

            # Store normalization parameters
            sess.run(tf.assign(net.shift, replay_memory.shift_x))
            sess.run(tf.assign(net.scale, replay_memory.scale_x))
            sess.run(tf.assign(net.shift_u, replay_memory.shift_u))
            sess.run(tf.assign(net.scale_u, replay_memory.scale_u))

            # Initialize variable to track validation score over time
            old_score = 1e20

            # Set initial learning rate and weight on kl divergence
            print('setting learning rate to ', args.learning_rate)
            sess.run(tf.assign(net.learning_rate, args.learning_rate))
            lr = args.learning_rate

            # Define temperature for annealing kl_weight
            anneal_time = 5
            T = anneal_time*replay_memory.n_batches_train

            # Define counting variables
            count = 0
            count_decay = 0
            decay_epochs = []

            # Loop over epochs
            for e in range(args.num_epochs):
                visualize_predictions(args, sess, net, replay_memory, env, e)

                # Initialize loss
                loss = 0.0
                kl_loss = 0.0
                pred_loss = 0.0
                loss_count = 0
                b = 0
                replay_memory.reset_batchptr_train()

                # Loop over batches
                while b < replay_memory.n_batches_train:
                    start = time.time()

                    # Update kl_weight
                    if e < 3:
                        kl_weight = 1e-6
                    else:
                        count += 1
                        kl_weight = min(args.kl_weight, 1e-6 + args.kl_weight*count/float(T))

                    # Get inputs
                    batch_dict = replay_memory.next_batch_train()
                    x = batch_dict['states']
                    u = batch_dict['inputs']

                    # Construct inputs for network
                    feed_in = {}
                    feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
                    feed_in[net.u] = u
                    feed_in[net.kl_weight] = kl_weight

                    # Find loss and perform training operation
                    feed_out = [net.cost, net.kl_loss, net.pred_loss, net.train]
                    out = sess.run(feed_out, feed_in)

                    # Update and display cumulative losses
                    loss += out[0]
                    kl_loss += out[1]
                    pred_loss += out[2]
                    loss_count += 1

                    end = time.time()
                    b += 1

                    # Print loss
                    if (e * replay_memory.n_batches_train + b) % 100 == 0 and b > 0:
                        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, loss/loss_count, end - start))
                        print("{}/{} (epoch {}), pred_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, pred_loss/loss_count, end - start))
                        print("{}/{} (epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, kl_loss/loss_count, end - start))

                        print('')
                        loss = 0.0
                        kl_loss = 0.0
                        pred_loss = 0.0
                        loss_count = 0

                # Evaluate loss on validation set
                score = val_loss(kl_weight)
                print('Validation Loss: {0:f}'.format(score))
                b = 0

                # Set learning rate
                if (old_score - score) < -0.01 and e >= 8:
                    count_decay += 1
                    decay_epochs.append(e)
                    if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                    lr = args.learning_rate * (args.decay_rate ** count_decay)
                    if args.ilqr and lr < 5e-5: break
                    print('setting learning rate to ', lr)
                    sess.run(tf.assign(net.learning_rate, lr))

                # Save model every epoch
                checkpoint_path = os.path.join(args.save_dir, args.save_name + '.ckpt')
                saver.save(sess, checkpoint_path, global_step = e)
                print("model saved to {}".format(checkpoint_path))
            
                old_score = score
            
            # Run trials to evaluate models and generate new training data
            if args.ilqr:
                avg_reward = perform_rollouts(args, net, env, sess, replay_memory)
                print('Number of training batches now is:', replay_memory.n_batches_train)
                if avg_reward > 180.0: break

if __name__ == '__main__':
    main()
