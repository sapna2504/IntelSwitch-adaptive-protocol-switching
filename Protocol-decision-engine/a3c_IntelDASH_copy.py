import numpy as np
import tensorflow as tf
import tflearn


GAMMA = 0.99
A_DIM = 2
ENTROPY_WEIGHT = 0.03
ENTROPY_EPS = 1e-6
S_INFO = 4

"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------
The "actor" network determines the action to take.
The "critic" network evaluates actions.
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Placeholders are nodes whose values are fed at execution time: If you have inputs into your computational graph that depend on some external data, 
these are placeholders for values that we are going to add into our computation during training. So, for placeholders, we don't provide any initial values.
--------------------------------------------------------------------------------------------------------------------------------------------------------------
self.input_network_params: List of placeholders to receive external values for the model parameters.
self.set_network_params_op: List of assignment operations to update the model’s parameters using the provided values.
--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
 


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        # sess: Tensorflow session for running computation, state_dim: input size, action_dim: output size, learning_rate: learning rate for training.
        print("welcome to a3c_inteldash")
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network which is a neural network with input self.inputs and produces self.out which represents the action probability distribution.
        self.inputs, self.out = self.create_actor_network()

        # This line of TensorFlow code retrieves all trainable variables under the scope 'actor' and assigns them to self.network_params.
        # tf.compat.v1.get_collection(...) is a function that retrieves a collection of TensorFlow objects (like variables, tensors, etc.). 
        # tf.compat.v1 is used for compatibility with TensorFlow 2.x
        # tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES: This specifies that we are retrieving only trainable variables, meaning those involved in the optimization process (e.g., weights and biases).
        self.network_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        
         # Set all network parameters
        self.input_network_params = [] # This initializes an empty list, self.input_network_params, which will later store TensorFlow placeholders.

        for param in self.network_params:
            self.input_network_params.append(
                tf.compat.v1.placeholder(tf.float32, shape=param.get_shape())) # The placeholder has the data type of the parameter (float values) and matches the shape of the original parameter.

        # This initializes an empty list, self.set_network_params_op, which will store TensorFlow assignment operations. These operations will be used later to update the model's parameters using the values provided in self.input_network_params
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param)) # This assigns the new value (from the placeholder) to the corresponding model parameter.

        # Selected action, 0-1 vector, Selected action stored as a one-hot vector (this is the original format).
        # This line creates a TensorFlow placeholder for storing action inputs in a reinforcement learning (RL) model.
        # self.a_dim: This represents the action space dimension, meaning how many action values are expected for each sample.
        self.acts = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        # Advantage estimates – originally stored as [batch, 1]
        self.advantage = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # new line added for PPO

        # OLD log probabilities for the chosen actions.
        # CHANGED: We now use a scalar per sample (shape [None]) instead of a vector.
        self.old_log_prob = tf.compat.v1.placeholder(tf.float32, shape=[None])

        # This part of code was used in policy gradient and no more a part of PPO
        '''
        # This gradient will be provided by the critic network
        # This line creates a TensorFlow placeholder to store gradient weights for actions, typically used in policy gradient methods or actor-critic reinforcement learning algorithms.
        # self.act_grad_weights hold gradient scaling weights, which are often used to adjust the magnitude of policy updates.
        # 1: Indicates that each sample has a single weight value
        # self.act_grad_weights = tf.compat.v1.placeholder(tf.float32, [None, 1])
        '''

        # --- PPO LOSS IMPLEMENTATION ---
        # Instead of using a tfp distribution we manually compute the log probability
        # and entropy using the softmax output.
        # (This retains your original approach of outputting probabilities as floats.)
        #
        # Compute the new log probability of the chosen action.
        # Here, since "acts" is one-hot and "self.out" is the softmax output,
        # tf.reduce_sum(self.out * self.acts, axis=1) picks out the probability of the chosen action.
        new_log_prob = tf.math.log(tf.reduce_sum(self.out * self.acts, axis=1) + ENTROPY_EPS)

        # Reshape advantage from [None,1] to [None] so it can be multiplied with our ratio.
        advantage_flat = tf.reshape(self.advantage, [-1])

        # Compute the probability ratio (new vs. old)
        ratio = tf.exp(new_log_prob - self.old_log_prob)
        clip_range = 0.2 
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)
        # PPO surrogate loss: take the minimum between unclipped and clipped objectives.
        surrogate_loss = tf.minimum(ratio * advantage_flat, clipped_ratio * advantage_flat)

        # Compute the entropy bonus (to encourage exploration).
        # The entropy of the softmax output is computed as: –sum(p * log p) per sample.
        entropy = -tf.reduce_sum(self.out * tf.math.log(self.out + ENTROPY_EPS), axis=1)

        # Total PPO loss: we want to maximize the surrogate objective plus entropy,
        # so we minimize the negative of that sum.
        self.obj = -tf.reduce_mean(surrogate_loss + ENTROPY_WEIGHT * entropy)
        # --- END PPO LOSS IMPLEMENTATION ---

        '''
        # Compute the objective (log action_vector and entropy)
        # tf.reduce_sum(tf.multiply(self.out, self.acts),reduction_indices=1, keep_dims=True)) : This performs an element-wise multiplication to extract the probability of the selected action.
        # self.out: The output of the actor network, representing the action probability distribution \pi(a | s)π(a∣s).
        # self.acts: The placeholder holding the selected action in one-hot form.
        # Takes the logarithm of the probability of the selected action. Used in policy gradient algorithm
        # self.act_grad_weights typically holds the advantage function A(s, a)A(s,a), which tells how good an action was. Since we are maximizing the objective, we negate the loss term.
        # self.obj = tf.reduce_sum(tf.multiply(
        #                tf.compat.v1.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
        #                                     reduction_indices=1, keep_dims=True)),
        #                -self.act_grad_weights)) \
        #            + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
        #                                                    tf.log(self.out + ENTROPY_EPS)))
        '''

        # Combine the gradients here
        # This line computes the gradients of the objective function (self.obj) with respect to the actor network's parameters (self.network_params). 
        # These gradients are then used to update the parameters during training. tf.gradients(y, x) computes partial derivatives of y with respect to x. Returns a list of gradient tensors, one for each variable in x.
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        # RMSProp maintains a moving average of squared gradients and divides the gradient update by the square root of this average.
        # Prevents the exploding or vanishing gradient problem. RMSPropOptimizer helps stabilize training by adjusting learning rates adaptively.
        # Adapts learning rates dynamically, making it well-suited for reinforcement learning, where gradients are noisy.
        # self.lr_rate: The learning rate, controlling how much the parameters change in each step.
        # zip(self.actor_gradients, self.network_params) pairs each gradient with its corresponding parameter.
        # apply_gradients(...) updates the actor network weights based on these gradients.
        # apply_gradients(...) updates the actor network weights based on these gradients.
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))
        

    """
    ------------------------------------------------------------------------------------------
    Fully connected layers (dense layers) work well when the data is not sequential or does not have strong temporal/spatial relationships.
    Split 0 & 1: Could be values like current bitrate, buffer length, network throughput, or latency—these are individual scalar values that don’t require sequence processing.
    Dense layers are used here because each extracted value is independent, and we just need a non-linear transformation.
    rows for which conv_1d is applied These rows likely represent sequential data (e.g., time-series data or action-history).
    Convolutional layers are useful when there is local structure in the data (e.g., patterns in time-series data).
    Split 2 & 3: Could represent past buffer values or past throughput values—these evolve over time and have short-term dependencies.
    Split 4: Might represent a sequence of previous actions taken (e.g., past bitrates chosen in adaptive streaming).
    conv_1d layers are used here because: They help detect patterns (e.g., trends in throughput, buffer variations). 
    They extract local features (e.g., how recent throughput values affect the next decision).
    They preserve order in the sequence.
    Since convolutional layers output multi-dimensional tensors, we flatten them to convert them into a single feature vector, which can be merged with other features.
    Some features (split_0, split_1, split_5) are static scalar values, so fully connected layers work best.
    Other features (split_2, split_3, split_4) are sequential/temporal, so conv_1d layers help detect patterns.
    """
    

    def create_actor_network(self):
        #Creates a variable scope named 'actor', ensuring all variables belong to this network.
        with tf.compat.v1.variable_scope('actor'):
            # The state input is a 3D tensor: None → Allows dynamic batch size., self.s_dim[0] → First dimension of the state., self.s_dim[1] → Second dimension of the state.
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            # Extracts specific rows (0:1, 1:2) from the input along the first dimension.
            # Selects the last element (-1) in the second dimension (the last column).
            # Passes the result through a fully connected (dense) layer with 128 neurons and ReLU activation.
            
            # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 64, activation='relu')
            # split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 64, activation='relu')
            # split_2 = tflearn.fully_connected(inputs[:, 2:3, -1], 64, activation='relu')
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 64, 4, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 64, activation='relu')
            # split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 64, 4, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 64, 4, activation='relu')
            # split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 64, 4, activation='relu')
            # split_4 = tflearn.conv_1d(inputs[:, 4:5, :], 64, 4, activation='relu')
            
            split_0_flat = tflearn.flatten(split_0)
            # split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            # split_3_flat = tflearn.flatten(split_3)
            # split_4_flat = tflearn.flatten(split_4)
            

            # Concatenates ('concat') all processed parts into one vector.
            merge_net = tflearn.merge([split_0_flat, split_1, split_2_flat], 'concat')
            # merge_net = tflearn.merge([split_0, split_1, split_2], 'concat')

            # Applies a fully connected layer with 128 neurons and ReLU activation.
            # This acts as a hidden layer for final feature processing.
            dense_net_0 = tflearn.fully_connected(merge_net, 64, activation='relu')
            # Fully connected layer with self.a_dim neurons (number of possible actions).
            # Softmax activation ensures the output is a probability distribution over actions.
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')
            print("create critic mai ayaaaaaaaaaaaaa", out)
            return inputs, out
        
    # This function trains the actor network by updating its parameters using gradient-based optimization.
    # inputs: The state input to the actor network, which represents the environment’s current observation.
    # acts: The actions taken by the agent (one-hot encoded or probability distribution over actions).
    # act_grad_weights: The gradients of the action values (used for policy optimization in reinforcement learning).
    def train(self, inputs, acts, advantages, old_log_probs):
        # Executes the optimization operation (self.optimize), which was defined earlier using RMSPropOptimizer.
        # Feeds actual training data (inputs, acts, and act_grad_weights) into TensorFlow placeholders.
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            #self.act_grad_weights: act_grad_weights
            self.advantage: advantages,
            self.old_log_prob: old_log_probs
        })

    # This function is used to generate action predictions from the actor network based on a given input state.
    # inputs: Represents the current state of the environment (observations).
    # Returns: The output of the actor network (self.out), which is a probability distribution over possible actions.
    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })


    # This function computes the policy gradients for the actor network based on given inputs (states, actions, and action gradient weights). 
    # It is used in policy gradient reinforcement learning to update the policy parameters.
    # inputs: The current state of the environment (observations).
    # acts: The actions taken by the agent (one-hot encoded or probability distribution).
    # act_grad_weights: The gradients of the value function with respect to the action probabilities (used in actor-critic methods).
    # Returns: The computed gradients of the actor network parameters.
    def get_gradients(self, inputs, acts, act_grad_weights, old_log_probs):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.advantage: act_grad_weights, # "act_grad_weights" here are used as advantage estimates
            self.old_log_prob: old_log_probs   # Missing placeholder
        })

    # This function applies the computed policy gradients to update the parameters of the actor network. 
    # It takes the computed gradients and performs an optimization step using RMSPropOptimizer.
    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    # These two functions handle saving and restoring the parameters (weights and biases) of the actor network in a reinforcement learning setting.
    # Synchronizing multiple actor networks (e.g., in asynchronous methods like A3C).
    # Transferring a trained policy to another instance.
    # Saving and loading models in reinforcement learning.
    # Helps in transferring a trained policy between agents.
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
    
    # --- CHANGED: Updated get_log_prob to manually compute the log probability ---
    def get_log_prob(self, inputs, acts):
        # acts are expected as one-hot vectors.
        log_prob = self.sess.run(
            tf.math.log(tf.reduce_sum(self.out * acts, axis=1) + ENTROPY_EPS),
            feed_dict={self.inputs: inputs}
        )
        return log_prob
    # --- End Change ---

"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        
        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.compat.v1.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

         # Network target V(s)
        self.td_target = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # # Mean square error
        # self.loss = tflearn.mean_square(self.td_target, self.out)

         # Mean square error loss for the critic because reduce_mean is faster in modern gpu's
        self.loss = tf.reduce_mean(tf.square(self.td_target - self.out))

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))


    def create_critic_network(self):
        with tf.compat.v1.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            # Passes the result through a fully connected (dense) layer with 128 neurons and ReLU activation.
            # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 64, activation='relu')
            # split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 64, activation='relu')
            # split_2 = tflearn.fully_connected(inputs[:, 2:3, -1], 64, activation='relu')
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 64, 4, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 64, activation='relu')
            # split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 64, 4, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 64, 4, activation='relu')
            # split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 64, 4, activation='relu')
            # split_4 = tflearn.conv_1d(inputs[:, 4:5, :], 64, 4, activation='relu')
            
            split_0_flat = tflearn.flatten(split_0)
            # split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            # split_3_flat = tflearn.flatten(split_3)
            # split_4_flat = tflearn.flatten(split_4)
            

            # Concatenates ('concat') all processed parts into one vector.
            merge_net = tflearn.merge([split_0_flat, split_1, split_2_flat], 'concat')


            dense_net_0 = tflearn.fully_connected(merge_net, 64, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return inputs, out
        

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })
    

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })
    
    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })
    
    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })
    
    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    # Fetches the current trainable parameters as NumPy arrays.
    def get_network_params(self):
        return self.sess.run(self.network_params)

    # Updates the actor network's parameters using externally provided values.
    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

#Monte-Carlo based advantage Function
# def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
#     """
#     Given batches of s, a, r from a sequence, compute the gradients.
#     """
#     assert s_batch.shape[0] == a_batch.shape[0]
#     assert s_batch.shape[0] == r_batch.shape[0]
#     ba_size = s_batch.shape[0]

#     v_batch = critic.predict(s_batch)
#     R_batch = np.zeros(r_batch.shape)   

#     if terminal:
#         R_batch[-1, 0] = 0  # terminal state
#     else:
#         R_batch[-1, 0] = v_batch[-1, 0]  # bootstrap from last state

#     for t in reversed(range(ba_size - 1)):
#         R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

#     td_batch = R_batch - v_batch

#     # Get gradients from actor and critic.
#     old_log_probs = actor.get_log_prob(s_batch, a_batch)
#     actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch,old_log_probs)
#     critic_gradients = critic.get_gradients(s_batch, R_batch)
    

#     return actor_gradients, critic_gradients, td_batch, old_log_probs


#GAE(Generalized Advantage Estimation) based advantage Function
def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic, gamma=0.99, lam=0.95):
    """
    Given batches of s, a, r from a sequence, compute the gradients.
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    # Value predictions
    v_batch = critic.predict(s_batch)
    v_next = np.zeros_like(v_batch)  

    if terminal:
        v_next[-1, 0] = 0  # terminal state
    else:
        v_next[-1, 0] = critic.predict(np.expand_dims(s_batch[-1], axis=0))[0, 0]

    # Shift v_batch to get V(s_{t+1})
    v_next[:-1] = v_batch[1:]
    v_next = v_next.reshape(-1, 1)

    # Temporal difference errors
    deltas = r_batch + gamma * v_next - v_batch

    # GAE advantage computation
    advantages = np.zeros_like(r_batch)
    gae = 0
    for t in reversed(range(ba_size)):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae

    R_batch = advantages + v_batch

    # Get gradients from actor and critic.
    old_log_probs = actor.get_log_prob(s_batch, a_batch)
    actor_gradients = actor.get_gradients(s_batch, a_batch, advantages, old_log_probs)
    critic_gradients = critic.get_gradients(s_batch, R_batch)
    

    return actor_gradients, critic_gradients, advantages, old_log_probs



def discount(x, gamma):
    print("discount mai ayaaaaaaaaaaaaaaaaaaaaaa")
    """
    This function computes the discounted sum of a given vector x using a discount factor gamma. This is a key operation in reinforcement learning (RL), often used for reward discounting n policy gradient methods.
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

# This function computes the entropy of a probability distribution vector x.
# Entropy measures uncertainty in a probability distribution.
# If entropy is high, the distribution is more spread out (e.g., uniform distribution).
# If entropy is low, the distribution is more certain (e.g., one value dominates).
# Higher entropy → More randomness in action selection.
# Used in policy gradient methods and information theory.
def compute_entropy(x):
    print("compute entropy mai ayaaaaaaaaaaaaaaaaaaaaaa")
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


# This function creates TensorFlow summary operations to track key metrics during training. These summaries help in visualizing training progress using TensorBoard.
# def build_summaries():
#     # td_loss: Tracks Temporal Difference (TD) loss, a key loss function in reinforcement learning
#     td_loss = tf.compat.v1.Variable(0.)
#     tf.compat.v1.summary.scalar("TD_loss", td_loss)
#     # eps_total_reward: Stores total reward per episode.
#     eps_total_reward = tf.compat.v1.Variable(0.)
#     tf.compat.v1.summary.scalar("Eps_total_reward", eps_total_reward)
#     # avg_entropy: Tracks average entropy, measuring policy randomness (higher entropy = more exploration).
#     avg_entropy = tf.compat.v1.Variable(0.)
#     tf.compat.v1.summary.scalar("Avg_entropy", avg_entropy)

#     summary_vars = [td_loss, eps_total_reward, avg_entropy]
#     summary_ops = tf.compat.v1.summary.merge_all()
#     print("yaeeeeeeeeeee summaries pr aya" , summary_vars)
#     return summary_ops, summary_vars


def build_summaries():
    # td_loss: Tracks Temporal Difference (TD) loss, a key loss function in reinforcement learning
    td_loss = tf.compat.v1.placeholder(tf.float32, name="td_loss")
    tf.compat.v1.summary.scalar("TD_loss", td_loss)
    # eps_total_reward: Stores total reward per episode.
    eps_total_reward = tf.compat.v1.placeholder(tf.float32, name='Eps_total_reward')
    tf.compat.v1.summary.scalar("Eps_total_reward", eps_total_reward)
    # avg_entropy: Tracks average entropy, measuring policy randomness (higher entropy = more exploration).
    avg_entropy = tf.compat.v1.placeholder(tf.float32, name="entropy")
    tf.compat.v1.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.compat.v1.summary.merge_all()
    print("yaeeeeeeeeeee summaries pr aya" , summary_vars)
    return summary_ops, summary_vars
