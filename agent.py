import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import variance_scaling_initializer


class PolicyGradientAgent:

    def __init__(self):
        self.n_inputs = 4
        self.n_hidden = 4
        self.n_outputs = 1
        self.initializer = variance_scaling_initializer()
        self.learning_rate = 0.01

    def create_model(self, X):
        hidden = tf.layers.dense(X, self.n_hidden, activation=tf.nn.elu, kernel_initializer=self.initializer)
        logits = tf.layers.dense(hidden, self.n_outputs)
        outputs = tf.nn.sigmoid(logits)
        p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
        action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
        return action, logits

    def optimize_model(self, action, logits):
        y = 1 - tf.to_float(action)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        gradients = [grad for grad, variable in grads_and_vars]
        gradient_placeholders = []
        grads_and_vars_feed = []

        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))

        training_op = optimizer.apply_gradients(grads_and_vars_feed)
        return gradients, gradient_placeholders, training_op

    def discount_rewards(self, rewards, discount_rate):
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(self, all_rewards, discount_rate):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def compute_gradient(self, gradient_placeholders, all_gradients, all_rewards):
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        return feed_dict
