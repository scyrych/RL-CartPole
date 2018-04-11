import gym
import tensorflow as tf
from agent import PolicyGradientAgent

n_inputs = 4

pga = PolicyGradientAgent()
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

action, logits = pga.create_model(X)
gradients, gradient_placeholders, training_op = pga.optimize_model(action, logits)

env = gym.make("CartPole-v0")

n_games_per_update = 10
n_max_steps = 1000
n_iterations = 500
save_iterations = 10
discount_rate = 0.95

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = pga.discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = pga.compute_gradient(gradient_placeholders, all_gradients, all_rewards)
        sess.run(training_op, feed_dict=feed_dict)

        if iteration % save_iterations == 0:
            saver.save(sess, "./saved_model/policy_net_pg.ckpt")

env.close()
