import gym
import tensorflow as tf
from agent import PolicyGradientAgent

n_inputs = 4


def show_result(model_path, n_max_steps=3000):

    pga = PolicyGradientAgent()
    env = gym.make("CartPole-v0")
    obs = env.reset()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    action, logits = pga.create_model(X)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            env.render()
            if done:
                break
    env.close()


show_result("./saved_model/policy_net_pg.ckpt")
