import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, input_dem, output_dem, bias=True):
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (input_dem + output_dem))

        self.w = tf.Variable(
            rng.normal(shape=[input_dem, output_dem], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(shape=[1, output_dem]),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


class BasisExpansion(tf.Module):
    def __init__(self, M):
        self.mu = tf.Variable(
            tf.random.uniform(shape=[1, M], minval=0.0, maxval=1.0),
            trainable=True, 
            name="Linear/mu"
        )

        mu_spacing = (2.0*m.pi) / (M-1)
        self.sigma = tf.Variable(
            tf.ones(shape=[1, M]) * 0.1, trainable=True, name="sigma"
        )

    def __call__(self, x):

        phi = tf.exp(-((x - self.mu) ** 2) / (self.sigma) ** 2)

        return phi


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse
    import math as m
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    noise = config["data"]["noise_stddev"]

    data_x = rng.uniform(shape=[num_samples, 1], minval=0, maxval=1)
    data_y = rng.normal(
        shape=[num_samples, 1],
        mean=tf.math.sin(2 * tf.constant(m.pi) * data_x),
        stddev=noise,
    )

    M = 4
    basis = BasisExpansion(M)

    linear = Linear(M, 1)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    for i in trange(num_iters):
        batch_indices = rng.uniform(
            shape=[batch_size], minval=0, maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(data_x, batch_indices)
            y_batch = tf.gather(data_y, batch_indices)

            phi = basis(x_batch)
            y_hat = linear(phi)
            loss = tf.reduce_mean((1 / 2) * (y_batch - y_hat) ** 2)

        variables = linear.trainable_variables + basis.trainable_variables
        grads = tape.gradient(loss, variables)
        grad_update(step_size, variables, grads)

        step_size *= decay_rate

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(data_x, data_y, "x")
    a = tf.linspace(tf.reduce_min(data_x), tf.reduce_max(data_x), num_samples)[
        :, tf.newaxis
    ]
    phi_a = basis(a)
    ax[0].plot(a, tf.math.sin(2 * tf.constant(m.pi) * a))
    ax[0].plot(a, linear(phi_a), ":")

    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Linear fit using SGD")

    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    for i in range(M):
        ax[1].plot(a, phi_a[:, i])

    ax[1].set_title("Basis for Model")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    fig.savefig("linear.pdf")
