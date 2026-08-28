
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import numpy as np
import onnxruntime as ort

def export_onnx(
    params, act_size, ppo_params, obs_size, output_path="ONNX.onnx"
):
    print(" === EXPORT ONNX === ")

    # inference_fn = make_inference_fn(params, deterministic=True)

    class MLP(tf.keras.Model):
        def __init__(
            self,
            layer_sizes,
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            activate_final=False,
            bias=True,
            layer_norm=False,
            mean_std=None,
        ):
            super().__init__()

            self.layer_sizes = layer_sizes
            self.activation = activation
            self.kernel_init = kernel_init
            self.activate_final = activate_final
            self.bias = bias
            self.layer_norm = layer_norm

            if mean_std is not None:
                self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            else:
                self.mean = None
                self.std = None

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(self.layer_sizes):
                dense_layer = layers.Dense(
                    size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_init,
                    name=f"hidden_{i}",
                    use_bias=self.bias,
                )
                self.mlp_block.add(dense_layer)
                if self.layer_norm:
                    self.mlp_block.add(
                        layers.LayerNormalization(name=f"layer_norm_{i}")
                    )
            if not self.activate_final and self.mlp_block.layers:
                if (
                    hasattr(self.mlp_block.layers[-1], "activation")
                    and self.mlp_block.layers[-1].activation is not None
                ):
                    self.mlp_block.layers[-1].activation = None

            self.submodules = [self.mlp_block]

        def call(self, inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            if self.mean is not None and self.std is not None:
                print(self.mean.shape, self.std.shape)
                inputs = (inputs - self.mean) / self.std
            logits = self.mlp_block(inputs)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    def make_policy_network(
        param_size,
        mean_std,
        hidden_layer_sizes=[256, 256],
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        layer_norm=False,
    ):
        policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            mean_std=mean_std,
        )
        return policy_network

    def extract_policy_params(p):
        policy = getattr(p, "policy", p)
        return policy.get("params") if isinstance(policy, dict) else None

    mean = params[0].mean["state"]
    std = params[0].std["state"]

    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    policy_params = extract_policy_params(params[1])
    dense_layers = sorted(
        policy_params,
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    hidden_layer_sizes = [
        int(np.asarray(policy_params[name]["kernel"]).shape[1])
        for name in dense_layers[:-1]
    ]

    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=tf.nn.swish,
    )

    example_input = tf.zeros((1, obs_size))
    example_output = tf_policy_network(example_input)
    print(example_output.shape)

    def transfer_weights(jax_params, tf_model):
        """
        Transfer weights from a JAX parameter dictionary to the TensorFlow model.

        Parameters:
        - jax_params: dict
        Nested dictionary with structure {block_name: {layer_name: {params}}}.
        For example:
        {
            'CNN_0': {
            'Conv_0': {'kernel': np.ndarray},
            'Conv_1': {'kernel': np.ndarray},
            'Conv_2': {'kernel': np.ndarray},
            },
            'MLP_0': {
            'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
            }
        }

        - tf_model: tf.keras.Model
        An instance of the adapted VisionMLP model containing named submodules and layers.
        """
        for layer_name, layer_params in jax_params.items():
            try:
                tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params["kernel"])
                bias = np.array(layer_params["bias"])
                print(
                    f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}"
                )
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

        print("Weights transferred successfully.")

    transfer_weights(policy_params, tf_policy_network)

    # Example inputs for the model
    test_input = [np.ones((1, obs_size), dtype=np.float32)]

    # Define the TensorFlow input signature
    spec = [
        tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
    ]

    tensorflow_pred = tf_policy_network(test_input)[0]
    # Build the model by calling it with example data
    print(f"Tensorflow prediction: {tensorflow_pred}")

    tf_policy_network.output_names = ["continuous_actions"]

    # opset 11 matches isaac lab.
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network, input_signature=spec, opset=11, output_path=output_path
    )

    # Compare the exported policy against the original JAX parameterization.
    # This catches normalization, layer ordering and action-splitting mistakes
    # before an ONNX file can reach the robot.
    reference = (test_input[0] - np.asarray(mean)) / np.asarray(std)
    for layer_name in dense_layers:
        layer_params = policy_params[layer_name]
        reference = (
            reference @ np.asarray(layer_params["kernel"])
            + np.asarray(layer_params["bias"])
        )
        if layer_name != dense_layers[-1]:
            sigmoid = np.empty_like(reference)
            positive = reference >= 0
            sigmoid[positive] = 1.0 / (1.0 + np.exp(-reference[positive]))
            exp_value = np.exp(reference[~positive])
            sigmoid[~positive] = exp_value / (1.0 + exp_value)
            reference = reference * sigmoid
    reference = np.tanh(np.split(reference, 2, axis=-1)[0])

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_prediction = session.run(None, {"obs": test_input[0]})[0]
    np.testing.assert_allclose(onnx_prediction, reference, rtol=1e-5, atol=1e-5)
    print("JAX parameter / ONNX output check passed")

    return
