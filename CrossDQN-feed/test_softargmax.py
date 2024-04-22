import numpy as np


def soft_argmax(x, beta=1.0):
    """Soft argmax function.

    Args:
        x (numpy.ndarray): Input array.
        beta (float): Softness parameter (default: 1.0).

    Returns:
        numpy.ndarray: Soft argmax result.
    """
    e_x = np.exp(beta * x)
    weights = e_x / np.sum(e_x)
    indices = np.arange(len(x))
    return np.sum(indices * weights)


if __name__ == '__main__':

    # Example usage
    input_array = np.array([0.1, 0.3, 0.5, 0.4])
    result = soft_argmax(input_array)
    print("Soft argmax result:", round(result))
    print("max is : ", input_array[int(round(result))])
