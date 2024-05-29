import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter


# Define image operation functions
def white_balance(x, alpha):
    return x * (1 + alpha) if alpha >= 0 else x * (1 / (1 - alpha))


def adjust_brightness(x, beta):
    return np.clip(x + beta, 0, 1)


def adjust_contrast(x, gamma):
    return np.clip((x - 0.5) * gamma + 0.5, 0, 1)


def adjust_saturation(img, delta):
    hsv = rgb2hsv(img)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * delta, 0, 1)
    return hsv2rgb(hsv)


def gamma_correction(x, lambda_gamma):
    return np.clip(x**lambda_gamma, 0, 1)


def linear_interpolate(x, p, q):
    if x <= p[0]:
        return 0 + (q[0] - 0) * (x - 0) / (p[0] - 0)
    elif x <= p[1]:
        return q[0] + (q[1] - q[0]) * (x - p[0]) / (p[1] - p[0])
    elif x <= p[2]:
        return q[1] + (q[2] - q[1]) * (x - p[1]) / (p[2] - p[1])
    else:
        return q[2] + (1 - q[2]) * (x - p[2]) / (1 - p[2])


def apply_curve(img, q_r, q_g, q_b):
    result = np.zeros_like(img)
    for i, q in enumerate([q_r, q_g, q_b]):  # Apply separately for R, G, and B channels
        result[:, :, i] = np.vectorize(
            lambda x: linear_interpolate(x, [0.25, 0.5, 0.75], q)
        )(img[:, :, i])
    return result


def chromatic_aberration(img, alpha_ch, alpha_cw):
    shifted_img = np.copy(img)
    for channel in [0, 2]:  # Assuming R and B channels
        shifted_img[:, :, channel] = np.roll(
            shifted_img[:, :, channel], shift=(alpha_ch, alpha_cw), axis=(0, 1)
        )
    return shifted_img


def apply_blur(img, sigma):
    return gaussian_filter(img, sigma=sigma)


def add_noise(x, alpha_a, alpha_b):
    noise = np.random.normal(0, 1, x.shape)
    return np.clip(x + np.sqrt(alpha_a * x + alpha_b) * noise, 0, 1)


def apply_image_operations(image, parameters):
    # Function to check if a parameter is valid
    def is_valid(param):
        if isinstance(param, (list, np.ndarray)):
            return all(p >= 0 for p in param)
        else:
            return param >= 0

    # Unpack parameters
    alpha_wb = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]
    delta = parameters[3]
    lambda_gamma = parameters[4]
    q = parameters[5:14]
    alpha_ch = parameters[14]
    alpha_cw = parameters[15]
    sigma = parameters[16]
    alpha_a = parameters[17]
    alpha_b = parameters[18]

    # Apply white balance adjustment if valid
    if is_valid(alpha_wb):
        image = white_balance(image, alpha_wb)

    # Apply brightness adjustment if valid
    if is_valid(beta):
        image = adjust_brightness(image, beta)

    # Apply contrast adjustment if valid
    if is_valid(gamma):
        image = adjust_contrast(image, gamma)

    # Apply saturation adjustment if valid
    if is_valid(delta):
        image = adjust_saturation(image, delta)

    # Apply gamma correction if valid
    if is_valid(lambda_gamma):
        image = gamma_correction(image, lambda_gamma)

    # Apply channel curve adjustment if valid
    if is_valid(q):
        q_r = q[:3]
        q_g = q[3:6]
        q_b = q[6:]
        image = apply_curve(image, q_r, q_g, q_b)

    # Apply chromatic aberration if valid
    if is_valid(alpha_ch) and is_valid(alpha_cw):
        image = chromatic_aberration(image, int(alpha_ch), int(alpha_cw))

    # Apply blurring if valid
    if is_valid(sigma):
        image = apply_blur(image, sigma)

    # Apply noise if valid
    if is_valid(alpha_a) or is_valid(alpha_b):
        alpha_a = alpha_a if is_valid(alpha_a) else 0
        alpha_b = alpha_b if is_valid(alpha_b) else 0
        image = add_noise(image, alpha_a, alpha_b)

    return image


def normalize_parameters(params):
    normalized = []
    ranges = {
        "alpha_wb": (-1, 1),
        "beta": (-0.5, 0.5),
        "gamma": (0.5, 2),
        "delta": (0.5, 1.5),
        "lambda_gamma": (0.5, 2.5),
        "q": (0, 1),
        "alpha_ch": (-5, 5),
        "alpha_cw": (-5, 5),
        "sigma": (0, 3),
        "alpha_a": (0, 0.1),
        "alpha_b": (0, 0.1),
    }

    idx = 0
    for key, (low, high) in ranges.items():
        if key == "q":
            for i in range(9):
                if params[idx] >= 0:
                    normalized.append((params[idx] - low) / (high - low))
                else:
                    normalized.append(-1)
                idx += 1
        else:
            if params[idx] >= 0:
                normalized.append((params[idx] - low) / (high - low))
            else:
                normalized.append(-1)
            idx += 1

    return normalized


def denormalize_parameters(norm_params):
    denormalized = []
    ranges = {
        "alpha_wb": (-1, 1),
        "beta": (-0.5, 0.5),
        "gamma": (0.5, 2),
        "delta": (0.5, 1.5),
        "lambda_gamma": (0.5, 2.5),
        "q": (0, 1),
        "alpha_ch": (-5, 5),
        "alpha_cw": (-5, 5),
        "sigma": (0, 3),
        "alpha_a": (0, 0.1),
        "alpha_b": (0, 0.1),
    }

    idx = 0
    for key, (low, high) in ranges.items():
        if key == "q":
            for i in range(9):
                if norm_params[idx] >= 0:
                    denormalized.append(norm_params[idx] * (high - low) + low)
                else:
                    denormalized.append(-1)
                idx += 1
        else:
            if norm_params[idx] >= 0:
                denormalized.append(norm_params[idx] * (high - low) + low)
            else:
                denormalized.append(-1)
            idx += 1

    return denormalized


def generate_random_parameters():
    parameters = {
        "alpha_wb": np.random.uniform(-1, 1),  # White balance
        "beta": np.random.uniform(-0.5, 0.5),  # Brightness
        "gamma": np.random.uniform(0.5, 2),  # Contrast
        "delta": np.random.uniform(0.5, 1.5),  # Saturation
        "lambda_gamma": np.random.uniform(0.5, 2.5),  # Gamma correction
        "q": np.random.uniform(0, 1, size=9),  # Channel curve output points for R, G, B
        "alpha_ch": np.random.randint(-5, 6),  # Chromatic aberration horizontal shift
        "alpha_cw": np.random.randint(-5, 6),  # Chromatic aberration vertical shift
        "sigma": np.random.uniform(0, 3),  # Blurring
        "alpha_a": np.random.uniform(0, 0.1),  # Noise Poisson component
        "alpha_b": np.random.uniform(0, 0.1),  # Noise Gaussian component
    }

    # Randomly select a number of parameters to use
    n = np.random.randint(1, 4)
    selected_keys = np.random.choice(list(parameters.keys()), n, replace=False)

    # Create a parameter list with default large values (e.g., float('inf'))
    param_list = [float(-1)] * sum(
        len(v) if isinstance(v, np.ndarray) else 1 for v in parameters.values()
    )
    param_order = list(parameters.keys())

    index = 0
    for key in param_order:
        if key in selected_keys:
            if isinstance(parameters[key], np.ndarray):
                for i in range(len(parameters[key])):
                    param_list[index] = parameters[key][i]
                    index += 1
            else:
                param_list[index] = parameters[key]
                index += 1
        else:
            if isinstance(parameters[key], np.ndarray):
                index += len(parameters[key])
            else:
                index += 1

    return param_list
