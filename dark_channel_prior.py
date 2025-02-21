#!/usr/bin/env/python3


import cv2
import numpy as np


def get_dark_channel(image, window_size):
    # Returns the minimum value of the color channels of the image
    min_channel = np.min(image, axis=2)

    # This is a "morphological transformation" which is essentially eroding away at
    # objects in the foreground.  This is what creates the bright white artifacts 
    # around the telephone pole.  Each of the color channels are processed with erosion.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    get_dark_channel = cv2.erode(min_channel, kernel)

    return get_dark_channel


def get_atmospheric_light(image, get_dark_channel):
    # Returns the dimensions of the image.  In the case of the traffic light,
    # it is a 1500 x 1200 pixel image.
    h, w = get_dark_channel.shape

    # Total pixel count of the image.
    num_pixels = h * w

    # The floor fucntion returns a value less than or equal to a given value.
    # The computation looks like this:
    # (1,800,000 * 0.1) = 1800.0
    # What is returned is an integer thanks to the type cast to int.
    num_brightest = int(max(np.floor(num_pixels * 0.001), 1))

    # Gets the 2D array from the `get_dark_channel` method which is a grayscale
    # of the original image.  That 2D array is then flattened into a 1D array.
    # This makes it easier to sort out the darkest and brightest pixels across
    # all of the color channels in an image
    dark_vec = get_dark_channel.reshape(num_pixels)

    # `image.reshape` flattens the 2D grayscale into a 1D array with `num_pixels`
    # of elements in the array.
    image_vec = image.reshape(num_pixels, 3)

    # Here is where the atmospheric light is estimated.  The `np.argpartition`
    # method will take the flattened dark channel and the number of brightest
    # pixels from the image and find the indices of those pixels so that the
    # global atmospheric light can be estimated.
    indices = np.argpartition(-dark_vec, num_brightest)[:num_brightest]
    atmospheric_light = np.mean(image_vec[indices], axis=0)

    return atmospheric_light


def get_transmission_estimate(image, atmospheric_light, window_size, omega=0.95):
    # The image is "normalized" meaning that the RGB image has all of its channels
    # divided by the vector `atmospheric_light`.  This results in the haze of the
    # image being uniform across the whole image.  The `1e-6` is a constant to prevent
    # dividing by zero.  The dehazing model as described in our notes is an "ill-posed"
    # function meaning there is a potential for division by zero since the transmission
    # function can still be zero.
    normalized_image = image / (atmospheric_light + 1e-6)

    # The transmission map is recovered from the following.  Omega here is a constant
    # representing a value between 0 and 1.  Omega represents the haze that is present
    # in the image.  Therefore, the `1 - omega` represents the haze that is NOT removed
    # from the image.  This is done to preserve its natural look.
    transmission = 1 - omega * get_dark_channel(normalized_image, window_size)

    return np.clip(transmission, 0.1, 1)


def get_recovered_image(image, transmission, atmospheric_light, t0=0.1):
    # Since we are constricted by an "ill-posed" equation, we have to ensure that we
    # do NOT divide by zero hence the use of `np.maximum` here.  The `t0` variable be
    # assigned a value of `0.1` is a failsafe of sorts that prevents that scenario.
    transmission = np.maximum(transmission, t0)

    # Here we can see the resemblence to the mathematical model for a dehazed image with
    # the recovery of J(x) (the unhazed image).  The atmospheric light is removed from the
    # image and then divided by the transmission map that was recovered earlier to remove
    # the haziness.  A is then added back to restore some of the natural light that was in
    # the image.
    J = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light

    # The `np.clip` method essentially sets bounds for the pixels.  If there is anything
    # above 255, its value will be 255 and vice versa for anything bellow zero.
    J = np.clip(J * 255, 0, 255)

    return J.astype(np.uint8)


def dehaze(image, window_size=15, omega=0.95, t0=0.1):
    image = image.astype(np.float64) / 255

    dark = get_dark_channel(image, window_size)
    atmospheric_light = get_atmospheric_light(image, dark)
    raw_transmission = get_transmission_estimate(image, atmospheric_light, window_size, omega)

    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255

    dehazed_image = get_recovered_image(image, raw_transmission, atmospheric_light, t0)
    return dehazed_image


if __name__ == "__main__":
    image = cv2.imread("hazy_image.jpg")
    dehazed = dehaze(image)
    cv2.imwrite("dehazed_image.jpg", dehazed)
    cv2.imshow("Dehazed Image", dehazed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

