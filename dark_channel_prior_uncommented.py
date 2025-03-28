#!/usr/bin/env/python3


import cv2
import numpy as np


def get_dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)

    cv2.imshow(f"Dark Channel of {image}", dark_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dark_channel


def get_atmospheric_light(image, get_dark_channel):
    h, w = get_dark_channel.shape

    num_pixels = h * w

    num_brightest = int(max(np.floor(num_pixels * 0.2), 1))

    dark_vec = get_dark_channel.reshape(num_pixels)

    image_vec = image.reshape(num_pixels, 3)

    indices = np.argpartition(-dark_vec, num_brightest)[:num_brightest]
    atmospheric_light = np.mean(image_vec[indices], axis=0)

    return atmospheric_light


def get_transmission_estimate(image, atmospheric_light, window_size, omega=0.95):
    normalized_image = image / (atmospheric_light + 1e-6)

    transmission = 1 - omega * get_dark_channel(normalized_image, window_size)

    return np.clip(transmission, 0.01, 1)


def get_recovered_image(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.maximum(transmission, t0)

    J = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light

    J = np.clip(J * 255, 0, 255)

    return J.astype(np.uint8)


def dehaze(image, window_size=15, omega=0.95, t0=0.1):
    image = image.astype(np.float64) / 255

    dark = get_dark_channel(image, window_size)

    atmospheric_light = get_atmospheric_light(image, dark)

    raw_transmission = get_transmission_estimate(image, atmospheric_light, window_size, omega)

    dehazed_image = get_recovered_image(image, raw_transmission, atmospheric_light, t0)

    return dehazed_image


if __name__ == "__main__":
    hazed_images_array = [
                    "gas_station.png",
                    "highway.jpg",
                    "interstate.png",
                    "parking_garage.jpg",
                    "snowy_highway.jpg",
                    "street_light.jpg",
                    "city.jpg"
                    ]

    for image in hazed_images_array:
        hazed_image = cv2.imread(f"./Images/Hazed/{image}")
        dehazed_image = dehaze(hazed_image)

        cv2.imwrite(f"./Images/Dehazed/{image}", dehazed_image)
        cv2.imshow(f"{image}", dehazed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
