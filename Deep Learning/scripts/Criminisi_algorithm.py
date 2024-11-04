import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace


class CriminisiAlgorithm:
    def __init__(self, img_name, image, mask, patch_size=9, plot_progress=False):
        """
        Initializes the CriminisiAlgorithm object with an image, mask, patch size, and an option to display progress.
        
        Parameters:
        - img_name: Name of the image (str)
        - image: Input image as a NumPy array
        - mask: Mask indicating regions to inpaint
        - patch_size: Size of the patch (int)
        - plot_progress: Flag to display progress (bool)
        """
        self.image = image.astype("uint8")
        self.mask = mask.round().astype("uint8")
        self.img_name = img_name
        self.patch_size = patch_size
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.priority = None

    def inpaint(self):
        """
        Starts the image inpainting process, executing the main loop to fill the mask using the Criminisi algorithm.
        
        Returns:
        - The transformed image with inpainted regions.
        """

        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()

            target_pixel = self._find_highest_priority_pixel()
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            print("Time to find best: %f seconds" % (time.time() - find_start_time))

            self._update_image(target_pixel, source_patch)

            keep_going = not self._finished()
        with open(
            f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks/scripts/time.txt",
            "a",
        ) as f:
            f.write(
                f"Took {(time.time() - start_time)} seconds to complete, img :{self.img_name} \n"
            )
        return self.working_image

    def _validate_inputs(self):
        """
        Checks if the image and mask have the same dimensions; raises an error if they don't.
        """
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError("mask and image must be of the same size")

    def _plot_image(self):
        """
        Displays the image with currently inpainted pixels and marks the current boundary of the region to be filled.
        """
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)

    def _initialize_attributes(self):
        """
        Initializes the necessary attributes for the algorithm, including masks, confidence values, data, and working masks.
        """
        height, width = self.image.shape[:2]

        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([height, width])

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

    def _find_front(self):
        """
        Identifies pixels at the edge of the mask using the Laplace operator.
        
        Pixels on the edge are identified as the "front," marking areas to be inpainted.
        """
        self.front = (laplace(self.working_mask) > 0).astype("uint8")

    def _update_priority(self):
         """
        Calculates new confidence values for all pixels at the edge of the mask by averaging the confidence values of neighboring pixels.
        """
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        """
        Calculates the data gradient for all pixels at the edge of the mask based on normal vectors and data gradients.
        """
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(
                sum(self._patch_data(self.confidence, patch))
            ) / self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):
        """
        Calculates the data gradient for all pixels at the edge of the mask based on normal vectors and data gradients.
        """
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal * gradient
        self.data = (
            np.sqrt(normal_gradient[:, :, 0] ** 2 + normal_gradient[:, :, 1] ** 2)
            + 0.001
        )  # To be sure to have a greater than 0 data

    def _calc_normal_matrix(self):
        """
        Calculates the normal vector matrix from the mask using x and y convolution kernels.
        
        Returns:
        - unit_normal: Matrix of unit normal vectors.
        """
        x_kernel = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
        y_kernel = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = (
            np.sqrt(y_normal**2 + x_normal**2)
            .reshape(height, width, 1)
            .repeat(2, axis=2)
        )
        norm[norm == 0] = 1

        unit_normal = normal / norm
        return unit_normal

    def _calc_gradient_matrix(self):
        """
        Calculates the brightness gradient of the image for pixels at the edge of the mask.
        
        Returns:
        - max_gradient: Brightness gradient for each pixel at the edge.
        """
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(), patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        """
        Finds the pixel with the highest priority in the priority matrix.
        
        Returns:
        - The coordinates of the highest priority pixel.
        """
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        """
        Finds the best matching source patch for a given target pixel.
        
        Parameters:
        - target_pixel: The target pixel for which a source patch is sought
        
        Returns:
        - Coordinates of the best matching source patch.
        """
        target_patch = self._get_patch(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image)

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [[y, y + patch_height - 1], [x, x + patch_width - 1]]
                if self._patch_data(self.working_mask, source_patch).sum() != 0:
                    continue

                difference = self._calc_patch_difference(
                    lab_image, target_patch, source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        """
        Updates the image with data from the source patch, inpainting the area around the target pixel.
        
        Parameters:
        - target_pixel: The pixel to inpaint
        - source_patch: The patch used as a reference for inpainting
        """
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(self.working_mask, target_patch) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)

        self._copy_to_patch(self.working_image, target_patch, new_data)
        self._copy_to_patch(self.working_mask, target_patch, 0)

    def _get_patch(self, point):
        """
        Retrieves a patch around a given pixel, bounded by the patch size.
        
        Parameters:
        - point: The central pixel of the patch
        
        Returns:
        - The coordinates defining the patch area.
        """
        half_patch_size = (self.patch_size - 1) // 2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height - 1),
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width - 1),
            ],
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        """
        Calculates the difference between the target and source patches.
        
        Parameters:
        - image: The input image
        - target_patch: The patch to inpaint
        - source_patch: The candidate source patch
        
        Returns:
        - A difference score for the patch pair.
        """
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(image, target_patch) * rgb_mask
        source_data = self._patch_data(image, source_patch) * rgb_mask
        squared_distance = ((target_data - source_data) ** 2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0]) ** 2
            + (target_patch[1][0] - source_patch[1][0]) ** 2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance

    def _finished(self):
        """
        Checks if the inpainting process is complete by verifying that there are no remaining masked pixels.
        
        Returns:
        - A boolean indicating whether the inpainting is complete.
        """
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        print("%d of %d completed" % (total - remaining, total))
        return remaining == 0

    @staticmethod
    def _patch_area(patch):
        """
        Calculates the area of a given patch.
        
        Parameters:
        - patch: The patch for which to calculate the area
        
        Returns:
        - The area of the patch.
        """
        return (1 + patch[0][1] - patch[0][0]) * (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        """
        Determines the shape (height and width) of a given patch.
        
        Parameters:
        - patch: The patch for which to determine the shape
        
        Returns:
        - The height and width of the patch.
        """
        return (1 + patch[0][1] - patch[0][0]), (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        """
        Extracts the data from a given patch within a source matrix.
        
        Parameters:
        - source: The source matrix (e.g., image or mask)
        - patch: The patch coordinates
        
        Returns:
        - The data contained within the specified patch.
        """
        return source[patch[0][0] : patch[0][1] + 1, patch[1][0] : patch[1][1] + 1]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        """
        Copies the given data into the specified patch within a destination matrix.
        
        Parameters:
        - dest: The destination matrix
        - dest_patch: The patch coordinates in the destination
        - data: The data to copy into the patch
        """
        dest[
            dest_patch[0][0] : dest_patch[0][1] + 1,
            dest_patch[1][0] : dest_patch[1][1] + 1,
        ] = data

    @staticmethod
    def _to_rgb(image):
        """
        Converts a single-channel image to a 3-channel RGB image.
        
        Parameters:
        - image: The single-channel image
        
        Returns:
        - The RGB image as a NumPy array.
        """
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
