import numpy as np


def tile_image(image_array, tile_size, tile_spacing=(0,0)):
    """
    Render an array of image as tiles in one image
    :param image_array: Data structure for the image array
    :param tile_size: Size of the image tile
    :param tile_spacing: tuple that defines the row spacing and column spacing
    :return: the rendered image
    """
    # Make sure that the input parameters are of correct size
    assert len(image_array.shape) == 4
    assert len(tile_size) == 2
    assert len(tile_spacing) == 2

    # Get Information on parameters
    shape = image_array.shape
    (num_row, num_column, img_width, img_height) = shape
    (tile_width, tile_height) = tile_size
    (spacing_width, spacing_height) = tile_spacing

    # Allocate array for final image
    rendered_width = tile_width * num_column + spacing_width * (num_column + 1)
    rendered_height = tile_height * num_row + spacing_height * (num_row + 1)
    img_rendered = np.zeros((rendered_height, rendered_width))

    # Put image in tiles
    for i in range(num_row):
        for j in range(num_column):
            start_row = spacing_height * (i+1) + tile_height * i
            end_row = spacing_height * (i+1) + tile_height * (i+1)
            start_col = spacing_width * (j+1) + tile_width * j
            end_col = spacing_width * (j+1) + tile_width * (j+1)
            img_rendered[start_row:end_row, start_col:end_col] = image_array[i][j]

    return img_rendered


