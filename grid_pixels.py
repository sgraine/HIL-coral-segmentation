import numpy as np

def select_grid_pixels(image_size, num_pixels):
    # Get image dimensions
    image_height, image_width = image_size

    # Calculate the number of rows and columns based on the desired number of pixels
    rows = int(np.ceil(np.sqrt(num_pixels * image_height / image_width)))
    cols = int(np.ceil(num_pixels / rows))

    # Adjust rows and cols to ensure the total number of selected pixels is <= num_pixels
    while rows * cols > num_pixels:
        if rows > cols:
            rows -= 1
        else:
            cols -= 1

    # Calculate the step size for equally spaced grid
    row_step = int(np.floor(image_height / rows))
    col_step = int(np.floor(image_width / cols))

    # Calculate the offset to center the grid
    row_offset = max(0, int(row_step / 2))
    col_offset = max(0, int(col_step / 2))

    # Create a binary mask with zeros everywhere
    mask = np.zeros((image_height, image_width), dtype=np.int64)

    # Select pixels in an equally spaced grid with the center offset
    for i in range(rows):
        for j in range(cols):
            # Calculate indices based on the step size, offset, and padding
            row_index = i * row_step + row_offset
            col_index = j * col_step + col_offset

            # Ensure indices do not exceed image dimensions
            row_index = min(row_index, image_height - 1)
            col_index = min(col_index, image_width - 1)

            # Set the selected pixel to 1 in the mask
            mask[row_index, col_index] = 1

    return mask


# Example usage:
# rectangular_image_size = (10,30)  # Example rectangular image size
# num_pixels_to_select = 10

# result_mask = select_grid_pixels(rectangular_image_size, num_pixels_to_select)
# print(result_mask)
