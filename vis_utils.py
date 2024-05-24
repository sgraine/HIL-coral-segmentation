from PIL import Image, ImageDraw
import numpy as np
import os, copy
import matplotlib.pyplot as plt
import torch
from matplotlib import cm

def bound_coordinates(point, image_size):
    x, y = point
    max_x, max_y = image_size

    # Ensure x is within bounds
    x = max(0, min(x, max_x - 1))
    
    # Ensure y is within bounds
    y = max(0, min(y, max_y - 1))

    return x, y

def save_labels(image_name, image_path, save_path, sparse_labels, tag=None, image_size=448):

    file = os.path.join(image_path, image_name)

    with open(file, "rb") as f:
        img = Image.open(f).resize((image_size,image_size))
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    r = 5

    colors = [[167, 18, 159], [180, 27, 92], [104, 139, 233], [49, 198, 135], [98, 207, 26], [118, 208, 133], [158, 118, 90], [12, 72, 166], [69, 79, 238], [81, 195, 49],[221, 236, 52], [160, 200, 222],[255, 63, 216], [16, 94, 7], [226, 47, 64], [183, 108, 5], 
            [55, 252, 193], [147, 154, 196], [233, 78, 165], [108, 25, 95], [184, 221, 46], [54, 205, 145], [14, 101, 210], [199, 232, 230], [66, 10, 103], [161, 228, 59], [108, 2, 104], [13, 49, 127], [186, 99, 38], [97, 140, 246], [44, 114, 202], [36, 31, 118], [146, 77, 143],
            [188, 100, 14],[131, 69, 63]]

    for i in range(image_size):
        for j in range(image_size):
            index = sparse_labels[0][0][i,j].item()
            if index > 0:
                leftUpPoint = (j-r,i-r)
                leftUpPoint = bound_coordinates(leftUpPoint, (image_size,image_size))
                rightDownPoint = (j+r,i+r)
                rightDownPoint = bound_coordinates(rightDownPoint, (image_size,image_size))
                twoPointList = [leftUpPoint, rightDownPoint]

                dot_color = copy.deepcopy(colors[index-1][::-1])
                dot_color_tuple = tuple(dot_color)

                draw.ellipse(twoPointList, fill=dot_color_tuple, outline="black")

    if tag is not None:
        img.save(os.path.join(save_path, image_name[:-4]+"_"+tag+"_label.png"))
    else:
        img.save(os.path.join(save_path, image_name[:-4]+"_label.png"))

def save_output_mask(similarity_mask, sparse_labels, image_name, save_path, image_size=448):
    # Draw the corresponding mask
    img = Image.fromarray(similarity_mask)
    img = img.convert('P')

    # colour_palette is a num_classes x 3 numpy array - where the 3 columns are the RGB values for each class
    # UCSD Mosaics
    colors = [[167, 18, 159], [180, 27, 92], [104, 139, 233], [49, 198, 135], [98, 207, 26], [118, 208, 133], [158, 118, 90], [12, 72, 166], [69, 79, 238], [81, 195, 49],[221, 236, 52], [160, 200, 222],[255, 63, 216], [16, 94, 7], [226, 47, 64], [183, 108, 5], 
        [55, 252, 193], [147, 154, 196], [233, 78, 165], [108, 25, 95], [184, 221, 46], [54, 205, 145], [14, 101, 210], [199, 232, 230], [66, 10, 103], [161, 228, 59], [108, 2, 104], [13, 49, 127], [186, 99, 38], [97, 140, 246], [44, 114, 202], [36, 31, 118], [146, 77, 143],
        [188, 100, 14],[131, 69, 63]]

    bgr=np.array(colors)
    colour_palette = bgr[:,::-1]

    img.putpalette(colour_palette.astype(np.uint8))

    if sparse_labels is not None:
        draw = ImageDraw.Draw(img)
        r = 5

        print("Overlaying labeled points on image...")
        for i in range(image_size):
            for j in range(image_size):
                index = sparse_labels[0][0][i,j].item()
                if index > 0:
                    leftUpPoint = (j-r,i-r)
                    leftUpPoint = bound_coordinates(leftUpPoint, (image_size,image_size))
                    rightDownPoint = (j+r,i+r)
                    rightDownPoint = bound_coordinates(rightDownPoint, (image_size,image_size))
                    twoPointList = [leftUpPoint, rightDownPoint]

                    dot_color = copy.deepcopy(colors[index-1][::-1])
                    dot_color_tuple = tuple(dot_color)

                    draw.ellipse(twoPointList, fill=dot_color_tuple, outline="black")

    print("Saving image...")
    img.save(os.path.join(save_path, image_name+".png"))
    img.close()

def save_probability_mask(probabilities, image_name, save_path, tag=None):

    if torch.is_tensor(probabilities):
        mask_np = probabilities.cpu().numpy()  # shape = [image_size, image_size]
    else:
        mask_np = probabilities

    fig = plt.figure(figsize=(15,10),facecolor='w')
    im_plt = plt.imshow(mask_np, cmap=cm.get_cmap('RdYlGn'), vmin=0.0, vmax=1.0)
    fig.suptitle("Probability of Pixel Selection")
    fig.colorbar(im_plt)
    if tag is not None:
        plt.savefig(os.path.join(save_path, image_name+"_"+tag+".png"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_path,image_name+".png"), bbox_inches='tight')
    plt.clf()
    plt.close()
