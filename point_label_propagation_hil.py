### THIS IS THE MAIN SCRIPT THAT SHOULD BE RUN FOR POINT LABEL PROPAGATION ###

# Import the necessary packages
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os, argparse
from tqdm import tqdm
import torchmetrics
import faiss

from utils_metrics import calculate_iou, calculate_pixel_accuracy, calculate_global_accuracy
from vis_utils import save_labels, save_output_mask, save_probability_mask

from scipy.ndimage import distance_transform_edt
import DenoisingViT

# Extract our features from the denoised DINOv2 model
def extract_features(vit, model, filename, image_path, image_size, resize=None,vit_stride=14,):

    normalizer = vit.transformation.transforms[-1]
    if resize is not None:
        transform_func = T.Compose(
            [T.ToTensor(), T.Resize(resize), normalizer]
        )
    else:
        transform_func = T.Compose([T.ToTensor(), normalizer])

    file = os.path.join(image_path, filename)

    upsample = torch.nn.UpsamplingBilinear2d((image_size, image_size))  # shape = [1, 1024, image_size, image_size]

    with open(file, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    img = transform_func(img)

    # make the image divisible by the patch size
    w, h = (
        img.shape[1] - img.shape[1] % vit_stride,
        img.shape[2] - img.shape[2] % vit_stride,
    )
    img = img[:, :w, :h].unsqueeze(0)

    # Get raw and denoised outputs from model
    output_dict = model(img.to(device), return_dict=True)

    # Upsample to obtain a feature for each pixel
    raw_vit_feats = output_dict["raw_vit_feats"]
    raw_vit_feats = torch.permute(raw_vit_feats, (0,3,1,2))
    raw_vit_feats_up = upsample(raw_vit_feats)

    denoised_feats = output_dict["pred_denoised_feats"]
    denoised_feats = torch.permute(denoised_feats, (0,3,1,2))
    denoised_feats_up = upsample(denoised_feats)

    return raw_vit_feats_up, denoised_feats_up

# Load in the denoised DINOv2 checkpoint
def load_model(load_denoiser_from, vit_type, vit_stride, noise_map_height, noise_map_width, enable_pe=True):
    # build model
    vit = DenoisingViT.ViTWrapper(
        model_type=vit_type,
        stride=vit_stride,
    )
    vit = vit.to(device)
    model = DenoisingViT.Denoiser(
        noise_map_height=noise_map_height,
        noise_map_width=noise_map_width,
        feature_dim=vit.n_output_dims,
        vit=vit,
        enable_pe=enable_pe,
    ).to(device)
    if load_denoiser_from is not None:
        freevit_model_ckpt = torch.load(load_denoiser_from)["denoiser"]
        msg = model.load_state_dict(freevit_model_ckpt, strict=False)
    for k in model.state_dict().keys():
        if k in freevit_model_ckpt:
            print(k, "loaded")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    
    return model, vit

# Identify the instances in our ground truth mask - this is for simulating the human expert
def label_instances(class_mask):
    instances = np.zeros_like(class_mask)
    instance_label = 1

    for i in range(class_mask.shape[0]):
        for j in range(class_mask.shape[1]):
            if class_mask[i, j] == 1 and instances[i, j] == 0:
                stack = [(i, j)]
                while stack:
                    (x, y) = stack.pop()
                    if class_mask[x, y] == 1 and instances[x, y] == 0:
                        instances[x, y] = instance_label

                        if x > 0:
                            stack.append((x - 1, y))
                        if x < class_mask.shape[0] - 1:
                            stack.append((x + 1, y))
                        if y > 0:
                            stack.append((x, y - 1))
                        if y < class_mask.shape[1] - 1:
                            stack.append((x, y + 1))

                instance_label += 1
    return instances, instance_label - 1

# Creates a distance map for the previously labeled pixels
def create_distance_mask(label_array, sigma):
    # Create a binary mask where labeled pixels are True
    binary_mask = label_array == 0
    
    # Compute the Euclidean distance transform
    distance_transform_result = distance_transform_edt(binary_mask)

    # Apply Gaussian smoothing to the distance transform
    distance_mask = 1 - np.exp(- np.power(distance_transform_result,2) / (2 * np.power(sigma, 2)))
    
    return distance_mask

# Adds a new sparse point label to our current sparse GT mask
def add_to_sparse_mask(GT_mask_torch, sparse_labels, smart_indices):
    sx = GT_mask_torch.shape[0]

    sparse_labels_new = torch.squeeze(sparse_labels.clone())

    for index in smart_indices:
        x = index % sx
        y = torch.div(index, sx, rounding_mode='trunc')

        sparse_labels_new[y,x] = GT_mask_torch[y,x]+1

    return sparse_labels_new.unsqueeze(0).unsqueeze(0)

# This function takes our sparse labels and current cosine similarities and determines where the new label should
# be placed; it then adds the new label and returns the new clustering result
def human_in_the_loop_labeling(GT_mask_torch, similarities_torch, sparse_labels, sigma, cos_weight, k, image_name):

    distances = torch.squeeze(torch.from_numpy(create_distance_mask(sparse_labels.cpu().numpy(), sigma)).to(device))
    distances_ft = torch.flatten(distances)

    # Uncomment the below to save the probability masks
    # if vis_path is not None:
    #     save_probability_mask(distances, image_name[:-4]+"_"+str(p), vis_path, tag="dist")

    sim_sort, _ = torch.sort(similarities_torch, descending=True, dim=1)
    sim_sort = sim_sort[:,0].to(device)

    probs = 1 - sim_sort
    probs[probs < 0] = 0

    # Uncomment the below to save the probability masks
    # if vis_path is not None:
    #     probs_reshaped = torch.reshape(probs, (image_size, image_size)) 
    #     save_probability_mask(probs_reshaped, image_name[:-4]+"_"+str(p), vis_path, tag="sim")

    dist_weight = 1

    merge_probs = (dist_weight*distances_ft + probs*cos_weight)/ (dist_weight + cos_weight)
    merge_probs_reshaped = torch.reshape(merge_probs, (image_size, image_size))

    merge_probs_filt = torch.where(torch.squeeze(sparse_labels) < 1, merge_probs_reshaped, 0)
    merge_probs_filt_ft = torch.flatten(merge_probs_filt)

    # Uncomment the below to save the probability masks
    # if vis_path is not None:
    #     save_probability_mask(merge_probs_filt, image_name[:-4]+"_"+str(p), vis_path, tag="merge") 

    _, smart_index = torch.topk(merge_probs_filt_ft, 1, sorted=False)  

    new_sparse_labels = add_to_sparse_mask(GT_mask_torch, sparse_labels, smart_index).to(device)    # shape = [B, 1, H, W]    
    new_sparse_labels_ft = torch.permute(torch.flatten(new_sparse_labels.clone()[0], 1,2), (1,0))  # shape = [image_size x image_size, 1]

    # add all labels to all sparse labels
    new_mask = (new_sparse_labels_ft > 0)
    new_template = torch.where(new_mask == True, features_ft, 0)
    new_labeled_features = new_template[abs(new_template).sum(dim=1) != 0]  # shape = [num_labelled_points, 1024]

    # Obtain the class of our labeled features
    new_labeled_points = torch.squeeze(new_sparse_labels_ft[abs(new_sparse_labels_ft).sum(dim=1) != 0], 1)  # shape = [num_labelled_points]

    del new_template, new_sparse_labels_ft
    torch.cuda.empty_cache()

    new_x = np.ascontiguousarray(new_labeled_features.cpu().numpy()).astype(np.float32)  # labeled_features shape = [num_labelled_points, 1024]

    # if we do L2 normalization and then use the metric inner product, the result is the cosine similarity
    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    index_2 = faiss.index_factory(768, "Flat", faiss.METRIC_INNER_PRODUCT)
    gpu_index_2 = faiss.index_cpu_to_gpu(res, 0, index_2)
    gpu_index_2.ntotal
    faiss.normalize_L2(new_x)
    gpu_index_2.add(new_x)
    faiss.normalize_L2(q)
    new_similarities, new_sorted_indices = gpu_index_2.search(q, k) 
    gpu_index_2.reset()

    new_labeled_points = new_labeled_points.cpu().numpy()
    new_sorted_classes = new_labeled_points[new_sorted_indices] - 1   # shape = [image_size x image_size, k]

    new_similarities_torch = torch.from_numpy(new_similarities)        
    
    if k > 1:
        new_sorted_classes = torch.from_numpy(new_sorted_classes)
        new_nearest = torch.mode(new_sorted_classes)[0].numpy()
    else:
        new_nearest = new_sorted_classes[:,0]  # shape = [image_size x image_size,]
        new_sorted_classes = torch.from_numpy(new_sorted_classes)

    # Reshape class_nums back to the image shape
    new_similarity_mask = np.reshape(new_nearest, (image_size, image_size))   # shape = [image_size, image_size]
    new_similarity_mask_torch = torch.from_numpy(new_similarity_mask)

    return new_similarities_torch, new_similarity_mask_torch, new_nearest, new_sparse_labels, new_sorted_classes

# This function mimics a human expert by selecting points in the middle of the largest instances
def select_pixels_by_size(segmentation_mask, num_pixels=10):
    segmentation_mask = segmentation_mask.cpu().numpy()
    unique_classes = np.unique(segmentation_mask)
    sparse_mask = np.zeros_like(segmentation_mask)

    instance_pixels = []
    instance_index =0
    for cls in unique_classes:
        class_mask = (segmentation_mask == cls).astype(int)
        if np.sum(class_mask) == 0:
            continue

        # Label instances within the class
        instances, num_instances = label_instances(class_mask)

        # Calculate areas of instances and store them
        for instance_label in range(1, num_instances + 1):
            instance_mask = (instances == instance_label)
            area = np.sum(instance_mask)
            non_zero_indices = np.nonzero(instance_mask)
            middle_index = tuple(np.array([int(np.median(indices)) for indices in non_zero_indices]))
            instance_pixels.append((instance_index, cls, middle_index, area))
            instance_index += 1
            
    selected_pixels = select_pixels(instance_pixels, num_pixels)

    for pixel_tuple in selected_pixels:
        pixel = pixel_tuple[2]
        cls = pixel_tuple[1]
        sparse_mask[pixel[0], pixel[1]] = cls + 1

    return torch.from_numpy(sparse_mask)

# This function selects the pixel from the largest instances
def select_pixels(tuples_list, num_pixels):
    # Create a dictionary to store the largest area for each class
    class_max_area = {}

    # Iterate through the tuples list to find the largest area for each class
    for idx, cls, _, area in tuples_list:
        if cls not in class_max_area or area > class_max_area[cls]:
            class_max_area[cls] = area

    # Sort the tuples list based on area in descending order
    tuples_list = sorted(tuples_list, key=lambda x: x[3], reverse=True)

    selected_pixels = []
    selected_classes = set()
    class_counts = {}

    # Iterate through the sorted list to select pixels
    for idx, cls, coords, area in tuples_list:
        # Check if the current class has already been selected
        if cls not in selected_classes:
            # Check if the area of the current pixel is greater than or equal to 300
            if area >= 300:
                selected_pixels.append((idx, cls, coords, area))
                selected_classes.add(cls)
                class_counts[cls] = 1
        # Check if the desired number of pixels has been selected
        elif len(selected_pixels) < num_pixels:
            # Check if the area of the current pixel is greater than or equal to 300
            if area >= 300:
                selected_pixels.append((idx, cls, coords, area))
                class_counts[cls] = class_counts.get(cls, 0) + 1

    # If not enough pixels have been selected, pick from remaining classes
    if len(selected_pixels) < num_pixels:
        remaining_classes = set(class_max_area.keys()) - selected_classes
        remaining_tuples = [(idx, cls, _, area) for idx, cls, _, area in tuples_list if cls in remaining_classes]
        remaining_tuples.sort(key=lambda x: x[3], reverse=True)
        for idx, cls, _, area in remaining_tuples:
            if len(selected_pixels) < num_pixels and area >= 300:
                selected_pixels.append((idx, cls, coords, area))
                class_counts[cls] = class_counts.get(cls, 0) + 1

    # If more pixels were selected than required, remove excess
    if len(selected_pixels) > num_pixels:
        selected_pixels = selected_pixels[:num_pixels]

    return selected_pixels

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Input specifications for generating augmented ground truth from sparse point labels.')

    # Paths - these are required
    parser.add_argument('-i', '-image_path', action='store', type=str, dest='image_path', help='the path to the images', required=True)
    parser.add_argument('-g', '-gt_path', action='store', type=str, dest='gt_path', help='the path to the provided labels', required=True)

    # Optional path if you want to save the masks for training a model to perform semantic segmentation
    parser.add_argument('-p', '--save_path', action='store', type=str, dest='save_path', help='OPTIONAL: the destination of your propagated labels')
    parser.add_argument('-v', '--vis_path', action='store', type=str, dest='vis_path', help='OPTIONAL: the destination of visualisations showing the propagated labels in RGB')

    # Optional parameters
    # Default values correspond to the UCSD Mosaics dataset
    parser.add_argument('-s', '--sigma', action='store', type=int, default=50, dest='sigma', help='sigma value, controls smoothness of the distance mask')
    parser.add_argument('-l', '--lambda', action='store', type=float, default=2.2, dest='cos_weight', help='lambda value, controls weighting of the inverted similarity and distance masks')
    parser.add_argument('-k', '--knn', action='store', type=int, default=1, dest='k', help='k value in KNN')
    parser.add_argument('-n', '--num_total_pixels', action='store', type=int, default=25, dest='num_total_pixels', help='the number of labeled points per image')
    parser.add_argument('-m', '--num_human_pixels', action='store', type=int, default=10, dest='num_human_pixels', help='the number of human-labeled points per image for initializing the approach')
    
    # Save input arguments as variables
    args = parser.parse_args()

    image_path = args.image_path
    gt_path = args.gt_path          
    save_path = args.save_path
    vis_path = args.vis_path

    # Set hyperparameters
    sigma = args.sigma    # sigma = 50
    cos_weight = args.cos_weight  # lambda = 2.2
    k = args.k  # k=1
    num_total_pixels = args.num_total_pixels # e.g. 5, 10, 25 or 300
    num_human_points = args.num_human_pixels # 10

    all_images = [image_name for image_name in os.listdir(image_path)]

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print('device:',device)

    # DINOv2 parameters and models
    stride = 14
    load_denoiser_from = "dinov2_v1.pth"
    # load_denoiser_from = "reg4_v1.pth"
    vit_type = "vit_base_patch14_dinov2.lvd142m"
    # vit_type = "vit_base_patch14_reg4_dinov2.lvd142m"
    noise_map_height = 37
    noise_map_width = 37

    model, vit = load_model(load_denoiser_from, vit_type, stride, noise_map_height, noise_map_width)

    # UCSD Mosaics parameters
    unlabeled = 34
    NUM_CLASSES = 35
    original_image_size = 512
    image_size = 448
    image_height = image_size
    image_width = image_size
    num_feats = 768


    confmat = torchmetrics.ConfusionMatrix(num_classes = NUM_CLASSES, task='multiclass').to(device)

    for image_name in tqdm(all_images):

        GT_pil_img = Image.open(os.path.join(gt_path,image_name)).resize((image_width, image_height), Image.NEAREST)
        GT_mask_np = np.array(GT_pil_img)

        GT_mask_torch = torch.from_numpy(GT_mask_np)  # shape = [image_size,image_size]

        # FOR INSTANCE LABELS:
        if num_total_pixels > num_human_points:
            sparse_labels = torch.unsqueeze(torch.unsqueeze(select_pixels_by_size(GT_mask_torch, num_human_points), 0), 0).to(device) # shape = [B, 1, H, W]
        else:
            sparse_labels = torch.unsqueeze(torch.unsqueeze(select_pixels_by_size(GT_mask_torch, num_total_pixels), 0), 0).to(device) # shape = [B, 1, H, W]
        
        # save_labels(image_name, image_path, SAVE_PATH, sparse_labels, tag="start", image_size=image_size)

        _, features = extract_features(vit, model, image_name, image_path, image_size, (image_size,image_size))  # shape = [1, 1024, image_size, image_size]

        sparse_labels_ft = torch.permute(torch.flatten(sparse_labels[0], 1,2), (1,0))  # shape = [image_size x image_size, 1]
        features_ft = torch.permute(torch.flatten(features[0], 1,2), (1,0))  # shape = [image_size x image_size, 1024]

        template = torch.where(sparse_labels_ft > 0, features_ft, 0)
        labeled_features = template[abs(template).sum(dim=1) != 0]  # shape = [num_labelled_points, 1024]

        # Obtain the class of our labeled features
        labeled_points = torch.squeeze(sparse_labels_ft[abs(sparse_labels_ft).sum(dim=1) != 0], 1)  # shape = [num_labelled_points]
        _, counts = torch.unique(labeled_points, return_counts=True)

        del template, sparse_labels_ft
        torch.cuda.empty_cache()

        x = np.ascontiguousarray(labeled_features.cpu().numpy()).astype(np.float32)  # labeled_features shape = [num_labelled_points, 1024]
        q = np.ascontiguousarray(features_ft.cpu().numpy()).astype(np.float32)  # shape = [image_size x image_size, 1024]

        res = faiss.StandardGpuResources()  # use a single GPU

        upsample = torch.nn.Upsample((original_image_size, original_image_size), mode='nearest')  # shape = [1, 1, image_size, image_size]

        # if we do L2 normalization and then use the metric inner product, the result is the cosine similarity
        # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
        index = faiss.index_factory(num_feats, "Flat", faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.ntotal
        faiss.normalize_L2(x)
        gpu_index.add(x)
        faiss.normalize_L2(q)
        similarities, sorted_indices = gpu_index.search(q, k)  
        gpu_index.reset()

        labeled_points = labeled_points.cpu().numpy()
        sorted_classes = labeled_points[sorted_indices] - 1   # shape = [image_size x image_size, k]

        similarities_torch = torch.from_numpy(similarities)
        similarities_torch_start =  torch.from_numpy(np.reshape(similarities, (image_size, image_size))) 

        num_smart_pixels = torch.floor(num_total_pixels - torch.sum(counts))

        if num_smart_pixels > 0:
            current_sparse_labels = sparse_labels.clone()
            sorted_classes_torch = torch.from_numpy(sorted_classes)
            for p in range(num_smart_pixels):
                new_similarities_torch, new_similarity_mask_torch, new_nearest, new_sparse_labels, new_sorted_classes = human_in_the_loop_labeling(GT_mask_torch, similarities_torch, current_sparse_labels, sigma, cos_weight, k, image_name)

                similarities_torch = new_similarities_torch
                current_sparse_labels = new_sparse_labels
                sorted_classes_torch = new_sorted_classes

        else:
            current_sparse_labels = sparse_labels
            if k > 1:
                sorted_classes = torch.from_numpy(sorted_classes)
                new_nearest = torch.mode(sorted_classes)[0]
                new_similarity_mask_torch = torch.reshape(new_nearest, (image_size, image_size)) 
            
            else:
                new_similarity_mask_torch = torch.reshape(torch.from_numpy(sorted_classes[:,0]), (image_size, image_size)) 

        similarity_mask_upsample = upsample(new_similarity_mask_torch.unsqueeze(0).unsqueeze(0).float()).squeeze().int().to(device) # shape = [1, 1, orig_image_size, orig_image_size]

        similarity_mask_upsample_np = similarity_mask_upsample.clone().cpu().numpy()
        
        if save_path is not None:
            # Save the propagated mask as a .png file in the specified directory
            propagated_as_image = Image.fromarray(similarity_mask_upsample_np.astype(np.uint8))
            propagated_as_image.save(os.path.join(save_path, image_name[:-4])+".png", "PNG")

        if vis_path is not None:
            save_output_mask(np.uint8(new_similarity_mask_torch.cpu().numpy()), sparse_labels.int(), image_name[:-4], vis_path, image_size=448)
            save_labels(image_name, image_path, vis_path, current_sparse_labels, tag="sparse_labels_at_end")

        # Perform evaluation against the dense ground truth mask
        GT_orig = torch.from_numpy(np.array(Image.open(os.path.join(gt_path,image_name)))).int()
        GT_orig = GT_orig.to(device)

        c_mat = confmat(similarity_mask_upsample, GT_orig)

    c_mat = confmat.compute()
    confusion_matrix = c_mat.cpu().numpy()

    print("####################################################################")

    print("human-in-the-loop labeling with num labels:", num_total_pixels)
    print("k used in nearest neighbors:", k)
    print("number of initial human points:", num_human_points)

    global_accuracy = calculate_global_accuracy(confusion_matrix, ignore_class=unlabeled)
    print("Global Accuracy (ignoring class {}): {:.6f}".format(unlabeled, global_accuracy))

    class_accuracies, mean_pixel_accuracy = calculate_pixel_accuracy(confusion_matrix, ignore_class=unlabeled)
    formatted_accuracies = ["{:.6f}".format(acc) for acc in class_accuracies]
    print("Pixel Accuracy for each class (ignoring class {}): ".format(unlabeled), formatted_accuracies)
    print("Mean Pixel Accuracy (ignoring class {}): {:.6f}".format(unlabeled, mean_pixel_accuracy))

    iou_scores, mean_iou = calculate_iou(confusion_matrix, ignore_class=unlabeled)
    formatted_ious = ["{:.6f}".format(iou) for iou in iou_scores]
    print("IoU Scores for each class (ignoring class {}): ".format(unlabeled), formatted_ious)
    print("Mean IoU (ignoring class {}): {:.6f}".format(unlabeled, mean_iou))
