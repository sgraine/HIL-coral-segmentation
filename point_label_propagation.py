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
from vis_utils import save_labels, save_output_mask
from grid_pixels import select_grid_pixels

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
    parser.add_argument('-k', '--knn', action='store', type=int, default=1, dest='k', help='k value in KNN')
    parser.add_argument('-n', '--num_total_pixels', action='store', type=int, default=5, dest='num_total_pixels', help='the number of labeled points per image')
    
    # Flags to specify functionality
    parser.add_argument('--random', action='store_true', dest='random', help='use this flag when you would like to use randomly distributed points, otherwise the default is to use grid-spaced points')

    # Save input arguments as variables
    args = parser.parse_args()

    random = args.random

    image_path = args.image_path 
    gt_path = args.gt_path
    save_path = args.save_path
    vis_path = args.vis_path
    
    # Set hyperparameters
    k = args.k  # k=1
    num_total_pixels = args.num_total_pixels

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

        # FOR RANDOM LABELS:
        if random:
            sparse_mask = np.zeros(image_height*image_width, dtype=int)
            sparse_mask[:num_total_pixels] = 1
            np.random.shuffle(sparse_mask)
            sparse_mask = np.reshape(sparse_mask, (image_height, image_width))

        # FOR GRID LABELS:
        else:
            # Use a grid to select a subset of the labelled points in the ground truth mask
            sparse_mask = torch.from_numpy(select_grid_pixels((image_height, image_width), num_total_pixels))  # shape = [image_size,image_size]

        # We add one to all the classes so that '0' becomes all the unlabeled pixels
        sparse_labels = torch.add(GT_mask_torch, 1) * sparse_mask
        sparse_labels = torch.unsqueeze(torch.unsqueeze(sparse_labels, 0),0).to(device)              # shape = [B, 1, H, W]

        if vis_path is not None:
            save_labels(image_name, image_path, vis_path, sparse_labels)

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

        # Perform evaluation against the dense ground truth mask
        GT_orig = torch.from_numpy(np.array(Image.open(os.path.join(gt_path,image_name)))).int()
        GT_orig = GT_orig.to(device)

        c_mat = confmat(similarity_mask_upsample, GT_orig)
        break

    c_mat = confmat.compute()
    confusion_matrix = c_mat.cpu().numpy()

    print("####################################################################")

    print("standard pixels, num labels:", num_total_pixels)
    print("k nearest neighbours:", k)

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
