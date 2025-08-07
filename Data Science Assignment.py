import os
import glob
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def create_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def cleardir(tempfolder):
    for filepath in glob.glob(os.path.join(tempfolder, "*")):
        os.unlink(filepath)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        color = np.array([*cmap(obj_id % 10)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', linewidth=2))

def process_img_png_mask(image_path, mask_path, visualize=False):
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(image)
        show_box([xmin, ymin, xmax, ymax], ax)
        plt.title("Bounding Box on Original Image")
        plt.show()

    return xmin, xmax, ymin, ymax

def track_item_boxes(imgpath1, imgpath2, box_list, visualize=True):
    tempfolder = "./tempdir"
    create_if_not_exists(tempfolder)
    cleardir(tempfolder)

    shutil.copy(imgpath1, os.path.join(tempfolder, "00000.jpg"))
    shutil.copy(imgpath2, os.path.join(tempfolder, "00001.jpg"))

    model_cfg = "sam2_hiera_t.yaml"
    checkpoint = "sam2_hiera_tiny.pt"
    predictor_vid = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda')

    state = predictor_vid.init_state(video_path=tempfolder)
    predictor_vid.reset_state(state)

    for (bbox, obj_id) in box_list:
        box = np.array([bbox[0], bbox[2], bbox[1], bbox[3]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor_vid.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            box=box,
        )

    results = {}
    for frame_idx, out_obj_ids, out_mask_logits in predictor_vid.propagate_in_video(state):
        results[frame_idx] = {
            obj_id: (logit > 0).cpu().numpy()
            for obj_id, logit in zip(out_obj_ids, out_mask_logits)
        }

    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(Image.open(imgpath1))
        show_box(box_list[0][0], ax)
        plt.title("Original Image with Bounding Box")
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(Image.open(imgpath2))
        for obj_id, mask in results[1].items():
            show_mask(mask, ax, obj_id)
        plt.title("Tracked Object in Second Image")
        plt.show()

    return results

if __name__ == "__main__":
    img1 = "./CMU10_3D/data_2D/can_chowder_000001.jpg"
    mask1 = "./CMU10_3D/data_2D/can_chowder_000001_1_gt.png"
    img2 = "./CMU10_3D/data_2D/can_chowder_000002.jpg"

    xmin, xmax, ymin, ymax = process_img_png_mask(img1, mask1, visualize=True)
    box_list = [([xmin, xmax, ymin, ymax], 1)]
    tracked_output = track_item_boxes(img1, img2, box_list, visualize=True)
