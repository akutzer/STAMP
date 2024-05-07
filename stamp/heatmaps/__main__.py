import argparse
from collections.abc import Collection
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from fastai.vision.learner import Learner, load_learner
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from PIL import Image
from torch import Tensor
import math
from tqdm import tqdm

from stamp.preprocessing.helpers.common import supported_extensions



def load_slide_ext(wsi_dir: Path) -> openslide.OpenSlide:
    # Check if any supported extension matches the file
    if wsi_dir.suffix not in supported_extensions:
        raise FileNotFoundError(
            f"No supported slide file found for slide {wsi_dir.name} in provided directory {wsi_dir.parent}\
                                 \nOnly support for: {supported_extensions}"
        )
    else:
        return openslide.open_slide(wsi_dir)


def get_stride(coords: Tensor) -> int:
    xs = coords[:, 0].unique(sorted=True)
    stride = (xs[1:] - xs[:-1]).min()
    return stride


@torch.enable_grad()
def gradcam_per_category(
    learn: Learner, feats: Tensor, slide_ids: Tensor, coords: Tensor,
    categories: Collection, legacy: bool = False
) -> tuple[Tensor, Tensor]:
    
    feats_batch = feats.expand((len(categories), *feats.shape)).detach().requires_grad_()
    # feats_batch.shape: (num_categories, num_tiles, d)

    gradcam_per_category.activation = None
    gradcam_per_category.grad = None

    # register hooks after normalization layer of the last multi-head 
    # self-attention (MHSA) block 
    def fwd_hook(module, args, output):
        gradcam_per_category.activation = output[:, 1:]
    def bckw_hook(module, grad_input, grad_output):
        gradcam_per_category.grad = grad_output[0][:, 1:]

    layer = learn.model.transformer.layers[-1][0].norm
    # layer = learn.model.transformer.layers[0][0].norm

    hook_handles = []
    hook_handles.append(layer.register_forward_hook(fwd_hook))
    hook_handles.append(layer.register_full_backward_hook(bckw_hook))

    # gradient computation for Grad-CAM
    pred, attn_mask = learn.model(
        feats_batch,
        torch.tensor([len(feats)] * len(categories), device=feats_batch.device),
        slide_ids,
        coords,
        return_last_attn=True
    )
    pred.trace().backward()

    activation = gradcam_per_category.activation
    grad =  gradcam_per_category.grad
    gradcam = (activation * grad.mean(dim=-2, keepdim=True)).sum(dim=-1).relu_()
    # negated gradients can be used for `counterfactual explanations`
    neg_gradcam = (activation * (-grad).mean(dim=-2, keepdim=True)).sum(dim=-1).relu_()
    

    # old gradcam code from Maru
    if legacy:
        gradcam = (feats_batch.detach() * feats_batch.grad).mean(dim=-1).abs_()

    # remove hooks after using them to avoid memory leaks
    for handle in hook_handles:
        handle.remove()
    
    preds = torch.softmax(pred, dim=-1)
    # attention of the [CLS] token from the last MHSA block
    attn_mask = attn_mask[0, :, 0]
    
    return preds, gradcam, attn_mask


def vals_to_im(
    scores: Float[Tensor, "n_tiles *d_feats"],
    norm_coords: Int[Tensor, "n_tiles *d_feats"],
) -> Float[Tensor, "i j *d_feats"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = norm_coords.max(0).values.flip(0) + 1
    im = torch.zeros((*size, *scores.shape[1:]))
    mask = torch.zeros(*size, dtype=torch.bool)

    flattened_im = im.flatten(end_dim=1)
    flattened_mask = mask.flatten(end_dim=1)
    flattened_coords = norm_coords[:, 1] * im.shape[1] + norm_coords[:, 0]
    flattened_im[flattened_coords] = scores
    flattened_mask[flattened_coords] = 1

    im = flattened_im.reshape_as(im)
    mask = flattened_mask.reshape_as(mask)
    
    return im, mask


def show_thumb(slide, thumb_ax: Axes, attention: Tensor) -> None:
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    thumb_ax.set_title("Original WSI")
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


def show_class_map(
    class_ax: Axes, top_scores: Tensor, categories: Collection[str]
) -> None:
    cmap = plt.get_cmap("Set1")
    classes = cmap(top_scores.indices[..., 0])

    c = len(categories)
    # clips values from range [1 / c, 1] to [0, 1]
    alpha = (top_scores.values[..., 0] * c - 1) / (c - 1)
    classes[..., -1] = alpha.clip_(0, 1)
    # classes[..., -1] = top_scores.values[..., 0] != 0

    class_ax.imshow(classes)
    class_ax.legend(handles=[
        Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
    ])
    class_ax.set_title("Prediction of each tile")

def plot_attention_heads(attn_mask_2d, mask, slide_output_dir, h5_path):
    heads = attn_mask_2d.shape[-1]
    nrows = 2
    ncols = math.ceil(heads / nrows)
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols*3, nrows*3))
    plt.tight_layout(pad=1.5)
    fig.suptitle("Attention heads of $[CLS]$ token")
    for i in range(heads):
        # attn_mask_img = plt.get_cmap("viridis")(attn_mask_2d[:, :, i] / attn_mask_2d[:, :, i].max())
        # attn_mask_img[..., -1] = mask
        # axs[i // ncols, i % ncols].imshow(attn_mask_img)
        axs[i // ncols, i % ncols].imshow(attn_mask_2d[:, :, i])
        axs[i // ncols, i % ncols].axis("off")
    fig.savefig(slide_output_dir / f"{h5_path.stem}-attention-heads.png")


def get_n_toptiles(
    slide,
    category: str,
    output_dir: Path,
    coords: Tensor,
    scores: Tensor,
    stride: int,
    n: int = 8,
) -> None:
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

    (output_dir / f"toptiles_{category}").mkdir(exist_ok=True, parents=True)

    # determine the scaling factor between heatmap and original slide
    # 256 microns edge length by default, with 224px = ~1.14 MPP (± 10x magnification)
    feature_downsample_mpp = (
        256 / stride
    )  # NOTE: stride here only makes sense if the tiles were NON-OVERLAPPING
    scaling_factor = feature_downsample_mpp / slide_mpp

    top_score = scores.topk(n)

    # OPTIONAL: if the score is not larger than 0.5, it's indecisive on directionality
    # then add [top_score.values > 0.5]
    top_coords_downscaled = coords[top_score.indices]
    top_coords_original = np.uint(top_coords_downscaled * scaling_factor)

    # NOTE: target size (stride, stride) only works for NON-OVERLAPPING tiles
    # that were extracted in previous steps.
    for score_idx, pos in enumerate(top_coords_original):
        tile = (
            slide.read_region(
                (pos[0], pos[1]),
                0,
                (np.uint(stride * scaling_factor), np.uint(stride * scaling_factor)),
            )
            .convert("RGB")
            .resize((stride, stride))
        )
        tile.save(
            (output_dir / f"toptiles_{category}")
            / f"score_{top_score.values[score_idx]:.2f}_toptiles_{category}_{(pos[0], pos[1])}.jpg"
        )


@torch.no_grad()
def main(
    slide_name: str,
    feature_dir: Path,
    wsi_dir: Path,
    model_path: Path,
    output_dir: Path,
    n_toptiles: int = 8,
    overview: bool = True,
    legacy: bool = False,
    device: str = "cpu"
) -> None:
    learn = load_learner(model_path)
    learn.model.eval()
    categories: Collection[str] = learn.dls.train.dataset._datasets[
        -1
    ].encode.categories_[0]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    learn.model.to(device)

    # for h5_path in feature_dir.glob(f"**/{slide_name}.h5"):
    for slide_path in tqdm(list(wsi_dir.glob(f"**/{slide_name}.*"))):
        h5_path = feature_dir / slide_path.with_suffix(".h5").name
        slide_output_dir = output_dir / h5_path.stem
        slide_output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Creating heatmaps for {slide_path.name}...")
        with h5py.File(h5_path) as h5:
            feats = torch.tensor(h5["feats"][:],
                                 dtype=torch.float, device=device
            )
            coords = torch.tensor(h5["coords"][:].astype(np.int32),
                                  dtype=torch.int, device=device
            )
            slide_ids = torch.zeros(h5["feats"][:][:].shape[0],
                                    dtype=torch.int, device=device
            )
    
        # min-max normalize coords
        coords_norm = (coords - coords.amin(axis=0, keepdim=True)).to(torch.float)
        coords_norm = coords_norm / (1e-8 + coords_norm.amax(axis=0, keepdim=True))
    
        # calculate the gradcam and tile level predictions on the gpu,
        # afterwards everything is done on the cpu
        preds, gradcam, attn_mat = gradcam_per_category(
            learn=learn, feats=feats, slide_ids=slide_ids[None],
            coords=coords_norm[None], categories=categories, legacy=legacy
        )
        preds, gradcam, attn_mat = preds.cpu(), gradcam.cpu(), attn_mat.cpu()

        tile_preds = learn.model(
            feats[:, None],
            torch.ones(len(feats), dtype=torch.int, device=feats.device),
            slide_ids[:, None],
            coords_norm[:, None]
        ).softmax(dim=-1)
        tile_preds = tile_preds.cpu()
        
        # stride is 224 using normal operations
        coords = coords.cpu()
        stride = get_stride(coords)
        grid_coords = torch.div(coords, stride, rounding_mode="floor")
        gradcam_2d, mask = vals_to_im(gradcam.permute(-1, -2), grid_coords)
        tile_preds_2d, _ = vals_to_im(tile_preds, grid_coords)

        # plot attention matricies
        attn_mat = attn_mat[:, 1:] # ignore class token, but no ideal
        attn_mat_2d, _ = vals_to_im(attn_mat.permute(-1, -2), grid_coords)
        plot_attention_heads(attn_mat_2d, mask, slide_output_dir, h5_path)

        # plot heatmaps
        fig, axs = plt.subplots(nrows=2, ncols=max(2, len(categories)), figsize=(12, 8))
        plt.tight_layout(pad=1.5)
        show_class_map(
            class_ax=axs[0, 1],
            top_scores=tile_preds_2d.topk(2),
            categories=categories,
        )

        slide = load_slide_ext(slide_path)
        for ax, (pos_idx, category) in zip(axs[1, :], enumerate(categories)):
            ax: Axes
            if legacy:
                topk = tile_preds_2d.topk(2)
                category_support = torch.where(
                    # To get the "positiveness",
                    # it checks whether the "hot" class has the highest score for each pixel
                    topk.indices[..., 0] == pos_idx,
                    # Then, if the hot class has the highest score,
                    # it assigns a positive value based on its difference from the second highest score
                    tile_preds_2d[..., pos_idx] - topk.values[..., 1],
                    # Likewise, if it has NOT a negative value based on the difference of that class' score to the highest one
                    tile_preds_2d[..., pos_idx] - topk.values[..., 0],
                )
                # So, if we have a pixel with scores (.4, .4, .2) and would want to get the heat value for the first class,
                # we would get a neutral color, because it is matched with the second class
                # But if our scores were (.4, .3, .3), it would be red,
                # because now our class is .1 above its nearest competitor

                attention = torch.where(
                    topk.indices[..., 0] == pos_idx,
                    gradcam_2d[..., pos_idx],
                    gradcam_2d[..., list(set(range(len(categories))) - {pos_idx})].amax(dim=-1)
                )
                attention = attention / gradcam_2d[mask].max()

                weighted_gradcam = category_support * attention / 2 + .5
                score_im = plt.get_cmap("RdBu_r")(weighted_gradcam)
            
            else:
                category_support = 2 * tile_preds_2d[..., pos_idx] - 1
                attention = gradcam_2d[..., pos_idx] / gradcam_2d[mask].max()
                score_im = plt.get_cmap("Reds")(attention)

            # score_im[..., -1] = mask

            ax.imshow(score_im)
            ax.set_title(f"{category} {preds[0,pos_idx]:1.2f}")

            Image.fromarray(np.uint8(score_im * 255)).resize(
                np.array(score_im.shape[:2][::-1]) * 8, resample=Image.NEAREST
            ).save(
                slide_output_dir
                / f"scores-{h5_path.stem}--score_{category}={preds[0][pos_idx]:0.2f}.png"
            )

            get_n_toptiles(
                slide=slide,
                category=category,
                stride=stride,
                output_dir=slide_output_dir,
                scores=tile_preds[:, pos_idx],
                coords=coords.cpu(),
                n=n_toptiles,
            )

        if overview:
            thumb = show_thumb(
                slide=slide,
                thumb_ax=axs[0, 0],
                attention=attention,
            )
            Image.fromarray(thumb).save(slide_output_dir / f"thumbnail-{h5_path.stem}.png")

            for ax in axs.ravel():
                ax.axis("off")

            fig.savefig(slide_output_dir / f"overview-{h5_path.stem}.png")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("heatmaps")
    parser.add_argument(
        "--slide-name",
        metavar="PATH",
        type=str,
        required=True,
        help="Name of the WSI to create heatmap for (no extensions)",
    )
    parser.add_argument(
        "--wsi-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the SVSs",
    )
    parser.add_argument(
        "--feature-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the slides' features",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the trained model's export.pkl",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory to save the heatmaps to",
    )
    parser.add_argument(
        "--n-toptiles",
        type=int,
        default=8,
        required=False,
        help="Number of toptiles to generate, 8 by default",
    )
    parser.add_argument(
        "--overview",
        type=bool,
        default=True,
        required=False,
        help="Generate final overview image",
    )
    args = parser.parse_args()
    main(**vars(args))
