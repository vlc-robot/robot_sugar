import matplotlib.pyplot as plt
from pathlib import Path

import torch

def plot_attention(
    attentions: torch.Tensor, rgbs: torch.Tensor, pcds: torch.Tensor, dest: Path
) -> plt.Figure:
    attentions = attentions.detach().cpu()
    rgbs = rgbs.detach().cpu()
    pcds = pcds.detach().cpu()

    ep_dir = dest.parent
    ep_dir.mkdir(exist_ok=True, parents=True)
    name = dest.stem
    ext = dest.suffix

    # plt.figure(figsize=(10, 8))
    num_cameras = len(attentions)
    for i, (a, rgb, pcd) in enumerate(zip(attentions, rgbs, pcds)):
        # plt.subplot(num_cameras, 4, i * 4 + 1)
        plt.imshow(a.permute(1, 2, 0).log())
        plt.axis("off")
        plt.colorbar()
        plt.savefig(ep_dir / f"{name}-{i}-attn{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

        # plt.subplot(num_cameras, 4, i * 4 + 2)
        # plt.imshow(a.permute(1, 2, 0))
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.clf()

        # plt.subplot(num_cameras, 4, i * 4 + 3)
        plt.imshow(((rgb + 1) / 2).permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(ep_dir / f"{name}-{i}-rgb{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

        pcd_norm = (pcd - pcd.min(0).values) / (pcd.max(0).values - pcd.min(0).values)
        # plt.subplot(num_cameras, 4, i * 4 + 4)
        plt.imshow(pcd_norm.permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(ep_dir / f"{name}-{i}-pcd{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

    return plt.gcf()