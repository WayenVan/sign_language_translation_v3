import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import marimo as mo
    import hydra
    import sys
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torch

    sys.path.append("../src")
    from csi_sign_pretrain.data.datamodule import DataModule
    from csi_sign_pretrain.modeling_sign_visual.sign_pt_model import (
        SignVisualModelForPretrain,
    )
    import torch.nn.functional as F
    from sklearn.decomposition import PCA

    CHECKPOINT_PATH = "/root/projects/sign_langauge_visual_pretrain/outputs/info_nce_pretrain/2025-10-04_03-47-05/checkpoint-10500"
    BATCH_SIZE = 1
    print("Current working directory:", os.getcwd())


@app.cell
def _():
    with hydra.initialize(
        config_path="../../root/projects/sign_langauge_visual_pretrain/configs"
    ):
        cfg = hydra.compose(config_name="base_eval")
        cfg.data.data_root = "/root/projects/sign_langauge_visual_pretrain/dataset/PHOENIX-2014-T-release-v3"

    datamodule = DataModule(
        cfg.data,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=datamodule.collator,
    )
    return (loader,)


@app.cell
def _(loader):
    model = SignVisualModelForPretrain.from_pretrained(CHECKPOINT_PATH)

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"]  # (B, C, H, W)
            feats = model(pixel_values=pixel_values).feats  # (B, D)
            break
    feats = F.normalize(feats, dim=-1)
    return feats


@app.cell
def visualize_logits(feats):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(feats.numpy())

    # 绘制散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c="blue")

    # 给每个点加上索引
    for i in range(X_pca.shape[0]):
        plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, str(i), fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Sample Index")
    plt.show()


@app.function
def tensor_to_image(image):
    image = image.clone()
    image = image.permute(0, 2, 3, 1)  # B, C, H, W to H, W, C
    image = image * 0.5 + 0.5  # assuming mean=0.5, std=0.5
    image = image.clamp(0, 1)
    image = image.numpy()
    return image


if __name__ == "__main__":
    app.run()
