import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import marimo as mo
    import numpy as np
    import hydra
    import sys
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import cv2
    from transformers import AutoTokenizer

    sys.path.append("../src")
    from csi_slt.data.datamodule import DataModule

    BATCH_SIZE = 2
    print("Current working directory:", os.getcwd())


@app.cell
def _():
    with hydra.initialize_config_dir(config_dir=os.path.abspath("../configs")):
        cfg = hydra.compose(config_name="base_train")
        cfg.data.data_root = "/root/projects/sign_langauge_visual_pretrain/dataset/PHOENIX-2014-T-release-v3"

    cfg.data.chat_template_jinjia = "../jinjas/gemma_slt.jinja"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    datamodule = DataModule(
        cfg.data,
        tokenizer=tokenizer,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset
    collator = datamodule.train_collator
    collator.debug = True

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    return (loader,)


@app.function
def tensor_to_image(image):
    image = image.clone()
    image = image.permute(0, 2, 3, 1)  # C, H, W to H, W, C
    image = image * 0.5 + 0.5  # assuming mean=0.5, std=0.5
    image = image.clamp(0, 1)
    image = image.numpy()

    return image


@app.cell
def visualize_batch(loader):
    for batch in loader:
        input_text = batch["input_text"]
        label_text = batch["label_text"]
        pixel_values = batch["pixel_values"]
        pixel_values_lengths = batch["pixel_values_length"]
        original_videos = batch["original_videos"]
        break

    videos = tensor_to_image(pixel_values)
    indices = np.cumsum(pixel_values_lengths.numpy())[:-1]
    videos_split = np.split(videos, indices, axis=0)

    N = 2
    fig, axes = plt.subplots(BATCH_SIZE, N, figsize=(16, 8))
    for b in range(BATCH_SIZE):
        for n in range(N):
            axes[b, n].imshow(videos_split[b][n])
            axes[b, n].axis("off")
            axes[b, n].set_title("Anchor")

    plt.tight_layout()
    plt.show()
    print(input_text)
    print(label_text)
    return (original_videos,)


@app.function
def float_frames_to_video(
    frames: np.ndarray, output_path: str, fps: float = 20.0, is_color: bool = True
):
    """
    frames: numpy array, shape = (T, H, W, C) 或 (T, H, W)，像素值在 [0.0, 1.0]
    输出视频文件 output_path
    """
    # 检查维度
    if frames.ndim == 4:
        T, H, W, C = frames.shape
        if C == 3:
            color_flag = True
        elif C == 1:
            color_flag = False
            frames = frames.reshape((T, H, W))
        else:
            raise ValueError("通道数应为 1 或 3")
    elif frames.ndim == 3:
        T, H, W = frames.shape
        color_flag = False
    else:
        raise ValueError("frames 必须是 3 维或 4 维")

    # 转换浮点帧到 uint8
    # 方法：先 clip 到 [0,1]，乘 255，再四舍五入 / 转换为 uint8

    frames_uint8 = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # Use "avc1" for H.264 codec, which is more broadly compatible with browsers.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或者尝试 "avc1"
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=color_flag)
    print(color_flag)
    if not out.isOpened():
        raise RuntimeError(f"无法打开 VideoWriter，检查输出路径 / codec / 参数")

    for i in range(T):
        frame = frames_uint8[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if color_flag else frame
        out.write(frame)

    out.release()
    print("视频已保存到", output_path)


@app.cell
def videos_splitsave_video(original_videos):
    float_frames_to_video(
        original_videos[0],
        output_path="./sample_video.mp4",
        fps=30.0,
        is_color=True,
    )
    return


@app.cell
def _():
    # 假设你的视频文件名为 "my_video.mp4"，并且放在 marimo 应用程序可以访问到的路径
    video_path = "./sample_video.mp4"

    # 使用 mo.html 来创建 HTML <video> 元素
    video_element = mo.Html(
        f"""
        <video controls width="640" height="360">
            <source src="{video_path}" type="video/mp4">
            您的浏览器不支持 HTML video 标签。
        </video>
        """
    )

    mo.hstack([video_element])
    return


if __name__ == "__main__":
    app.run()
