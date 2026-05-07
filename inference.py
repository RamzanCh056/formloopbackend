"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter

# MattingNetwork F.interpolate scale_factor: 1.0 = full-res backbone — best for thin objects (bars, hair).
# Lower values are faster but drop fine detail. Do not blur alpha post-process; that erodes thin foreground.
DEFAULT_DOWNSAMPLE_RATIO = 1.0


def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  output_video_pix_fmt: str = "yuv420p",
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  progress_interval: int = 12,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: Backbone scale factor; if None, uses DEFAULT_DOWNSAMPLE_RATIO (1.0 = full res, best for thin detail).
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        progress_interval: Emit "RVM_PROGRESS" log every N frames.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 8 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000),
                pix_fmt=output_video_pix_fmt,
            )
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000),
                pix_fmt=output_video_pix_fmt,
            )
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000),
                pix_fmt=output_video_pix_fmt,
            )
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            # Recurrent states persist across the whole sequence (not reset per frame or per batch).
            rec = [None] * 4
            ds = downsample_ratio if downsample_ratio is not None else DEFAULT_DOWNSAMPLE_RATIO
            total_frames = int(len(source))
            done_frames = 0
            emit_every = max(1, int(progress_interval))
            last_emit = -emit_every
            for src in reader:

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, ds)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    writer_com.write(com[0])
                
                bar.update(src.size(1))
                done_frames += int(src.size(1))
                if total_frames > 0:
                    pct = int(min(100, max(0, (done_frames * 100) // total_frames)))
                    if (done_frames - last_emit >= emit_every) or (done_frames >= total_frames):
                        print(f"RVM_PROGRESS:{done_frames}/{total_frames}:{pct}", flush=True)
                        last_emit = done_frames

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().float()
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = self.model.to(device)
        self.model = torch.jit.script(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=8)
    parser.add_argument(
        '--output-video-pix-fmt',
        type=str,
        default='yuv420p',
        choices=('yuv420p', 'yuv444p'),
        help='H.264 pixel format: yuv444p keeps full chroma (better for thin bars/weights); yuv420p smaller.',
    )
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    parser.add_argument('--progress-interval', type=int, default=12)
    args = parser.parse_args()
    
    converter = Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        output_video_pix_fmt=args.output_video_pix_fmt,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress,
        progress_interval=args.progress_interval,
    )
    
    
