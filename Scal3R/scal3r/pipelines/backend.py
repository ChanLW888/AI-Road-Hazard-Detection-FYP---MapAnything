import gc
import os
import csv
import shutil
import torch
import argparse
import numpy as np
import cv2

from typing import List
from os.path import join
import time as wall_time
from copy import deepcopy
from einops import rearrange

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from scal3r.utils.ray_utils import get_rays
from scal3r.utils.data_utils import to_cuda
from scal3r.utils.result_utils import save_results
from scal3r.models import build_sampler_from_config
from scal3r.utils.base_utils import DotDict as dotdict
from scal3r.utils.cam_utils import decode_camera_params
from scal3r.utils.math_utils import affine_inverse, affine_padding
from scal3r.engine.path import get_default_output_dir, resolve_release_path
from scal3r.utils.console_utils import (
    get_logger,
    log_block,
    log_exceptions,
    tqdm,
)
from scal3r.utils.runtime_utils import (
    StageRecorder,
    StopAfterStage,
    maybe_stop_after,
    release_memory,
)
from scal3r.utils.image_utils import (
    build_image_only_block,
    collect_image_paths,
    load_and_preprocess_images,
)
from scal3r.utils.loop import (
    accumulate_transform,
    build_loop_batches,
    build_map_processor,
    build_sim3_loop_optimizer,
    combine_transform,
    detect_loops,
    visualize_loop,
)
from scal3r.utils.offload_utils import (
    cleanup_offload_root,
    clear_dpt_state,
    get_offload_root,
    get_runtime_root,
    materialize_payload,
    materialize_tensor_dict,
    offload_batch_block,
    offload_output_block,
    persist_dpt_state,
    remove_payload,
    should_release_runtime_state,
    store_agg_state,
)


logger = get_logger("scal3r.backend")


# ============================================================
# YOLO CONFIGURATION
# ============================================================

DEFAULT_YOLO_MODEL = (
    "/home/lcha0115/bt60_scratch/lcha_data/Scal3R/"
    "weights/best.pt"
)


YOLO_CLASS_NAMES = {
    0: "blocked_footpath_trip_hazard",
    1: "fallen_fence",
    2: "grass",
    3: "loose_trash",
    4: "mud",
    5: "paint",
    6: "road",
    7: "sidewalk",
    8: "skip_bin",
}


YOLO_HAZARD_CLASSES = {
    0: "blocked_footpath_trip_hazard",
    1: "fallen_fence",
    3: "loose_trash",
    4: "mud",
    5: "paint",
    8: "skip_bin",
}


# ============================================================
# SCAL3R
# ============================================================

def collect_intermediate_layers(
    model,
    start_index: int,
) -> list[int]:

    return [
        start_index + offset
        for offset in range(
            model.agg_regator.aa_block_size
        )
        if start_index + offset
        in model.agg_regator.intermediate_layer_idx
    ]


def format_runtime_config(args) -> list[str]:

    return [
        f"block_size: {args.block_size}",
        f"overlap_size: {args.overlap_size}",
        f"use_loop: {bool(args.use_loop)}",
        f"preprocess_workers: {int(args.preprocess_workers)}",
        f"max_align_points_per_frame: "
        f"{args.max_align_points_per_frame}",
        f"pgo_workers: {int(args.pgo_workers)}",
        f"save_dpt: {bool(args.save_dpt)}",
        f"save_xyz: {bool(args.save_xyz)}",
        f"streaming_state: {bool(args.streaming_state)}",
        f"result_dir: {args.result_dir}",
        f"runtime_dir: {args.runtime_dir}",
        f"offload_batches: {bool(args.offload_batches)}",
        f"offload_outputs: {bool(args.offload_outputs)}",
        f"probe_dir: {args.probe_dir or ''}",
        f"stop_after_stage: {args.stop_after_stage or ''}",
        f"segmentation_enable: "
        f"{bool(args.segmentation_enable)}",
        f"segmentation_model: "
        f"{args.segmentation_model}",
        f"segmentation_confidence: "
        f"{args.segmentation_confidence}",
        f"segmentation_iou: "
        f"{args.segmentation_iou}",
        f"segmentation_imgsz: "
        f"{args.segmentation_imgsz}",
        f"segmentation_batch_size: "
        f"{args.segmentation_batch_size}",
    ]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(
    dataset_cfg: dotdict,
    args,
    recorder: StageRecorder | None = None,
):

    if recorder is not None:

        recorder.record(
            "collect_images.begin",
            input_dir=args.input_dir,
            image_patterns=args.image_patterns,
        )

    image_paths = collect_image_paths(
        args.input_dir,
        args.image_patterns,
    )

    if args.max_images > 0:

        image_paths = image_paths[
            :args.max_images
        ]

    if recorder is not None:

        recorder.record(
            "collect_images.done",
            image_count=len(image_paths),
            first_image=(
                image_paths[0]
                if image_paths
                else ""
            ),
            last_image=(
                image_paths[-1]
                if image_paths
                else ""
            ),
        )

        maybe_stop_after(
            "collect_images.done",
            args,
            recorder,
        )

        recorder.record(
            "preprocess_images.begin",
            image_count=len(image_paths),
            preprocess_workers=int(
                args.preprocess_workers
            ),
        )

    sequence, height, width = (
        load_and_preprocess_images(
            image_paths,
            dataset_cfg,
            preprocess_workers=args.preprocess_workers,
        )
    )

    if recorder is not None:

        recorder.record(
            "preprocess_images.done",
            image_count=len(sequence),
            height=int(height),
            width=int(width),
            preprocess_workers=int(
                args.preprocess_workers
            ),
        )

        maybe_stop_after(
            "preprocess_images.done",
            args,
            recorder,
        )

    n_samples = len(sequence)

    block_size = args.block_size
    overlap_size = args.overlap_size

    assert block_size > overlap_size, (
        f"[ERROR] block_size {block_size} "
        f"must be larger than overlap_size "
        f"{overlap_size}"
    )

    n_srcs = block_size

    n_blocks = (
        (
            n_samples
            - overlap_size
            + (n_srcs - overlap_size)
            - 1
        )
        // (n_srcs - overlap_size)
    )

    if (
        n_blocks == 0
        or n_samples <= block_size
    ):

        block_size = n_samples
        n_srcs = n_samples
        n_blocks = 1

    batches = []
    indices = []

    batch_index = 0

    def build_block(
        sequence,
        block_indices: List[int],
        block_height: int,
        block_width: int,
        block_overlap_size: int,
        block_dataset_cfg: dotdict,
    ):

        nonlocal batch_index

        batch = build_image_only_block(
            sequence,
            block_indices,
            block_height,
            block_width,
            block_overlap_size,
            block_dataset_cfg,
        )

        if args.offload_batches:

            batch = offload_batch_block(
                batch,
                args,
                batch_index,
            )

            release_memory(args.device)

        batch_index += 1

        return batch

    if recorder is not None:

        recorder.record(
            "build_blocks.begin",
            n_samples=int(n_samples),
            block_size=int(block_size),
            overlap_size=int(overlap_size),
            n_blocks=int(n_blocks),
        )

    pbar = tqdm(
        total=n_blocks,
        desc="Loading image blocks",
    )

    for i in range(n_blocks):

        sampler_index = (
            i * (n_srcs - overlap_size)
        )

        block_indices = list(
            range(
                sampler_index,
                min(
                    sampler_index + n_srcs,
                    n_samples,
                ),
            )
        )

        batches.append(
            build_block(
                sequence,
                block_indices,
                height,
                width,
                overlap_size,
                dataset_cfg,
            )
        )

        indices.append(
            (
                sampler_index,
                min(
                    sampler_index + n_srcs,
                    n_samples,
                ),
            )
        )

        pbar.update()

    pbar.close()

    if recorder is not None:

        recorder.record(
            "build_blocks.done",
            n_blocks=int(len(batches)),
            offload_batches=bool(
                args.offload_batches
            ),
            offload_root=(
                get_offload_root(args)
                if args.offload_batches
                else ""
            ),
        )

        maybe_stop_after(
            "build_blocks.done",
            args,
            recorder,
        )

    args.n_blocks = n_blocks

    logger.info(
        "Loading finished, %d blocks are loaded "
        "from %d images",
        len(batches),
        len(image_paths),
    )

    if args.use_loop:

        if recorder is not None:

            recorder.record(
                "loop_detection.begin",
                image_count=len(image_paths),
            )

        os.makedirs(
            get_runtime_root(args),
            exist_ok=True,
        )

        loop_list = detect_loops(
            image_paths,
            result_dir=get_runtime_root(args),
            loop_ckpt=args.loop_ckpt,
            nms_threshold=25,
            min_frame_gap=10,
        )

        batches_loop, indices_loop = (
            build_loop_batches(
                sequence,
                indices,
                loop_list,
                args.loop_size // 2,
                height,
                width,
                dataset_cfg,
                build_block,
                logger.info,
            )
        )

        batches.extend(batches_loop)
        indices.extend(indices_loop)

        args.n_blocks_loop = len(
            batches_loop
        )

        if recorder is not None:

            recorder.record(
                "loop_detection.done",
                loop_pairs=int(
                    len(loop_list)
                ),
                loop_blocks=int(
                    len(batches_loop)
                ),
            )

            maybe_stop_after(
                "loop_detection.done",
                args,
                recorder,
            )

    del sequence

    release_memory(args.device)

    return batches, indices


# ============================================================
# TEST-TIME TRAINING
# ============================================================

def apply_ttt(
    model,
    agg_state_refs,
    dpt_state_refs,
    args,
    layer_index: int,
    dpt_layer_set: set[int],
):

    ttt_order_grad = deepcopy(
        model.ttt_order[0:1]
    )

    ttt_order_grad = [
        order
        ._replace(use_cached=False)
        ._replace(cache_last=False)
        for order in ttt_order_grad
    ]

    ttt_order_apply = deepcopy(
        model.ttt_order[-1:]
    )

    ttt_order_apply = [
        order
        ._replace(use_cached=False)
        ._replace(cache_last=False)
        for order in ttt_order_apply
    ]

    w0_grad_sum = None
    w1_grad_sum = None
    w2_grad_sum = None

    shared_tokens = None

    for block_index in range(
        len(agg_state_refs)
    ):

        agg_state = materialize_payload(
            agg_state_refs[block_index]
        )

        if shared_tokens is None:

            shared_tokens = (
                agg_state["tokens"]
            )

        g0, g1, g2 = (
            model.agg_regator.ttt_gradient(
                index=layer_index,
                ttt_order=ttt_order_grad,
                **to_cuda(
                    agg_state,
                    args.device,
                ),
            )
        )

        if w0_grad_sum is None:

            w0_grad_sum = g0
            w1_grad_sum = g1
            w2_grad_sum = g2

        else:

            w0_grad_sum.add_(g0)
            w1_grad_sum.add_(g1)
            w2_grad_sum.add_(g2)

        del agg_state
        del g0, g1, g2

        if should_release_runtime_state(
            args
        ):

            release_memory(args.device)

    (
        shared_w0,
        shared_w1,
        shared_w2,
    ) = model.agg_regator.ttt_update(
        index=layer_index,
        tokens=shared_tokens,
        w0_grad=w0_grad_sum,
        w1_grad=w1_grad_sum,
        w2_grad=w2_grad_sum,
        ttt_order=ttt_order_grad,
    )

    del shared_tokens

    for block_index in range(
        len(agg_state_refs)
    ):

        agg_state = materialize_payload(
            agg_state_refs[block_index]
        )

        temp_output = (
            {
                layer_index:
                    materialize_payload(
                        dpt_state_refs[
                            block_index
                        ][layer_index]
                    )
            }
            if layer_index in dpt_layer_set
            else None
        )

        updated_state = (
            model.agg_regator.ttt_apply(
                index=layer_index,
                ttt_order=ttt_order_apply,
                output=temp_output,
                w0=shared_w0,
                w1=shared_w1,
                w2=shared_w2,
                **to_cuda(
                    agg_state,
                    args.device,
                ),
            )
        )

        agg_state_refs[
            block_index
        ] = store_agg_state(
            updated_state,
            args,
            block_index,
        )

        if temp_output is not None:

            persist_dpt_state(
                temp_output,
                dpt_state_refs[
                    block_index
                ],
                args,
                block_index,
            )

        del (
            agg_state,
            updated_state,
            temp_output,
        )

        if should_release_runtime_state(
            args
        ):

            release_memory(args.device)

    del (
        w0_grad_sum,
        w1_grad_sum,
        w2_grad_sum,
    )

    del (
        shared_w0,
        shared_w1,
        shared_w2,
    )

    if (
        should_release_runtime_state(args)
        and args.device.startswith("cuda")
    ):

        torch.cuda.empty_cache()


# ============================================================
# FORWARD
# ============================================================

def forward(
    model,
    batches,
    args,
    recorder: StageRecorder | None = None,
):

    output = [
        None
        for _ in range(len(batches))
    ]

    batch0 = materialize_payload(
        batches[0]
    )

    B, S = (
        batch0.meta.rgb.shape[:2]
    )

    H = batch0.meta.H[0].item()
    W = batch0.meta.W[0].item()

    N = len(batches)

    del batch0

    assert B == 1, (
        "[ERROR] this implementation only "
        "supports B=1 for sequential inference, "
        f"got B={B}."
    )

    agg_state_refs = [
        None
        for _ in range(len(batches))
    ]

    dpt_state_refs = [
        {}
        for _ in range(len(batches)
        )
    ]

    dpt_layer_set = set(
        model.agg_regator
        .intermediate_layer_idx
    )

    if recorder is not None:

        recorder.record(
            "embedder.begin",
            n_blocks=int(N),
            frames_per_block=int(S),
            height=int(H),
            width=int(W),
        )

    pbar = tqdm(
        total=N,
        desc="Forward DINOv2 embedder",
    )

    for b, batch_ref in enumerate(
        batches
    ):

        batch = materialize_payload(
            batch_ref
        )

        rgb = rearrange(
            to_cuda(
                batch.meta.rgb,
                args.device,
            ),
            "b n (h w) c -> b n c h w",
            h=H,
            w=W,
        )

        agg_state = (
            model.agg_regator.prepare(
                rgb
            )
        )

        agg_state_refs[b] = (
            store_agg_state(
                agg_state,
                args,
                b,
            )
        )

        block_tokens = (
            int(
                agg_state["tokens"].shape[1]
            )
            if "tokens" in agg_state
            else int(S)
        )

        del (
            batch,
            rgb,
            agg_state,
        )

        if should_release_runtime_state(
            args
        ):

            release_memory(args.device)

        if recorder is not None:

            recorder.record(
                f"embedder_block_{b:02d}.done",
                block_index=int(b),
                block_frames=block_tokens,
            )

        pbar.update()

    pbar.close()

    if recorder is not None:

        recorder.record(
            "embedder.done",
            n_blocks=int(N),
        )

        maybe_stop_after(
            "embedder.done",
            args,
            recorder,
        )

    if recorder is not None:

        recorder.record(
            "aggregator.begin",
            n_layers=int(
                model.agg_regator.aa_block_num
            ),
            ttt_layers=[
                int(i)
                for i in
                model.agg_regator.ttt_layer_idx
            ],
        )

    pbar = tqdm(
        total=model.agg_regator.aa_block_num,
        desc="Forward aggregator",
    )

    for j in range(
        model.agg_regator.aa_block_num
    ):

        forward_intermediate_layers = (
            collect_intermediate_layers(
                model,
                j,
            )
        )

        need_forward_outputs = (
            len(
                forward_intermediate_layers
            )
            > 0
        )

        if recorder is not None:

            recorder.record(
                f"aggregator_layer_{j:02d}.begin",
                layer_index=int(j),
            )

        for b, _ in enumerate(
            batches
        ):

            agg_state = materialize_payload(
                agg_state_refs[b]
            )

            temp_output = (
                {}
                if need_forward_outputs
                else None
            )

            updated_state = (
                model.agg_regator.forward_layer(
                    index=j,
                    output=temp_output,
                    **to_cuda(
                        agg_state,
                        args.device,
                    ),
                )
            )

            agg_state_refs[b] = (
                store_agg_state(
                    updated_state,
                    args,
                    b,
                )
            )

            if temp_output is not None:

                persist_dpt_state(
                    temp_output,
                    dpt_state_refs[b],
                    args,
                    b,
                )

            del (
                agg_state,
                updated_state,
                temp_output,
            )

            if should_release_runtime_state(
                args
            ):

                release_memory(
                    args.device
                )

        with torch.amp.autocast(
            "cuda",
            enabled=False,
        ):

            if (
                (
                    model.agg_regator
                    .frame_use_ttt
                    or
                    model.agg_regator
                    .global_use_ttt
                )
                and
                j in
                model.agg_regator
                .ttt_layer_idx
            ):

                apply_ttt(
                    model,
                    agg_state_refs,
                    dpt_state_refs,
                    args,
                    j,
                    dpt_layer_set,
                )

        pbar.update()

        if recorder is not None:

            recorder.record(
                f"aggregator_layer_{j:02d}.done",
                layer_index=int(j),
                uses_ttt=bool(
                    (
                        model.agg_regator
                        .frame_use_ttt
                        or
                        model.agg_regator
                        .global_use_ttt
                    )
                    and
                    j in
                    model.agg_regator
                    .ttt_layer_idx
                ),
            )

            maybe_stop_after(
                f"aggregator_layer_{j:02d}.done",
                args,
                recorder,
            )

    pbar.close()

    if recorder is not None:

        recorder.record(
            "aggregator.done",
            n_layers=int(
                model.agg_regator.aa_block_num
            ),
        )

        maybe_stop_after(
            "aggregator.done",
            args,
            recorder,
        )

    for b, dpt_state in enumerate(
        dpt_state_refs
    ):

        dpt_state[-1] = (
            dpt_state[
                model.agg_regator.depth - 1
            ]
        )

        remove_payload(
            agg_state_refs[b]
        )

        agg_state_refs[b] = None

    if recorder is not None:

        recorder.record(
            "decoder.begin",
            n_blocks=int(N),
        )

    with torch.amp.autocast(
        "cuda",
        enabled=False,
    ):

        pbar = tqdm(
            total=N,
            desc="Forward decoder",
        )

        for b, batch_ref in enumerate(
            batches
        ):

            if recorder is not None:

                recorder.record(
                    f"decoder_block_{b:02d}.begin",
                    block_index=int(b),
                )

            batch = materialize_payload(
                batch_ref
            )

            rgb_feats = to_cuda(
                materialize_tensor_dict(
                    dpt_state_refs[b]
                ),
                args.device,
            )

            rgb = rearrange(
                batch.meta.rgb,
                "b n (h w) c -> b n c h w",
                h=H,
                w=W,
            ).to(args.device)

            cam_maps = model.cam_decoder(
                rgb_feats
            )

            xyz_map, xyz_cnf = (
                model.xyz_decoder(
                    rgb_feats,
                    images=rgb,
                    patch_start_idx=(
                        model.agg_regator
                        .patch_start_idx
                    ),
                )
            )

            dpt_map, dpt_cnf = (
                model.dpt_decoder(
                    rgb_feats,
                    images=rgb,
                    patch_start_idx=(
                        model.agg_regator
                        .patch_start_idx
                    ),
                )
            )

            xyz_map = (
                xyz_map[..., :H, :W, :]
            )

            dpt_map = (
                dpt_map[..., :H, :W, :]
            )

            xyz_cnf = (
                xyz_cnf[..., :H, :W]
            )

            dpt_cnf = (
                dpt_cnf[..., :H, :W]
            )

            cam_map = (
                cam_maps[-1].cpu()
            )

            block_output = dotdict(
                cam_map=cam_map[..., :9],
                xyz_map=rearrange(
                    xyz_map,
                    "b s h w c -> b s (h w) c",
                ).cpu(),
                dpt_map=rearrange(
                    dpt_map,
                    "b s h w c -> b s (h w) c",
                ).cpu(),
                xyz_cnf=rearrange(
                    xyz_cnf,
                    "b s h w -> b s (h w) 1",
                ).cpu(),
                dpt_cnf=rearrange(
                    dpt_cnf,
                    "b s h w -> b s (h w) 1",
                ).cpu(),
            )

            if cam_map.shape[-1] > 9:

                block_output.scale = (
                    cam_map[..., -1]
                    .mean(dim=-1)
                )

            if args.offload_outputs:

                output[b] = (
                    offload_output_block(
                        block_output,
                        args,
                        b,
                    )
                )

            else:

                output[b] = (
                    block_output
                )

            clear_dpt_state(
                dpt_state_refs[b]
            )

            del (
                batch,
                block_output,
                cam_map,
            )

            del (
                cam_maps,
                xyz_map,
                xyz_cnf,
                dpt_map,
                dpt_cnf,
                rgb_feats,
                rgb,
            )

            if should_release_runtime_state(
                args
            ):

                release_memory(
                    args.device
                )

            if recorder is not None:

                recorder.record(
                    f"decoder_block_{b:02d}.done",
                    block_index=int(b),
                    offload_outputs=bool(
                        args.offload_outputs
                    ),
                )

                maybe_stop_after(
                    f"decoder_block_{b:02d}.done",
                    args,
                    recorder,
                )

            pbar.update()

        pbar.close()

    if recorder is not None:

        recorder.record(
            "decoder.done",
            n_blocks=int(N),
        )

        maybe_stop_after(
            "decoder.done",
            args,
            recorder,
        )

    return output


# ============================================================
# YOLO SEGMENTATION
# ============================================================

def load_yolo_segmentation_model(args):

    if not args.segmentation_enable:

        logger.info(
            "YOLO segmentation disabled."
        )

        return None

    if YOLO is None:

        raise ImportError(
            "Ultralytics is not installed.\n"
            "Install it with:\n\n"
            "pip install ultralytics"
        )

    model_path = (
        str(args.segmentation_model)
    )

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            "YOLO segmentation weights "
            "not found:\n"
            f"{model_path}"
        )

    logger.info(
        "Loading YOLO segmentation model: %s",
        model_path,
    )

    model = YOLO(
        model_path
    )

    task = getattr(
        model,
        "task",
        None,
    )

    logger.info(
        "YOLO task: %s",
        task,
    )

    if task not in (
        None,
        "segment",
    ):

        raise RuntimeError(
            "The supplied YOLO checkpoint "
            "does not appear to be a "
            "segmentation checkpoint.\n"
            f"Ultralytics reports task={task!r}"
        )

    logger.info(
        "YOLO classes:"
    )

    for class_id, class_name in (
        model.names.items()
    ):

        logger.info(
            "  %d: %s",
            int(class_id),
            class_name,
        )

    return model


def validate_yolo_classes(
    model,
):

    if model is None:
        return

    logger.info(
        "Checking YOLO hazard classes..."
    )

    for class_id, expected_name in (
        YOLO_HAZARD_CLASSES.items()
    ):

        actual_name = model.names.get(
            class_id
        )

        if actual_name != expected_name:

            raise RuntimeError(
                "YOLO class mismatch.\n"
                f"Class {class_id}: "
                f"expected "
                f"'{expected_name}', "
                f"but model contains "
                f"'{actual_name}'."
            )

    logger.info(
        "All expected hazard classes verified."
    )


def make_yolo_class_map(
    result,
    target_height,
    target_width,
    confidence_threshold,
):

    class_map = np.full(
        (
            target_height,
            target_width,
        ),
        -1,
        dtype=np.int32,
    )

    confidence_map = np.zeros(
        (
            target_height,
            target_width,
        ),
        dtype=np.float32,
    )

    if result is None:
        return (
            class_map,
            confidence_map,
        )

    if result.masks is None:
        return (
            class_map,
            confidence_map,
        )

    if result.boxes is None:
        return (
            class_map,
            confidence_map,
        )

    if len(result.boxes) == 0:
        return (
            class_map,
            confidence_map,
        )

    masks = (
        result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    order = np.argsort(
        -confidences
    )

    for instance_index in order:

        confidence = float(
            confidences[
                instance_index
            ]
        )

        if (
            confidence
            < confidence_threshold
        ):

            continue

        mask = masks[
            instance_index
        ]

        mask = (
            mask > 0.5
        ).astype(np.uint8)

        if mask.shape != (
            target_height,
            target_width,
        ):

            mask = cv2.resize(
                mask,
                (
                    target_width,
                    target_height,
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                ),
            )

        valid = (
            (mask > 0)
            &
            (
                confidence
                > confidence_map
            )
        )

        class_map[valid] = int(
            classes[instance_index]
        )

        confidence_map[valid] = (
            confidence
        )

    return (
        class_map,
        confidence_map,
    )


def run_yolo_on_block(
    model,
    image_paths,
    frame_indices,
    height,
    width,
    args,
):

    if model is None:

        return {}

    selected = []

    for frame_index in frame_indices:

        if (
            frame_index < 0
            or frame_index
            >= len(image_paths)
        ):

            raise IndexError(
                f"Frame index "
                f"{frame_index} is outside "
                f"image list of length "
                f"{len(image_paths)}."
            )

        selected.append(
            (
                frame_index,
                image_paths[
                    frame_index
                ],
            )
        )

    output = {}

    batch_size = max(
        1,
        int(
            args.segmentation_batch_size
        ),
    )

    for start in range(
        0,
        len(selected),
        batch_size,
    ):

        chunk = selected[
            start:
            start + batch_size
        ]

        paths = [
            path
            for _, path in chunk
        ]

        results = model.predict(
            source=[
                str(path)
                for path in paths
            ],
            conf=float(
                args.segmentation_confidence
            ),
            iou=float(
                args.segmentation_iou
            ),
            imgsz=int(
                args.segmentation_imgsz
            ),
            device=args.device,
            batch=len(paths),
            task="segment",
            verbose=False,
        )

        for (
            frame_index,
            image_path,
        ), result in zip(
            chunk,
            results,
        ):

            class_map, confidence_map = (
                make_yolo_class_map(
                    result,
                    height,
                    width,
                    float(
                        args.segmentation_confidence
                    ),
                )
            )

            output[
                frame_index
            ] = {

                "class_map":
                    class_map,

                "confidence_map":
                    confidence_map,

                "image_path":
                    str(image_path),
            }

    return output


# ============================================================
# SEMANTIC PLY
# ============================================================

def write_semantic_ply(
    path,
    xyz,
    rgb,
    semantic_class,
    semantic_confidence,
):

    """
    Save the complete overlap-filtered Scal3R
    point cloud with semantic highlighting.

    IMPORTANT:

    The XYZ point set is NOT reduced based on YOLO.

    YOLO only changes the colour of points that
    have a semantic label.

    Unlabelled points retain their original RGB.

    The file is binary little-endian PLY.
    """

    xyz = np.asarray(
        xyz,
        dtype=np.float32,
    )

    rgb = np.asarray(
        rgb,
        dtype=np.float32,
    )

    semantic_class = np.asarray(
        semantic_class,
        dtype=np.int32,
    )

    semantic_confidence = np.asarray(
        semantic_confidence,
        dtype=np.float32,
    )

    if not (
        len(xyz)
        == len(rgb)
        == len(semantic_class)
        == len(semantic_confidence)
    ):

        raise ValueError(
            "Semantic PLY arrays have "
            "different lengths."
        )

    # --------------------------------------------------------
    # Remove invalid points.
    # --------------------------------------------------------

    valid = (
        np.isfinite(xyz).all(axis=1)
        &
        np.isfinite(rgb).all(axis=1)
        &
        np.isfinite(
            semantic_confidence
        )
    )

    xyz = xyz[valid]

    rgb = rgb[valid]

    semantic_class = (
        semantic_class[valid]
    )

    semantic_confidence = (
        semantic_confidence[valid]
    )

    # --------------------------------------------------------
    # RGB conversion.
    #
    # Scal3R image tensors may be [0, 1].
    # Convert those to [0, 255].
    # --------------------------------------------------------

    if (
        rgb.size > 0
        and rgb.max() <= 1.0
    ):

        rgb = rgb * 255.0

    rgb = np.clip(
        rgb,
        0,
        255,
    )

    # --------------------------------------------------------
    # Semantic colours.
    # --------------------------------------------------------

    CLASS_COLORS = {

        0: np.array(
            [255, 0, 0],
            dtype=np.float32,
        ),

        1: np.array(
            [255, 128, 0],
            dtype=np.float32,
        ),

        2: np.array(
            [0, 220, 0],
            dtype=np.float32,
        ),

        3: np.array(
            [255, 220, 0],
            dtype=np.float32,
        ),

        4: np.array(
            [150, 75, 0],
            dtype=np.float32,
        ),

        5: np.array(
            [255, 0, 255],
            dtype=np.float32,
        ),

        6: np.array(
            [80, 80, 80],
            dtype=np.float32,
        ),

        7: np.array(
            [0, 150, 255],
            dtype=np.float32,
        ),

        8: np.array(
            [180, 0, 255],
            dtype=np.float32,
        ),
    }

    # --------------------------------------------------------
    # Semantic overlay.
    #
    # 0.0 = original RGB
    # 1.0 = pure semantic colour
    #
    # 0.40 = subtle highlight.
    # --------------------------------------------------------

    semantic_alpha = 0.40

    blended_rgb = rgb.copy()

    for class_id, semantic_colour in (
        CLASS_COLORS.items()
    ):

        mask = (
            semantic_class
            == class_id
        )

        if not np.any(mask):

            continue

        blended_rgb[mask] = (
            (
                1.0
                - semantic_alpha
            )
            * blended_rgb[mask]
            +
            semantic_alpha
            * semantic_colour
        )

    blended_rgb = np.clip(
        blended_rgb,
        0,
        255,
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Binary PLY structure.
    # --------------------------------------------------------

    vertex_dtype = np.dtype([

        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),

        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),

        ("semantic_class", "<i4"),

        ("semantic_confidence", "<f4"),

    ])

    vertices = np.empty(
        len(xyz),
        dtype=vertex_dtype,
    )

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    vertices["red"] = (
        blended_rgb[:, 0]
    )

    vertices["green"] = (
        blended_rgb[:, 1]
    )

    vertices["blue"] = (
        blended_rgb[:, 2]
    )

    vertices[
        "semantic_class"
    ] = semantic_class

    vertices[
        "semantic_confidence"
    ] = semantic_confidence

    # --------------------------------------------------------
    # Binary PLY header.
    # --------------------------------------------------------

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Scal3R complete point cloud "
        "with YOLO semantic highlighting\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int semantic_class\n"
        "property float semantic_confidence\n"
        "end_header\n"
    ).encode(
        "ascii"
    )

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "wb",
    ) as f:

        f.write(header)

        f.write(
            vertices.tobytes()
        )

    logger.info(
        "Saved semantic point cloud: %s",
        path,
    )

    logger.info(
        "Semantic PLY contains %d points.",
        len(vertices),
    )

    logger.info(
        "Semantic highlighting strength: %.2f",
        semantic_alpha,
    )


def write_semantic_class_file(
    path,
    model,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Semantic class mapping\n"
        )

        f.write(
            "# -1 = no YOLO segmentation\n"
        )

        f.write(
            "-1\tunlabelled\n"
        )

        if model is not None:

            for class_id, class_name in (
                sorted(
                    model.names.items()
                )
            ):

                f.write(
                    f"{int(class_id)}"
                    f"\t"
                    f"{class_name}\n"
                )

        else:

            for class_id, class_name in (
                sorted(
                    YOLO_CLASS_NAMES.items()
                )
            ):

                f.write(
                    f"{int(class_id)}"
                    f"\t"
                    f"{class_name}\n"
                )


# ============================================================
# POST PROCESSING
# ============================================================

def post_process(
    raw: dotdict,
    batches: list,
    indices: list,
    args,
    n_blocks_loop: int = 0,
    alignment: str = "sim3_wet",
    use_xyz_align: int = 1,
    recorder: StageRecorder | None = None,
    segmentation_model=None,
):

    i_src = []

    processed = dotdict(
        output=dotdict(
            c2w=[],
            ixt=[],
        )
    )

    if args.save_dpt:

        processed.output.dpt_map = []

    visualize = dotdict(
        block_xyz=[],
        block_rgb=[],
        block_msk=[],
        world_xyz=[],
        world_rgb=[],
        world_msk=[],
    )

    # --------------------------------------------------------
    # Semantic cloud accumulators.
    #
    # IMPORTANT:
    # These receive ONLY the same frames selected by
    # Scal3R's overlap-aware "midx" logic below.
    #
    # Therefore the semantic cloud does not independently
    # duplicate overlapping block frames.
    # --------------------------------------------------------

    semantic_world_xyz = []
    semantic_world_rgb = []
    semantic_world_class = []
    semantic_world_confidence = []

    # --------------------------------------------------------
    # Original timestamp-sorted image list.
    # --------------------------------------------------------

    image_paths = collect_image_paths(
        args.input_dir,
        args.image_patterns,
    )

    if args.max_images > 0:

        image_paths = image_paths[
            :args.max_images
        ]

    logger.info(
        "Semantic segmentation image list: "
        "%d images",
        len(image_paths),
    )

    if image_paths:

        logger.info(
            "First image: %s",
            image_paths[0],
        )

        logger.info(
            "Last image: %s",
            image_paths[-1],
        )

    batch0 = materialize_payload(
        batches[0]
    )

    height = (
        batch0.meta.H[0].item()
    )

    width = (
        batch0.meta.W[0].item()
    )

    del batch0

    logger.info(
        "Scal3R output resolution: "
        "%dx%d",
        width,
        height,
    )

    logger.info(
        "Using %s for pose graph optimization",
        alignment,
    )

    map_processor = build_map_processor(
        alignment,
        max_align_points_per_frame=(
            args.max_align_points_per_frame
        ),
    )

    if recorder is not None:

        recorder.record(
            "post_process.begin",
            n_blocks=int(
                len(batches)
            ),
            n_blocks_loop=int(
                n_blocks_loop
            ),
            alignment=alignment,
        )

    # --------------------------------------------------------
    # Prepare XYZ.
    # --------------------------------------------------------

    def prepare(
        batch,
        output,
    ):

        dpt = (
            output.dpt_map[0]
            .cpu()
        )

        cnf = (
            output.dpt_cnf[0]
            .cpu()
        )

        if use_xyz_align == 0:

            w2c, ixt = (
                decode_camera_params(
                    output.cam_map[0],
                    height,
                    width,
                    batch.meta.cam_param_type[0],
                    not batch.meta
                    .use_world_coord[0]
                    .item(),
                )
            )

            ray_o, ray_d = get_rays(
                height,
                width,
                ixt.cpu(),
                w2c[
                    ..., :3, :3
                ].cpu(),
                w2c[
                    ..., :3, 3:
                ].cpu(),
                z_depth=True,
                correct_pix=True,
            )

            ray_o = ray_o.reshape(
                w2c.shape[0],
                -1,
                3,
            )

            ray_d = ray_d.reshape(
                w2c.shape[0],
                -1,
                3,
            )

            xyz = (
                ray_o
                + ray_d * dpt
            )

        else:

            xyz = (
                output.xyz_map[0]
                .cpu()
            )

            cnf = (
                output.xyz_cnf[0]
                .cpu()
            )

        if "scale" in output:

            scale = (
                output.scale[0][
                    None,
                    None,
                    None,
                ].cpu()
            )

            xyz = (
                xyz * scale
            )

            dpt = (
                dpt * scale
            )

        return (
            xyz.numpy(),
            dpt.numpy(),
            cnf.numpy(),
        )

    norm_track = []
    loop_track = []

    n_blocks = (
        len(batches)
        if n_blocks_loop == 0
        else len(batches)
        - n_blocks_loop
    )

    # ========================================================
    # PHASE 1
    # ========================================================

    pbar = tqdm(
        total=n_blocks,
        desc="Preparing submaps",
    )

    for b in range(n_blocks):

        batch = materialize_payload(
            batches[b]
        )

        raw_block = materialize_payload(
            raw[b]
        )

        xyz, dpt, cnf = prepare(
            batch,
            raw_block,
        )

        map_processor.add_submap(
            xyz=xyz,
            dpt=dpt,
            cnf=cnf,
            msk=(
                batch.msk[0]
                .cpu()
                .numpy()
            ),
            file_name=(
                batch.src_inds[0]
                .cpu()
                .numpy()
                .tolist()
            ),
            compute_constraint=False,
        )

        del (
            batch,
            raw_block,
            xyz,
            dpt,
            cnf,
        )

        if (
            args.offload_batches
            or args.offload_outputs
        ):

            release_memory(
                args.device
            )

        pbar.update()

    pbar.close()

    # ========================================================
    # PHASE 2
    # ========================================================

    n_pairs = max(
        0,
        n_blocks - 1,
    )

    if args.pgo_workers > 0:

        n_workers = min(
            n_pairs,
            args.pgo_workers,
        )

    else:

        n_workers = min(
            n_pairs,
            os.cpu_count() or 4,
        )

    logger.info(
        "Aligning %d adjacent block pairs "
        "in parallel (%d workers)",
        n_pairs,
        n_workers,
    )

    norm_track = (
        map_processor
        .align_submaps_parallel(
            max_workers=n_workers
        )
    )

    # ========================================================
    # LOOP CLOSURE
    # ========================================================

    if n_blocks_loop > 0:

        pbar = tqdm(
            total=n_blocks_loop,
            desc="Processing loop closure blocks",
        )

        for k in range(
            n_blocks_loop
        ):

            block1 = (
                len(batches)
                - n_blocks_loop
                + k
            )

            block0, _, block2, _ = (
                indices[block1]
            )

            processor0 = (
                build_map_processor(
                    alignment,
                    max_align_points_per_frame=(
                        args.max_align_points_per_frame
                    ),
                )
            )

            processor2 = (
                build_map_processor(
                    alignment,
                    max_align_points_per_frame=(
                        args.max_align_points_per_frame
                    ),
                )
            )

            batch0 = materialize_payload(
                batches[block0]
            )

            batch1 = materialize_payload(
                batches[block1]
            )

            batch2 = materialize_payload(
                batches[block2]
            )

            raw0 = materialize_payload(
                raw[block0]
            )

            raw1 = materialize_payload(
                raw[block1]
            )

            raw2 = materialize_payload(
                raw[block2]
            )

            xyz0, dpt0, cnf0 = prepare(
                batch0,
                raw0,
            )

            xyz1, dpt1, cnf1 = prepare(
                batch1,
                raw1,
            )

            xyz2, dpt2, cnf2 = prepare(
                batch2,
                raw2,
            )

            processor0.add_submap(
                xyz=xyz0,
                dpt=dpt0,
                cnf=cnf0,
                msk=(
                    batch0.msk[0]
                    .cpu()
                    .numpy()
                ),
                file_name=(
                    batch0.src_inds[0]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
            )

            s0, R0, t0 = (
                processor0.add_submap(
                    xyz=xyz1,
                    dpt=dpt1,
                    cnf=cnf1,
                    msk=(
                        batch1.msk[0]
                        .cpu()
                        .numpy()
                    ),
                    file_name=(
                        batch1.src_inds[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    ),
                )
            )

            processor2.add_submap(
                xyz=xyz2,
                dpt=dpt2,
                cnf=cnf2,
                msk=(
                    batch2.msk[0]
                    .cpu()
                    .numpy()
                ),
                file_name=(
                    batch2.src_inds[0]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
            )

            s1, R1, t1 = (
                processor2.add_submap(
                    xyz=xyz1,
                    dpt=dpt1,
                    cnf=cnf1,
                    msk=(
                        batch1.msk[0]
                        .cpu()
                        .numpy()
                    ),
                    file_name=(
                        batch1.src_inds[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    ),
                )
            )

            s01, R01, t01 = (
                combine_transform(
                    s0,
                    R0,
                    t0,
                    s1,
                    R1,
                    t1,
                )
            )

            loop_track.append(
                (
                    block0,
                    block2,
                    (
                        s01,
                        R01,
                        t01,
                    ),
                )
            )

            del (
                batch0,
                batch1,
                batch2,
            )

            del (
                raw0,
                raw1,
                raw2,
            )

            del (
                xyz0,
                xyz1,
                xyz2,
                dpt0,
                dpt1,
                dpt2,
                cnf0,
                cnf1,
                cnf2,
            )

            del (
                processor0,
                processor2,
            )

            if (
                args.offload_batches
                or args.offload_outputs
            ):

                release_memory(
                    args.device
                )

            pbar.update()

        pbar.close()

        optimizer = (
            build_sim3_loop_optimizer()
        )

        ini_track = (
            optimizer
            .sequential_to_absolute_poses(
                norm_track
            )
        )

        opt_track = optimizer.optimize(
            norm_track,
            loop_track,
        )

        res_track = (
            accumulate_transform(
                opt_track
            )
        )

        vis_track = (
            optimizer
            .sequential_to_absolute_poses(
                opt_track
            )
        )

        visualize_loop(
            ini_track,
            vis_track,
            loop_track,
            get_runtime_root(args),
        )

        raw = raw[:n_blocks]
        batches = batches[:n_blocks]
        indices = indices[:n_blocks]

    else:

        res_track = [
            map_processor
            .optimizer
            .get_submap(i)
            .global_pose
            for i in range(
                len(batches)
            )
        ]

    del map_processor

    gc.collect()

    # ========================================================
    # OVERLAP HANDLING
    # ========================================================

    overlap_prev_inds = []
    overlap_curr_inds = []

    for b, batch_ref in enumerate(
        batches
    ):

        if (
            b
            == len(batches) - 1
        ):

            continue

        batch = materialize_payload(
            batch_ref
        )

        next_batch = (
            materialize_payload(
                batches[b + 1]
            )
        )

        prev_src_inds = (
            batch.src_inds[0]
            .tolist()
        )

        curr_src_inds = (
            next_batch.src_inds[0]
            .tolist()
        )

        overlap_prev_inds.append(
            torch.as_tensor(
                [
                    prev_src_inds.index(
                        idx
                    )
                    for idx
                    in batch
                    .orig_src_inds[0]
                    .tolist()
                    if idx
                    in curr_src_inds
                ],
                dtype=torch.long,
            )
            .numpy()
            .tolist()
        )

        overlap_curr_inds.append(
            torch.as_tensor(
                [
                    curr_src_inds.index(
                        idx
                    )
                    for idx
                    in batch
                    .orig_src_inds[0]
                    .tolist()
                    if idx
                    in curr_src_inds
                ],
                dtype=torch.long,
            )
            .numpy()
            .tolist()
        )

        del (
            batch,
            next_batch,
        )

    # ========================================================
    # FINAL ALIGNED RESULTS + YOLO
    # ========================================================

    pbar = tqdm(
        total=len(batches),
        desc="Getting aligned results",
    )

    for b, batch_ref in enumerate(
        batches
    ):

        batch = materialize_payload(
            batch_ref
        )

        raw_block = materialize_payload(
            raw[b]
        )

        S = (
            raw_block
            .cam_map[0]
            .shape[0]
        )

        j = b

        overlap_size = (
            batch.meta
            .overlap_size[0]
            .item()
        )

        w2c, ixt = (
            decode_camera_params(
                raw_block.cam_map[0],
                height,
                width,
                batch.meta
                .cam_param_type[0],
                not batch.meta
                .use_world_coord[0]
                .item(),
            )
        )

        if "scale" in raw_block:

            scale = (
                raw_block.scale[0][
                    None,
                    None,
                    None,
                ].cpu()
            )

            w2c[
                ..., :3, 3:
            ] = (
                w2c[
                    ..., :3, 3:
                ]
                * scale.to(
                    w2c.device
                )
            )

        inds = (
            torch.arange(S)
            .numpy()
            .tolist()
        )

        # ----------------------------------------------------
        # EXACT SCAL3R OVERLAP SELECTION
        # ----------------------------------------------------

        if (
            j % 2 == 0
            and j
            < len(batches) - 1
        ):

            midx = [
                k
                for k in inds
                if k not in
                overlap_prev_inds[
                    j
                ][
                    overlap_size // 2:
                ]
            ]

            if j > 0:

                midx = [
                    k
                    for k in midx
                    if k not in
                    overlap_curr_inds[
                        j - 1
                    ][
                        :overlap_size // 2
                    ]
                ]

        elif (
            j % 2 == 1
            and j > 0
        ):

            midx = [
                k
                for k in inds
                if k not in
                overlap_curr_inds[
                    j - 1
                ][
                    :overlap_size // 2
                ]
            ]

            if (
                j
                < len(batches) - 1
            ):

                midx = [
                    k
                    for k in midx
                    if k not in
                    overlap_prev_inds[
                        j
                    ][
                        overlap_size // 2:
                    ]
                ]

        elif (
            j % 2 == 0
            and j
            == len(batches) - 1
            and len(batches) > 1
        ):

            midx = [
                k
                for k in inds
                if k not in
                overlap_curr_inds[
                    j - 1
                ][
                    :overlap_size // 2
                ]
            ]

        else:

            midx = inds

        logger.info(
            "Block %d: keeping %d/%d frames "
            "after Scal3R overlap filtering.",
            j,
            len(midx),
            S,
        )

        i_src.extend(
            batch.src_inds[0][midx]
            .cpu()
            .numpy()
            .tolist()
        )

        processed.output.c2w.append(
            res_track[j][None]
            @ affine_inverse(
                affine_padding(
                    w2c[midx]
                )
            )
            .cpu()
            .numpy()
        )

        processed.output.ixt.append(
            ixt[midx]
            .cpu()
            .numpy()
        )

        if args.save_dpt:

            processed.output.dpt_map.append(
                raw_block
                .dpt_map[0][midx]
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # YOLO SEGMENTATION
        # ----------------------------------------------------

        segmentation_cache = None

        if segmentation_model is not None:

            block_frame_indices = [
                int(x)
                for x
                in batch.src_inds[0]
                .tolist()
            ]

            segmentation_cache = (
                run_yolo_on_block(
                    segmentation_model,
                    image_paths,
                    block_frame_indices,
                    height,
                    width,
                    args,
                )
            )

        # ----------------------------------------------------
        # POINT CLOUD
        # ----------------------------------------------------

        if args.save_xyz:

            block_xyz = []
            block_rgb = []
            block_msk = []

            xyz, _, cnf = prepare(
                batch,
                raw_block,
            )

            for k in range(
                raw_block
                .xyz_map[0]
                .shape[0]
            ):

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Skip frames that Scal3R did NOT select for
                # its final reconstruction.
                #
                # This is the actual duplicate-overlap fix.
                # ------------------------------------------------

                if k not in midx:

                    continue

                # ------------------------------------------------
                # XYZ
                # ------------------------------------------------

                xyz_map = xyz[k]

                xyz_map = np.concatenate(
                    [
                        xyz_map,
                        np.ones_like(
                            xyz_map[
                                ...,
                                :1
                            ]
                        ),
                    ],
                    axis=-1,
                )

                xyz_map = (
                    xyz_map
                    @ res_track[j][:3].T
                )

                # ------------------------------------------------
                # Scal3R confidence
                # ------------------------------------------------

                confidence_map = (
                    cnf[k][..., 0]
                )

                confidence_threshold = (
                    np.mean(
                        confidence_map
                    )
                    * args
                    .confidence_xyz_threshold
                )

                msk = (
                    confidence_map
                    > confidence_threshold
                )

                xyz_map = (
                    xyz_map[msk]
                )

                # ------------------------------------------------
                # RGB
                # ------------------------------------------------

                rgb_map = (
                    batch
                    .meta
                    .rgb[0][k]
                    .cpu()
                    .numpy()[msk]
                )

                # Scal3R RGB may be normalized.
                if (
                    rgb_map.size > 0
                    and rgb_map.max() <= 1.0
                ):

                    rgb_map = (
                        rgb_map
                        * 255.0
                    )

                rgb_map = np.clip(
                    rgb_map,
                    0,
                    255,
                ).astype(
                    np.uint8
                )

                # ------------------------------------------------
                # SEMANTIC ASSOCIATION
                # ------------------------------------------------

                if (
                    segmentation_model
                    is not None
                ):

                    frame_index = int(
                        batch
                        .src_inds[0][k]
                        .item()
                    )

                    seg = (
                        segmentation_cache[
                            frame_index
                        ]
                    )

                    class_map = (
                        seg[
                            "class_map"
                        ]
                    )

                    segmentation_confidence_map = (
                        seg[
                            "confidence_map"
                        ]
                    )

                    semantic_class_map = (
                        class_map.reshape(
                            -1
                        )
                    )

                    semantic_conf_map = (
                        segmentation_confidence_map
                        .reshape(
                            -1
                        )
                    )

                    semantic_class = (
                        semantic_class_map[
                            msk
                        ]
                    )

                    semantic_confidence = (
                        semantic_conf_map[
                            msk
                        ]
                    )

                else:

                    semantic_class = (
                        np.full(
                            len(xyz_map),
                            -1,
                            dtype=np.int32,
                        )
                    )

                    semantic_confidence = (
                        np.zeros(
                            len(xyz_map),
                            dtype=np.float32,
                        )
                    )

                # ------------------------------------------------
                # ADD TO SEMANTIC CLOUD
                #
                # Because k is restricted to midx above,
                # overlapping frames are NOT added again.
                # ------------------------------------------------

                semantic_world_xyz.append(
                    xyz_map[:, :3]
                    .astype(
                        np.float32
                    )
                )

                semantic_world_rgb.append(
                    rgb_map
                )

                semantic_world_class.append(
                    semantic_class
                    .astype(
                        np.int32
                    )
                )

                semantic_world_confidence.append(
                    semantic_confidence
                    .astype(
                        np.float32
                    )
                )

                # ------------------------------------------------
                # Existing Scal3R visualization
                # ------------------------------------------------

                visualize.world_xyz.append(
                    xyz_map
                )

                visualize.world_rgb.append(
                    rgb_map
                )

                visualize.world_msk.append(
                    msk.reshape(
                        height,
                        width,
                    )
                )

                block_xyz.append(
                    xyz_map
                )

                block_rgb.append(
                    rgb_map
                )

                block_msk.append(
                    msk.reshape(
                        height,
                        width,
                    )
                )

            visualize.block_xyz.append(
                block_xyz
            )

            visualize.block_rgb.append(
                block_rgb
            )

            visualize.block_msk.append(
                block_msk
            )

        del (
            batch,
            raw_block,
        )

        if (
            args.offload_batches
            or args.offload_outputs
        ):

            release_memory(
                args.device
            )

        pbar.update()

    pbar.close()

    # ========================================================
    # SORT FRAME OUTPUTS
    # ========================================================

    order = np.argsort(
        i_src,
        kind="stable",
    )

    for key in (
        processed.output.keys()
    ):

        processed.output[key] = (
            np.concatenate(
                processed.output[key],
                axis=0,
            )
        )

        processed.output[key] = (
            processed.output[key][
                order
            ]
        )

    # ========================================================
    # SAVE SEMANTIC POINT CLOUD
    # ========================================================

    if segmentation_model is not None:

        if semantic_world_xyz:

            semantic_xyz = (
                np.concatenate(
                    semantic_world_xyz,
                    axis=0,
                )
            )

            semantic_rgb = (
                np.concatenate(
                    semantic_world_rgb,
                    axis=0,
                )
            )

            semantic_class = (
                np.concatenate(
                    semantic_world_class,
                    axis=0,
                )
            )

            semantic_confidence = (
                np.concatenate(
                    semantic_world_confidence,
                    axis=0,
                )
            )

            semantic_ply_path = join(
                args.result_dir,
                "semantic_3d.ply",
            )

            write_semantic_ply(
                semantic_ply_path,
                semantic_xyz,
                semantic_rgb,
                semantic_class,
                semantic_confidence,
            )

            class_file_path = join(
                args.result_dir,
                "semantic_3d_classes.txt",
            )

            write_semantic_class_file(
                class_file_path,
                segmentation_model,
            )

            total_points = len(
                semantic_class
            )

            labelled_points = int(
                np.sum(
                    semantic_class >= 0
                )
            )

            unlabelled_points = (
                total_points
                - labelled_points
            )

            logger.info(
                "================================================"
            )

            logger.info(
                "SEMANTIC 3D POINT CLOUD"
            )

            logger.info(
                "================================================"
            )

            logger.info(
                "Total 3D points : %d",
                total_points,
            )

            logger.info(
                "YOLO labelled   : %d",
                labelled_points,
            )

            logger.info(
                "Unlabelled      : %d",
                unlabelled_points,
            )

            logger.info(
                "Semantic PLY    : %s",
                semantic_ply_path,
            )

            logger.info(
                "Class mapping   : %s",
                class_file_path,
            )

            logger.info(
                "================================================"
            )

            for class_id, class_name in sorted(
                segmentation_model.names.items()
            ):

                count = int(
                    np.sum(
                        semantic_class
                        == int(class_id)
                    )
                )

                if count > 0:

                    logger.info(
                        "  %d %-40s %d points",
                        int(class_id),
                        class_name,
                        count,
                    )

        else:

            logger.warning(
                "YOLO was enabled but no "
                "3D points were generated."
            )

    if recorder is not None:

        recorder.record(
            "post_process.done",
            n_frames=int(
                processed
                .output
                .c2w
                .shape[0]
            ),
            save_dpt=bool(
                args.save_dpt
            ),
            save_xyz=bool(
                args.save_xyz
            ),
            semantic_3d=bool(
                segmentation_model
                is not None
            ),
        )

        maybe_stop_after(
            "post_process.done",
            args,
            recorder,
        )

    return (
        processed,
        raw,
        batches,
        indices,
        visualize,
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SCAL3R standalone inference "
            "backend with optional YOLO26 "
            "semantic 3D point cloud"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/models/scal3r.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--runtime_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--image_patterns",
        type=str,
        default=(
            "*.png,*.jpg,*.jpeg,*.bmp"
        ),
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--overlap_size",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--loop_size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--loop_ckpt",
        type=str,
        default=(
            "data/checkpoints/"
            "dino_salad.ckpt"
        ),
    )

    parser.add_argument(
        "--use_xyz_align",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_align_points_per_frame",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pgo_workers",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--test_use_amp",
        action="store_true",
    )

    parser.add_argument(
        "--save_dpt",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--save_xyz",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--downsample_xyz_ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--confidence_xyz_threshold",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--use_loop",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--streaming_state",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--offload_batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--offload_outputs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--cleanup_offload",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--offload_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--probe_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--stop_after_stage",
        type=str,
        default="",
    )

    # ========================================================
    # YOLO
    # ========================================================

    parser.add_argument(
        "--segmentation_enable",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--segmentation_model",
        type=str,
        default=DEFAULT_YOLO_MODEL,
    )

    parser.add_argument(
        "--segmentation_confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--segmentation_iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--segmentation_imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--segmentation_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return dotdict(
        vars(
            parser.parse_args()
        )
    )


# ============================================================
# MAIN
# ============================================================

@log_exceptions(
    logger,
    "Unhandled exception in backend",
)
def main():

    args = parse_args()

    args.config = str(
        resolve_release_path(
            args.config
        )
    )

    args.result_dir = str(
        resolve_release_path(
            args.result_dir
            or get_default_output_dir(
                "run"
            )
        )
    )

    args.runtime_dir = str(
        resolve_release_path(
            args.runtime_dir
        )
        if args.runtime_dir
        else join(
            args.result_dir,
            "runtime",
        )
    )

    if args.checkpoint:

        args.checkpoint = str(
            resolve_release_path(
                args.checkpoint
            )
        )

    if args.loop_ckpt:

        args.loop_ckpt = str(
            resolve_release_path(
                args.loop_ckpt
            )
        )

    if args.offload_dir:

        args.offload_dir = str(
            resolve_release_path(
                args.offload_dir
            )
        )

    if args.probe_dir:

        args.probe_dir = str(
            resolve_release_path(
                args.probe_dir
            )
        )

    args.segmentation_model = (
        os.path.abspath(
            args.segmentation_model
        )
    )

    device = torch.device(
        args.device
    )

    recorder = StageRecorder(
        args.probe_dir or "",
        args.device,
    )

    log_block(
        "Runtime config: (",
        format_runtime_config(args),
        closing=")",
    )

    segmentation_model = None

    try:

        if recorder.enabled:

            recorder.record(
                "process.begin",
                input_dir=args.input_dir,
                result_dir=args.result_dir,
                runtime_dir=args.runtime_dir,
                checkpoint=args.checkpoint,
                device=str(device),
                segmentation_model=(
                    args.segmentation_model
                ),
            )

        # ----------------------------------------------------
        # Scal3R
        # ----------------------------------------------------

        if recorder is not None:

            recorder.record(
                "load_model.begin",
                config=args.config,
                checkpoint=(
                    args.checkpoint
                ),
            )

        sampler, dataset_cfg = (
            build_sampler_from_config(
                args.config,
                device,
                args.checkpoint,
            )
        )

        if recorder is not None:

            recorder.record(
                "load_model.done",
                checkpoint=(
                    args.checkpoint
                ),
                model_name=getattr(
                    sampler,
                    "__class__",
                    type(sampler),
                ).__name__,
            )

            maybe_stop_after(
                "load_model.done",
                args,
                recorder,
            )

        # ----------------------------------------------------
        # YOLO26
        # ----------------------------------------------------

        segmentation_model = (
            load_yolo_segmentation_model(
                args
            )
        )

        validate_yolo_classes(
            segmentation_model
        )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        batches, indices = load_data(
            dataset_cfg,
            args,
            recorder=recorder,
        )

        if recorder is not None:

            recorder.record(
                "load_data.done",
                n_blocks=int(
                    len(batches)
                ),
                n_blocks_loop=int(
                    args.get(
                        "n_blocks_loop",
                        0,
                    )
                ),
            )

            maybe_stop_after(
                "load_data.done",
                args,
                recorder,
            )

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats(
                device
            )

        t0 = wall_time.time()

        amp_enabled = bool(
            args.test_use_amp
            and device.type == "cuda"
        )

        amp_dtype = (
            torch.bfloat16
            if (
                device.type == "cuda"
                and torch.cuda
                .get_device_capability(
                    device
                )[0]
                >= 8
            )
            else torch.float16
        )

        with torch.no_grad():

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):

                output = forward(
                    sampler,
                    batches,
                    args,
                    recorder=recorder,
                )

        # ----------------------------------------------------
        # Post processing
        # ----------------------------------------------------

        (
            processed,
            output,
            batches,
            indices,
            visualize,
        ) = post_process(
            output,
            batches,
            indices,
            args,
            n_blocks_loop=args.get(
                "n_blocks_loop",
                0,
            ),
            alignment="sim3_wet",
            use_xyz_align=args.use_xyz_align,
            recorder=recorder,
            segmentation_model=(
                segmentation_model
            ),
        )

        t1 = wall_time.time()

        runtime = {

            "time":
                t1 - t0,

            "memory":
                (
                    torch.cuda
                    .max_memory_allocated(
                        device
                    )
                    / (1024 ** 3)
                    if device.type == "cuda"
                    else 0.0
                ),

            "n_frames":
                int(
                    processed
                    .output
                    .c2w
                    .shape[0]
                ),

            "segmentation":
                {
                    "enabled":
                        bool(
                            segmentation_model
                            is not None
                        ),

                    "model":
                        (
                            args.segmentation_model
                            if segmentation_model
                            is not None
                            else ""
                        ),

                    "confidence":
                        float(
                            args
                            .segmentation_confidence
                        ),

                    "iou":
                        float(
                            args
                            .segmentation_iou
                        ),
                },

            "offload":
                {
                    "states":
                        bool(
                            args.streaming_state
                        ),

                    "batches":
                        bool(
                            args.offload_batches
                        ),

                    "outputs":
                        bool(
                            args.offload_outputs
                        ),

                    "cleanup":
                        bool(
                            args.cleanup_offload
                        ),

                    "offload_dir":
                        (
                            get_offload_root(
                                args
                            )
                            if (
                                bool(
                                    args.streaming_state
                                )
                                or
                                bool(
                                    args.offload_batches
                                )
                                or
                                bool(
                                    args.offload_outputs
                                )
                            )
                            else ""
                        ),
                },

            "runtime_dir":
                get_runtime_root(
                    args
                ),

            "runtime_json":
                os.path.join(
                    get_runtime_root(
                        args
                    ),
                    "runtime.json",
                ),
        }

        runtime["fps"] = (
            runtime["n_frames"]
            / max(
                runtime["time"],
                1e-8,
            )
        )

        logger.info(
            "Inference finished, "
            "time cost: %.2fs, "
            "memory usage: %.2fGB, "
            "frames: %d",
            runtime["time"],
            runtime["memory"],
            runtime["n_frames"],
        )

        # ----------------------------------------------------
        # Normal Scal3R outputs
        # ----------------------------------------------------

        save_results(
            processed,
            batches,
            visualize,
            runtime,
            args,
            recorder=recorder,
        )

        logger.info(
            "Results saved to %s",
            args.result_dir,
        )

        logger.info(
            "Semantic 3D PLY saved to: %s",
            join(
                args.result_dir,
                "semantic_3d.ply",
            ),
        )

    except StopAfterStage as exc:

        logger.info(
            "Stopped after requested stage: %s",
            exc,
        )

    finally:

        cleanup_offload_root(
            args
        )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )