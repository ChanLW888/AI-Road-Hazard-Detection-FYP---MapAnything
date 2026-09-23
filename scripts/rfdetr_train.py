from rfdetr import RFDETRSegSmall


def main():
    model = RFDETRSegSmall(
        gradient_checkpointing=True
    )

    model.train(
        dataset_dir=r"/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/2D-Seg-Road-Nissan.v13i.coco-segmentation",

        epochs=100,

        batch_size=8,
        grad_accum_steps=8,

        lr=5e-5,

        output_dir=r"/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/2D-Seg-Road-Nissan.v13i.coco-segmentation/outputs",

        num_workers=10,

        early_stopping=True,
        early_stopping_patience=10,
    )


if __name__ == "__main__":
    main()