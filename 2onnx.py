import torch
import timm


def main():
    model = timm.create_model(
        model_name="shufflenet",
        num_classes=5,
        pretrained=False,
        checkpoint_path=r"D:\myfiles\school\Autodrive\traffic_light_classifier\AutoDrive\tools\best_model_weights_V5.pth"
    )

    input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        input,
        "8cls_path_sfnet.onnx",
        verbose=False,
        opset_version=12,
        export_params=True
    )


if __name__ == '__main__':
    main()
