import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



def get_model_instance_segementation(num_classes):
    # 下载预训练模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获得模型 cls_score 的输出特征
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 调整模型
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获得模型 mask_predictor.conv5_mask 的输出特征
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    # 调整模型
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    for parameter in model.state_dict():
        print(parameter, model.state_dict()[parameter].size())

    print('finetune model......')
    finetune_model = get_model_instance_segementation(2)
    for parameter in finetune_model.state_dict():
        print(parameter, finetune_model.state_dict()[parameter].size())