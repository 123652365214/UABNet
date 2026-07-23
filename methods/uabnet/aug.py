import torch
import torch.nn.functional as F
import numpy as np
import cv2


class CamouflageCropper:
    def __init__(self, output_size=384, scale_factor=1.5):
        """
        初始化裁剪器
        :param output_size: 输出尺寸（默认384）
        :param scale_factor: 初始放大因子（默认1.5）
        """
        self.output_size = output_size
        self.scale_factor = scale_factor

    def __call__(self, color_tensor, gt_tensor):
        """
        处理批量伪装物体Tensor
        :param color_tensor: 彩色图像Tensor (4,3,384,384)
        :param gt_tensor: GT Tensor (4,1,384,384)
        :return: 裁剪放大后的Tensor (4,3,384,384)
        """
        batch_size = color_tensor.shape[0]
        results = []
        bboxs = []

        for i in range(batch_size):
            # 处理单个样本
            color_img = color_tensor[i].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
            gt_img = gt_tensor[i].squeeze().detach().cpu().numpy()   # (H,W)

            # 处理并裁剪
            cropped = self._process_single_image(color_img, gt_img)
            # bboxs.append(bboxs1)
            # bboxs.append(bboxs2)
            # bboxs.append(bboxs3)
            # bboxs.append(bboxs4)
            # 转为Tensor并调整维度
            cropped_tensor = torch.from_numpy(cropped).permute(2, 0, 1)  # (3,H,W)
            results.append(cropped_tensor)

        # 堆叠结果并确保尺寸一致
        result_tensor = torch.stack(results)
        if result_tensor.shape[-2:] != (self.output_size, self.output_size):
            result_tensor = F.interpolate(result_tensor, size=(self.output_size, self.output_size),
                                          mode='bilinear', align_corners=False)

        return result_tensor.to(color_tensor.device)

    def _process_single_image(self, color_img, gt_img):
        """
        处理单张图像
        :param color_img: (H,W,3) numpy数组
        :param gt_img: (H,W) numpy数组
        :return: 裁剪放大后的图像 (H,W,3)
        """
        # 二值化GT图像
        gt_img = (gt_img * 255).astype(np.uint8)  # 假设输入是0-1浮点
        _, binary_gt = cv2.threshold(gt_img, 0.5, 1, cv2.THRESH_BINARY)

        # 寻找轮廓
        contours, _ = cv2.findContours(
            binary_gt.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # 如果没有找到轮廓，返回原图
            return color_img

        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 计算放大后的区域
        center_x, center_y = x + w // 2, y + h // 2
        new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)

        # 确保不超出图像边界
        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(color_img.shape[1], center_x + new_w // 2)
        y2 = min(color_img.shape[0], center_y + new_h // 2)

        # 裁剪图像
        cropped = color_img[y1:y2, x1:x2]

        # 上采样到目标尺寸
        if cropped.shape[0] != self.output_size or cropped.shape[1] != self.output_size:
            cropped = cv2.resize(cropped, (self.output_size, self.output_size),
                                 interpolation=cv2.INTER_LINEAR)

        return cropped


def map_to_original_space(pred, original_img, bboxes):
    """
    pred: 模型在裁剪图像上的预测 (N,C,H,W)
    original_img: 原始测试图像 (N,C,H,W)
    bboxes: 裁剪时的边界框坐标列表 [ (x1,y1,x2,y2), ... ]
    """
    full_res_preds = []
    for i in range(pred.shape[0]):
        x1, y1, x2, y2 = bboxes[i]
        # 将预测插值回原始裁剪区域大小
        region_pred = F.interpolate(pred[i:i + 1], size=(y2 - y1, x2 - x1))

        # 创建全图尺寸的空预测
        full_pred = torch.zeros_like(original_img[i:i + 1, :1])

        # 将区域预测放回原位
        full_pred[..., y1:y2, x1:x2] = region_pred
        full_res_preds.append(full_pred)

    return torch.cat(full_res_preds, dim=0)

# 使用示例
if __name__ == "__main__":
    # 模拟输入数据 (4,3,384,384) 和 (4,1,384,384)
    batch_size = 4
    color_tensor = torch.rand(batch_size, 3, 384, 384)
    gt_tensor = torch.rand(batch_size, 1, 384, 384)

    # 创建处理实例
    cropper = CamouflageCropper(output_size=384, scale_factor=1.5)

    # 处理数据
    output_tensor = cropper(color_tensor, gt_tensor)

    print(f"输入尺寸: {color_tensor.shape}")
    print(f"输出尺寸: {output_tensor.shape}")
