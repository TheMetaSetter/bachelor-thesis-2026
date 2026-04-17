import torch
import torch.nn as nn
import torch.nn.functional as F


from .cnn import ConvEncoder, ConvDecoder
from .classifier import NonLinClassifier


#########################################################################################################
# META CLASS
#########################################################################################################
class MetaAEC(nn.Module):
    def __init__(self, params):
        super(MetaAEC, self).__init__()

        self.encoder = None
        self.decoder = None
        self.classifier = None

        self.name = params.name

        self.classes = params.classes
        self.c_loss_ratio = params.c_loss_ratio  # 0.5

        self.apply_anomaly_mask = params.apply_anomaly_mask
        self.label_smoothing = params.label_smoothing
        self.alpha = params.alpha
        self.beta = params.beta

    def forward(self, x):
        # print('meta x [batch, window, in_feature]', x.shape)
        x_enc = self.encoder(x)
        # print('meta x_enc [batch, embedding, 1]', x_enc.shape)
        x_hat = self.decoder(x_enc)
        # print('meta x_hat [batch, window, in_feature]', x_hat.shape)
        x_out = self.classifier(x_enc.reshape(x_enc.size(0), -1))
        # print('meta x_out [batch, classes]', x_out.shape)
        return x_hat, x_out, x_enc

    def calculate_loss(self, inputs, predicted, label, pred_label, anomaly_mask, epoch):
        loss_AE = nn.MSELoss()
        loss_C = nn.CrossEntropyLoss(reduction="none")

        # Nếu cần apply anomaly mask để che đi những cell bất thường trong mỗi sub-sequence
        # Thường chỉ dùng trong training
        if self.apply_anomaly_mask:
            inputs = inputs * anomaly_mask
            predicted = predicted * anomaly_mask

        # Reconstructed loss được tính dựa trên synthetic (hay injected) input sub-sequence và reconstructed sub-sequence
        loss_ae = loss_AE(inputs, predicted)

        # Nếu cần apply label smoothing (hay label refurbishment) để tránh mô hình quá tự tin (hiện tượng over-confidence)
        # Thường chỉ dùng trong training
        if self.label_smoothing:
            # Vị trí của normal class trên one-hot vector là index 0
            normal_loc = 0

            # self.alpha là xác suất mà synthetic sub-sequence (thực ra là bất thường nhưng lại đang) bị gán nhãn nhầm thành bình thường (trong `label`).
            # self.beta là xác suất mà synthetic sub-sequence (thực ra là bất thường này nhưng lại đang) bị gán nhãn nhầm thành bất thường khác (trong `label`).
            label = (
                # 1 là tổng xác suất
                # self.beta * self.classes là tổng xác suất 1 anomaly class bị gán nhầm thành anomaly class kia (kể cả nhầm thành chính nó!)
                # self.alpha - self.beta * self.classes - self.beta là xác suất 1 anomaly class bị gán nhầm thành anomaly class khác nó.
                label * (1 - (self.alpha + self.beta * self.classes - self.beta))
                + (1 - label) * self.beta
            )
            # Các anomaly class mà có cơ hội là class thật sự thì sẽ được tăng thêm một lượng `self.beta`` trong `label` thay vì để là 0.

            # Class normal mà có cơ hội là class thật sự thì cũng sẽ được tăng thêm một lượng `self.alpha` trong `label` thay vì để là 0.
            # Do luôn phải "chia sẻ" cho các class khác một lượng lớn hơn alpha mà chỉ nhận được tối đa `alpha`, class normal không bao giờ bị lố quá 1.
            label[:, normal_loc] += self.alpha

        # Classification loss sẽ là Cross Entropy loss giữa output softmax của mô hình và `label` (có thể đã được làm mượt nếu đang trong giai đoạn training)
        loss_c = loss_C(pred_label, label)

        # Mỗi synthetic sub-sequence trong batch sẽ có một con số Cross Entropy loss
        # Tính trung bình để trả ra Cross Entropy loss chung cho toàn batch
        loss_c = torch.mean(loss_c)

        return (
            # self.c_loss_ratio là mức độ quan trọng của classification loss trong loss tổng
            (1 - self.c_loss_ratio) * loss_ae + self.c_loss_ratio * loss_c,
            loss_ae,
            loss_c,
        )

    def calculate_loss_residual(
        self, residual, predicted, label, pred_label, anomaly_mask, epoch
    ):
        loss_AE = nn.MSELoss()
        loss_C = nn.CrossEntropyLoss(reduction="none")

        loss_ae = loss_AE(residual, predicted)

        if self.label_smoothing:
            normal_loc = 0
            label = (
                label * (1 - self.alpha - self.beta * self.classes + self.beta)
                + (1 - label) * self.beta
            )
            label[:, normal_loc] += self.alpha

        loss_c = loss_C(pred_label, label)
        loss_c = torch.mean(loss_c)
        # print('loss_c',loss_c.shape, loss_c)
        # print('loss_ae', loss_ae, 'loss_c', loss_c)
        return (
            (1 - self.c_loss_ratio) * loss_ae + self.c_loss_ratio * loss_c,
            loss_ae,
            loss_c,
        )


class ConvAEC(MetaAEC):
    def __init__(self, params):
        super(ConvAEC, self).__init__(params=params)

        # x: (batch, n_time, n_features)

        num_inputs = params.n_features
        seq_len = params.n_time
        classes = params.classes

        num_filters = params.num_filters
        embedding_dim = params.embedding_dim
        kernel_size = params.kernel_size
        dropout = params.dropout
        normalization = params.normalization
        stride = params.stride
        padding = params.padding
        classifier_dim = params.classifier_dim  # 32

        self.encoder = ConvEncoder(
            num_inputs,
            num_filters,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            normalization=normalization,
        )
        self.decoder = ConvDecoder(
            embedding_dim,
            num_filters,
            seq_len,
            num_inputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            normalization=normalization,
        )
        self.classifier = NonLinClassifier(
            embedding_dim,
            classes,
            d_hidd=classifier_dim,
            dropout=dropout,
            norm=normalization,
        )
