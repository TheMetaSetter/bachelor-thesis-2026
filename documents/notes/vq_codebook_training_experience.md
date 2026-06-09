---
title: "Kinh nghiệm huấn luyện mô hình discrete codebook kiểu VQ-VAE"
author: "Tổng hợp từ các paper đầu ngành"
date: "2026-05-24"
lang: vi
header-includes:
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{booktabs}
  - \usepackage{ragged2e}
  - \usepackage{enumitem}
  - \setlist{nosep,leftmargin=*}
  - \renewcommand{\arraystretch}{1.18}
---

# Summary

Tài liệu này tổng hợp các paper nổi tiếng về Vector-Quantized VAE (VQ-VAE), VQGAN, codebook optimization, Finite Scalar Quantization (FSQ), và Residual Vector Quantization (RVQ). Trọng tâm không phải chỉ là liệt kê paper, mà là rút ra các training setup và kinh nghiệm thực tiễn có thể dùng ngay khi huấn luyện mô hình có discrete codebook.

Kết luận chính: learning rate, optimizer, batch size và số epoch quan trọng, nhưng với VQ-style models, yếu tố quyết định thường là codebook update rule, codebook utilization, initialization/reset, warmup, activation probability của code entries, và cách xử lý codebook collapse.

# Bảng tổng hợp paper, setup, kết quả và kinh nghiệm thực tiễn

\scriptsize
\begin{longtable}{p{0.15\textwidth} p{0.19\textwidth} p{0.21\textwidth} p{0.19\textwidth} p{0.18\textwidth}}
\toprule
\textbf{Paper} & \textbf{Venue / nguồn} & \textbf{Training setup được báo cáo} & \textbf{Experimental results liên quan} & \textbf{Đúc kết thực tiễn} \\
\midrule
\endfirsthead
\toprule
\textbf{Paper} & \textbf{Venue / nguồn} & \textbf{Training setup được báo cáo} & \textbf{Experimental results liên quan} & \textbf{Đúc kết thực tiễn} \\
\midrule
\endhead
\bottomrule
\endfoot

VQ-VAE: \textit{Neural Discrete Representation Learning} & NeurIPS 2017 & CIFAR-10: Adam, learning rate 2e-4, batch size 128, 250,000 steps. Encoder/decoder dùng strided convolution và residual blocks. & CIFAR-10 đạt 4.67 bits/dim. Mô hình tránh posterior collapse tốt hơn khi dùng discrete latent representation. & Baseline nhỏ hợp lý: Adam 2e-4, batch 128, train dài theo step. Dùng tốt cho reconstruction đơn giản; chưa đủ cho image tokenizer high-resolution. \\
\midrule
VQ-VAE-2: \textit{Generating Diverse High-Fidelity Images with VQ-VAE-2} & NeurIPS 2019 & Train hai giai đoạn: trước train hierarchical VQ-VAE, sau đó train PixelCNN prior trên latent codes. Codebook dùng EMA update, decay thường được báo là 0.99. LR/batch/epoch không được báo rõ trong phần chính. & Dùng latent hierarchy: bottom 64x64 và top 32x32 cho image 256x256. FID cải thiện khi thêm rejection sampling. & Đây là architecture recipe hơn là hyperparameter recipe. Điểm chính: hierarchical codes, EMA codebook, và train prior riêng. \\
\midrule
VQGAN: \textit{Taming Transformers for High-Resolution Image Synthesis} & CVPR 2021 & Public ImageNet 256 config: base learning rate 4.5e-6, batch size 12, embed dim 256, n\_embed 1024, disc\_start 250001, disc\_weight 0.8, codebook\_weight 1.0. & Học codebook của visual constituents rồi dùng transformer trên token sequence. Trở thành nền tảng cho discrete image tokenizer kiểu VQGAN. & Với image tokenizer, không nên chỉ dùng MSE. Dùng perceptual loss + adversarial loss, nhưng bật discriminator muộn để tránh làm autoencoder/codebook mất ổn định. \\
\midrule
ViT-VQGAN / Improved VQGAN: \textit{Vector-Quantized Image Modeling with Improved VQGAN} & ICLR 2022 & Batch size 256 trên 128 TPUv4 cores, input 256x256. Tokenizer train 500,000 steps; Adam beta1=0.9, beta2=0.96; warmup lên 1e-4 rồi inverse square-root decay. & ImageNet 256: Improved VQGAN báo IS 175.1, FID 4.17; tốt hơn vanilla VQGAN IS 70.6, FID 17.04. & Nếu làm tokenizer ảnh hiện đại, cải thiện tokenizer architecture/codebook learning thường quan trọng hơn chỉ tăng prior. Batch lớn + warmup + decay là recipe mạnh; scale xuống GPU nhỏ cần giảm LR hoặc dùng gradient accumulation. \\
\midrule
\textit{Straightening Out the Straight-Through Estimator} & ICML 2023 & Generative autoencoder: AdamW, LR 1e-4, weight decay 1e-4, beta=(0.9,0.95), 90 epochs, cosine LR decay, 10 warmup epochs, không augmentation. ViT-Tiny classification: AdamW LR 2e-4, wd 0.03, 90 epochs, 10 warmup epochs. & Phân tích mismatch giữa encoder embeddings và code vectors, sparse codebook gradients, asymmetric commitment loss, và index collapse. & Khi codebook chết, không nên chỉ chỉnh LR. Cần đo activation ratio/code usage. Dùng warmup, affine re-parameterization, initialization tốt hơn, và batch đủ lớn để nhiều code có cơ hội được update. \\
\midrule
SQ-VAE: \textit{Self-Annealed Stochastic Quantization} & ICML 2022 & Dùng stochastic quantization và self-annealing. Setup chi tiết phụ thuộc dataset; các phân tích sau cho thấy temperature annealing và loss scaling có ảnh hưởng lớn. & Thiết kế để cải thiện codebook utilization và giảm các heuristic như reset/splitting. & Nếu dùng stochastic quantization, anneal chậm. Đừng giảm temperature quá nhanh; theo dõi usage/entropy cùng reconstruction loss. \\
\midrule
CVQ-VAE: \textit{Online Clustered Codebook} & ICCV 2023 & Small datasets: official VQ-VAE setup, batch size 1024 trên 4x GTX TITAN X, 500 epochs. High-res FFHQ/ImageNet: thay quantizer trong VQGAN, batch size 64 trên 4x RTX A6000; FFHQ train 4 ngày, ImageNet 8 ngày. & Cải thiện rFID/LPIPS so với VQ-VAE/SQ-VAE trên nhiều dataset bằng cách xử lý dead codevectors qua online clustered codebook. & Nếu dùng VQ-VAE/VQGAN cũ và nhiều code chết, CVQ là hướng thay quantizer khá trực tiếp. Batch lớn giúp code activation tốt hơn, nhất là small datasets. \\
\midrule
Rotation Trick: \textit{Restructuring Vector Quantization with the Rotation Trick} & ICLR 2025 & Drop-in replacement cho straight-through gradient. Paper thử nhiều training paradigms nên không có một LR/batch/epoch chung để copy. & Truyền thông tin về góc và độ lớn giữa encoder output và codebook vector vào gradient, thay vì cho gradient đi quanh quantization layer như STE. & Nên thử khi quantization error cao hoặc codebook utilization thấp nhưng không muốn đổi toàn bộ architecture. So sánh trực tiếp với STE trong cùng setup. \\
\midrule
FSQ: \textit{Finite Scalar Quantization: VQ-VAE Made Simple} & ICLR 2024 & MaskGIT ImageNet 128 sweep: Stage I 100 epochs, Stage II 200 epochs. ImageNet 256 setup: Stage I 1M steps, batch 512; Stage II 2.5M steps, batch 256. FSQ dùng levels như [8,5,5,5] cho codebook xấp xỉ 2\textasciicircum{}10. & Đạt 100\% code usage trong MaskGIT setup, FID gần tương đương VQ baseline; không cần commitment loss, reseeding, code splitting, entropy penalty. & Nếu không bắt buộc phải dùng learned vector codebook, FSQ là baseline nghiêm túc. Mỗi scalar level nên khoảng >=5; quá ít level làm performance giảm. \\
\midrule
VQGAN-LC: \textit{Scaling the Codebook Size of VQGAN to 100,000} & NeurIPS 2024 & ImageNet-1K và FFHQ dùng 32 V100 GPUs. ImageNet: 20 epochs; FFHQ: 800 epochs. Adam LR 5e-4, half-cycle cosine decay sau 5-epoch linear warmup. Codebook 100k init từ CLIP ViT-L/14 patch features; projected code dimension mặc định 8. & ImageNet reconstruction với 100k codebook đạt >99\% utilization; với 1024 tokens, rFID 1.29, LPIPS 0.07, PSNR 27.0. & Với codebook rất lớn, không nên train từng codevector kiểu vanilla. Dùng pretrained feature codebook + projector + low-dimensional lookup. \\
\midrule
SimVQ: \textit{Addressing Representation Collapse in Vector Quantized Models with One Linear Layer} & ICCV 2025 & Vision: ImageNet 50 epochs, batch size 256, input 128x128, downsample factor 8 nên token map 16x16. Audio: LibriTTS-580h, 50 epochs, batch size 64, 1-second window. Optimizer/LR không báo rõ trong phần truy được. & ImageNet 128: SimVQ với 65,536 hoặc 262,144 codes đạt 100\% utilization; rFID lần lượt 2.24 và 1.99. & Nếu representation collapse do codebook được tối ưu rời rạc từng entry, reparameterize codebook bằng linear layer/latent basis để cập nhật có cấu trúc hơn. \\
\midrule
SoundStream: \textit{An End-to-End Neural Audio Codec} & IEEE/ACM TASLP 2021 & Fully convolutional encoder/decoder + residual vector quantizer, train end-to-end với reconstruction và adversarial losses. LR/batch/epoch không thấy đủ rõ trong phần truy được. & Nền tảng neural audio codec dùng RVQ, hướng tới chất lượng cao ở nhiều bitrate và real-time. & Dùng làm architecture reference cho audio RVQ: encoder -> RVQ stack -> decoder, loss gồm reconstruction + adversarial. Muốn copy hyperparameter thì EnCodec rõ hơn. \\
\midrule
EnCodec: \textit{High Fidelity Neural Audio Compression} & TMLR 2023 & Train 300 epochs, mỗi epoch 2000 updates. Adam, batch size 64 mẫu audio 1 giây, LR 3e-4, beta1=0.5, beta2=0.9, 8x A100 GPUs. Loss balancer: ví dụ 24kHz dùng lambda\_t=0.1, lambda\_f=1, lambda\_g=3, lambda\_feat=3. & EnCodec 3 kbps vượt Lyra-v2 6 kbps và Opus 12 kbps trong MUSHRA; Gumbel-Softmax/DiffQ không được báo sâu vì không tốt hơn. & Recipe audio RVQ rõ nhất: Adam 3e-4, betas 0.5/0.9, batch 64x1s, train dài, dùng adversarial + spectral/reconstruction loss. \\
\midrule
ERVQ: \textit{Enhanced Residual Vector Quantization} & IEEE/ACM TASLP / arXiv 2025 & Thêm intra-codebook optimization bằng online clustering + code balancing loss, và inter-codebook optimization để giảm similarity giữa residual quantizers. LR/batch/epoch phụ thuộc base codec, không có một recipe độc lập rõ ràng. & Nhắm trực tiếp vào codebook collapse trong RVQ; báo có thể đạt 100\% utilization trong một advanced codec và không tăng chi phí inference vì chỉ can thiệp khi training. & Với audio RVQ nhiều codebook nối tiếp, cần kiểm tra cả utilization từng codebook và redundancy giữa residual quantizers. Thêm balancing loss + diversity giữa codebooks là hướng hợp lý. \\

\end{longtable}
\normalsize

# Đúc kết thực tiễn theo tình huống

## Khi làm VQ-VAE reconstruction nhỏ

Bắt đầu với Adam hoặc AdamW quanh 1e-4 đến 2e-4, batch 128-256, và train dài theo step hoặc khoảng 90 epochs có warmup/cosine. VQ-VAE gốc dùng Adam 2e-4, batch 128, 250k steps; Huh et al. dùng AdamW 1e-4, 90 epochs, 10 warmup epochs cho generative autoencoder.

Điểm cần theo dõi không chỉ là reconstruction loss. Cần log codebook utilization, số unique codes được dùng, entropy của code distribution, và quantization error. Nếu reconstruction loss giảm nhưng code usage thấp, mô hình có thể đang học bằng vài code chính và bỏ chết phần lớn codebook.

## Khi làm image tokenizer kiểu VQGAN

Công thức thực dụng là reconstruction loss + perceptual loss + adversarial loss. Tuy nhiên, discriminator không nên bật quá sớm. Public config của VQGAN dùng disc\_start rất muộn, khoảng 250,001 steps, để autoencoder/codebook ổn định trước khi adversarial signal tác động mạnh.

Nếu dùng setup hiện đại hơn như Improved VQGAN hoặc VQGAN-LC, nên dùng warmup và scheduler rõ ràng. Batch lớn giúp codebook có nhiều cơ hội được activate; nếu GPU ít, dùng gradient accumulation thay vì giảm batch quá sâu.

## Khi gặp codebook collapse

Không nên phản xạ đầu tiên là tăng learning rate. Collapse có thể đến từ mismatch giữa encoder output và code vectors, sparse codebook gradients, initialization kém, batch quá nhỏ, codebook quá lớn, hoặc commitment loss không cân bằng.

Các hướng xử lý đáng thử theo thứ tự thực dụng: tăng batch/effective batch, dùng warmup, kiểm tra K-means hoặc pretrained initialization, dùng EMA/reset cho dead codes, thay quantizer bằng CVQ/SimVQ, hoặc thử Rotation Trick nếu muốn giữ architecture gần như cũ.

## Khi cần codebook rất lớn

Với vocabulary 50k-100k, vanilla trainable codebook thường không ổn định. VQGAN-LC gợi ý dùng pretrained visual features để khởi tạo codebook lớn, sau đó train projector hoặc low-dimensional lookup thay vì tối ưu từng codevector trực tiếp. SimVQ gợi ý reparameterize codebook bằng latent basis/linear layer để giảm disjoint codebook optimization.

## Khi không bắt buộc phải dùng learned vector codebook

FSQ nên được xem là baseline nghiêm túc. Nó bỏ learned vector codebook và thay bằng scalar quantization hữu hạn, nhờ đó tránh nhiều heuristic như commitment loss, reseeding, code splitting, entropy penalty. Nếu mục tiêu chính là tokenizer ổn định, FSQ có thể là lựa chọn tốt hơn VQ truyền thống.

## Khi làm audio tokenizer hoặc neural audio codec

EnCodec là recipe rõ nhất: Adam 3e-4, beta=(0.5,0.9), batch 64 đoạn audio 1 giây, train 300 epochs, dùng loss balancer cho reconstruction, spectral, adversarial và feature loss. SoundStream nên dùng làm architecture reference, còn ERVQ nên đọc khi gặp collapse hoặc redundancy trong nhiều residual codebooks.

# Ghi chú về warmup của Huh et al.

Huh et al. báo base learning rate là 1e-4, dùng AdamW, cosine learning rate decay, và 10 warmup epochs. Paper không ghi rõ warmup bắt đầu chính xác từ 0, 1e-6, hay 1e-5; cũng không nói rõ update learning rate theo epoch hay theo optimizer step.

Cách dùng 1e-5, 2e-5, ..., 1e-4 cho 10 epoch đầu là một implementation hợp lý nếu update learning rate mỗi epoch. Tuy nhiên, trong deep learning hiện đại, linear warmup thường được cập nhật theo optimizer step. Khi đó learning rate tăng mượt hơn từ gần 0 lên 1e-4 trong toàn bộ số batch của 10 epoch đầu, rồi sau đó dùng cosine decay.

Pseudo-code thực dụng:

```python
base_lr = 1e-4
warmup_steps = steps_per_epoch * 10
total_steps = steps_per_epoch * 90

if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = 0.5 * base_lr * (1 + cos(pi * progress))
```

# Checklist dùng ngay khi train VQ-style model

- Log reconstruction loss, perceptual loss, adversarial loss nếu có, quantization error, commitment loss, codebook utilization, unique code count, và entropy của code distribution.
- Nếu codebook usage thấp: thử tăng effective batch size, warmup dài hơn, initialization tốt hơn, EMA/reset, hoặc quantizer thay thế như CVQ/SimVQ/FSQ.
- Nếu loss dao động mạnh lúc đầu: giảm LR, dùng linear warmup theo step, hoặc delay discriminator.
- Nếu codebook lớn hơn 8k-16k: cân nhắc pretrained/frozen codebook, projector, low-dimensional lookup, hoặc reparameterized codebook.
- Nếu làm image tokenizer: đánh giá bằng rFID, LPIPS, PSNR và reconstruction samples; không chỉ nhìn MSE.
- Nếu làm audio RVQ: đánh giá cả subjective/perceptual quality, bitrate, codebook usage từng RVQ layer, và redundancy giữa các residual codebooks.

# References

1. van den Oord, A., Vinyals, O., and Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. NeurIPS. https://proceedings.neurips.cc/paper/7210-neural-discrete-representation-learning.pdf
2. Razavi, A., van den Oord, A., and Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. NeurIPS. https://papers.neurips.cc/paper/9625-generating-diverse-high-fidelity-images-with-vq-vae-2.pdf
3. Esser, P., Rombach, R., and Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. CVPR. https://arxiv.org/abs/2012.09841
4. CompVis VQGAN ImageNet config. https://github.com/CompVis/taming-transformers/blob/master/configs/imagenet_vqgan.yaml
5. Yu, J. et al. (2022). Vector-Quantized Image Modeling with Improved VQGAN. ICLR. https://openreview.net/forum?id=pfNyExj7z2
6. Huh, M., Cheung, B., Agrawal, P., and Isola, P. (2023). Straightening Out the Straight-Through Estimator. ICML. https://proceedings.mlr.press/v202/huh23a/huh23a.pdf
7. Takida, Y. et al. (2022). SQ-VAE: Variational Bayes on Discrete Representation with Self-Annealed Stochastic Quantization. ICML. https://arxiv.org/abs/2205.07547
8. Zheng, Y. and Vedaldi, A. (2023). Online Clustered Codebook. ICCV. https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Online_Clustered_Codebook_ICCV_2023_paper.pdf
9. Zheng, Y. and Vedaldi, A. (2023). Online Clustered Codebook Supplementary Material. https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zheng_Online_Clustered_Codebook_ICCV_2023_supplemental.pdf
10. Fifty, C. et al. (2025). Restructuring Vector Quantization with the Rotation Trick. ICLR. https://openreview.net/forum?id=GMwRl2e9Y1
11. Mentzer, F. et al. (2024). Finite Scalar Quantization: VQ-VAE Made Simple. ICLR. https://openreview.net/forum?id=8ishA3LxN8
12. Zhu, L. et al. (2024). Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99%. NeurIPS. https://papers.neurips.cc/paper_files/paper/2024/file/1716d022edeac750e57a2986a7135e13-Paper-Conference.pdf
13. Zhu, Y. et al. (2025). Addressing Representation Collapse in Vector Quantized Models with One Linear Layer. ICCV. https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_Addressing_Representation_Collapse_in_Vector_Quantized_Models_with_One_Linear_ICCV_2025_paper.pdf
14. Zeghidour, N. et al. (2021). SoundStream: An End-to-End Neural Audio Codec. IEEE/ACM TASLP. https://arxiv.org/pdf/2107.03312
15. Défossez, A. et al. (2023). High Fidelity Neural Audio Compression. TMLR. https://openreview.net/pdf?id=ivCd8z8zR2
16. Zheng, H. et al. (2025). ERVQ: Enhanced Residual Vector Quantization with Intra-and-Inter-Codebook Optimization for Neural Audio Codecs. https://arxiv.org/html/2410.12359v2

# Reliability note

Các hyperparameter không được paper/config báo rõ được ghi là "không báo rõ" hoặc "không có recipe độc lập rõ ràng". Không suy đoán LR/batch/epoch khi nguồn không đủ chắc.
