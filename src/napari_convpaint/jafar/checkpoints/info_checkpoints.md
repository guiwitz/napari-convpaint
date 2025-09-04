The folder /jafar/checkpoints/ should contain:
- DINOv2 backbone (~80MB): `dinov2_vits14_reg4_pretrain.pth`
- JAFAR upscaler matching the DINOv2 backbone (~7MB): `vit_small_patch14_reg4_dinov2.pth`
We choose the smallest DINOv2 (vits/small), WITH registers (reg), patch size 14, trained on default dataset (NOT lvd142).

Download link upscaler:
https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_reg4_dinov2.pth

Downlaod link backbone:
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth