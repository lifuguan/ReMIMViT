
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python lazyconfig_train_net.py --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) --config-file configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x_reattention.py --num-gpus 4  mae_checkpoint.path=pretrained/model_converted.pth

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python lazyconfig_train_net.py --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) --resume --config-file configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x.py --num-gpus 4  mae_checkpoint.path=pretrained/mae_pretrain_vit_base_full.pth

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python lazyconfig_train_net.py --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) --config-file configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x_reattention.py --num-gpus 4  mae_checkpoint.path=pretrained/model_converted.pth


# python lazyconfig_train_net.py --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) --config-file configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x_reattention.py --num-gpus 8  mae_checkpoint.path=pretrained/model_converted.pth

python lazyconfig_train_net.py --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) --config-file configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x.py --num-gpus 8  mae_checkpoint.path=pretrained/mae_pretrain_vit_base_full.pth
