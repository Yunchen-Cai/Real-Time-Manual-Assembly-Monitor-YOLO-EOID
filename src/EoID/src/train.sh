## EoID Rice Cooker setting

export CUDA_LAUNCH_BLOCKING=1  # 添加这行设置环境变量

torchrun \
        main.py \
        --pretrained E:\ONGSKFYP\EoID\output\EoID_rice_cooker\checkpoint_35.pth \
        --resume E:\ONGSKFYP\EoID\output\EoID_rice_cooker\checkpoint_35.pth \
        --output_dir output/EoID_rice_cooker \
        --dataset_file rice_cooker \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 53 \
        --num_verb_classe 117 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 40 \
        --lr_drop 60 \
        --batch_size 10 \
        --clip_backbone RN50x16 \
        --model eoid \
        --inter_score \
        --vdetach \
        --rice_cooker_path "E:/ONGSKFYP/add_frames"
