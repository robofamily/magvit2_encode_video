# magvit2_encode_video

## Prepare
Set up HF-Mirror:
```
export HF_ENDPOINT=https://hf-mirror.com
apt install -y curl git-lfs aria2
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

Download Open-MagViT2:
```
./hfd.sh TencentARC/Open-MAGVIT2 --model --tool aria2c -x 4
```

Download SDXL-VAE:
```
./hfd.sh stabilityai/sdxl-vae --model --tool aria2c -x 4
```

## Encode and Decode Mp4 Video
Use SDXL-VAE:
```
python sd_encode_and_decode_mp4.py --video_path path/to/mp4 --ckpt_path path/to/ckpt/dir
```
Use Open-MagViT2:
```
python encode_and_decode_mp4.py --video_path path/to/mp4 --ckpt_path path/to/ckpt
```
You should set `ch_mult` in `magvit2/config.py`
```
ch_mult: tuple[int] = (1, 2, 2, 4) # For low compression ratio ckpt
ch_mult: tuple[int] = (1, 1, 2, 2, 4) # For low compression ratio ckpt
```
