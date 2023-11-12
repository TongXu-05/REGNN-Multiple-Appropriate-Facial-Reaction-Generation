# REGNN-Multiple-Appropriate-Facial-Reaction-Generation
Reversible Graph Neural Network-based Reaction Distribution Learning for Multiple Appropriate Facial Reactions Generation

## Prepare Datasets
Extract the image features using the pretrained swin_transformer and the audio features using vggish. Then Store them under the directory like
`train/video_pth/NoXI/001_2016-03-17_Paris/speaker/001.pth`

Then process the neighbor.npy file, store it as a json in `. /data/sup_data/neighbor.json`.

## Train

`bash train.sh`

## Inference
modify the model_pth variable in the `inference.sh` with the checkpoint path you want to use. Then

`bash inference.sh`
