# MGeo
MGeo: Multi-Modal Geographic Language Model Pre-Training

# Release Note
- 2023.01.16 Pretrained model and datasets are released in modelscope
- 2023.04.05 Paper is accepted by SIGIR2023. Paper link: https://arxiv.org/abs/2301.04283
- 2023.05.17 Pretraining and finetune codes in paper are released
- 2023.07.06 Pretrained model used to reproduce paper is released


# Download

- Pretrained model and finetune code for a more general geographic model on 6 tasks are availabel at https://modelscope.cn/models/damo/mgeo_backbone_chinese_base/files

- Datasets are availabel at https://modelscope.cn/datasets/damo/GeoGLUE/files

- Pretrained model used to reproduce paper results: https://drive.google.com/file/d/1j6S52jkxks4UBsCU8ZgLroqksUYSz5x7/view?usp=sharing

# Reproduce results in paper
## Prepare environment
```shell
conda create -n mgeo python=3.7
pip install -r requirements.txt
```
## Download resources
```shell
cd data
unzip datasets.zip
cd ../prepare_data
download_pretrain_models.sh
```
## Generate pretrain data
We only provide samples of pretrain data. To produce your own pretrain data, you simply need text-geolocation pairs which can be genrate by various way (e.g., user click, POI data, position of delivery clerks). The geolocation and text just need to be related, no need to be exactly precise. 

Having text-geolocation pairs, you can follow steps below to generate pretrain data. Demo pairs are saved in resources/text_location_pair.demo for testing.

- Download proper map from [OpenStreetMap](https://download.geofabrik.de/). For example, our geolocations used in paper are in HangZhou. Thus, we download [China](https://download.geofabrik.de/asia/china.html) map.
- Import map and your text-geolocation data to a GIS database, like PostGIS. Every text is assigned with an ID. In demo case, ID is row number.
- Find COVERED and NEARBY relations between map and geolocation using GIS database. Export to resources/location_in_aoi.demo, resources/location_near_aoi.demo, resources/location_near_road.demo.
- Export needed AOIs and roads to resources/hz_aoi.txt and resources/hz_roads.txt. In our paper's setting, only elements in HangZhou are included. 
- Generate pretrain data using command: cd prepare_data && python calculate_geographic_context.py . Use the length of geom_ids in calculate_geographic_context.py to replace vocab_size in resources/gis_config.json.

## Pretrain geographic encoder
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh run_gis_encoder_pretrain.sh
```
## Pretrain multimodal interaction module
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh run_mm_pretrain.sh
```
## Finetune on rerank task
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_rerank.sh
```
## Finetune on retrieval task
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_retrieval.sh
```
## Contact
Please contact ada.drx@alibaba-inc.com to get pretrained model or other resources.

## Reference
```bib
@article{ding2023multimodal,
  title={MGeo: A Multi-Modal Geographic Pre-Training Method},
  author={Ruixue Ding and Boli Chen and Pengjun Xie and Fei Huang and Xin Li and Qiang Zhang and Yao Xu},
  journal={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}
```
