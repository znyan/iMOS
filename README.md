# iMOS
Implementation of "A Foundation Model for General Moving Object Segmentation in Medical Images"  (accepted by **IEEE ISBI 2024**)
[arXiv](https://arxiv.org/abs/2309.17264)

# Dataset
| Source      | Modality              | Link                                                        |
| ----------- | --------------------- | ----------------------------------------------------------- |
| AMOS2022    | CT / MRI              | [link](https://amos22.grand-challenge.org/)                 |
| CAMUS       | Ultrasound            | [link](https://www.creatis.insa-lyon.fr/Challenge/camus/)   |
| CholecSeg8k | Endoscopy             | [link](https://www.kaggle.com/datasets/newslab/cholecseg8k) |
| EPFL        | Electron Microscopy   | [link](https://www.epfl.ch/labs/cvlab/data/data-em/)        |

# Run
```
python -m torch.distributed.launch --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain_stage3_only --stage 3 --load_network XMem.pth
```
Pretrained XMem weight can be found [here](https://github.com/hkchengrex/XMem).

# Acknowledgments
Our code is based on [XMem](https://github.com/hkchengrex/XMem). We appreciate the authors for their great works.

# Citation
```
@misc{yan2023foundation,
      title={A Foundation Model for General Moving Object Segmentation in Medical Images}, 
      author={Zhongnuo Yan and Tong Han and Yuhao Huang and Lian Liu and Han Zhou and Jiongquan Chen and Wenlong Shi and Yan Cao and Xin Yang and Dong Ni},
      year={2023},
      eprint={2309.17264},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
