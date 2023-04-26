# An-Erudite-FGVC-Model
Code release for “An Erudite Fine-Grained Visual Classification Model" (CVPR 2023)



<!-- ![Labrador](./labrador.jpg) -->



## Changelog
- 2023/04/18 upload the code.


## Requirements

- python 3.6
- PyTorch 1.7.1+cu110
- torchvision  0.8.2+cu110
- learn2learn 0.1.7

## Data
- Download datasets
- Extract them to `data/cars/`, `data/birds/` and `data/airs/`, respectively.
- Split the dataset into train and test folder, the index of each class should follow the Birds.xls, Air.xls, and Cars.xls

* e.g., CUB-200-2011 dataset
```
  -/birds/train
	         └─── 001.Black_footed_Albatross
	                   └─── Black_Footed_Albatross_0001_796111.jpg
	                   └─── ...
	         └─── 002.Laysan_Albatross
	         └─── 003.Sooty_Albatross
	         └─── ...
   -/birds/test	
             └─── ...         
```



## Training
- `python main.py` (Mix Cars and Flowers)




## Citation
If you find this paper useful in your research, please consider citing:
```
@InProceedings{Chang2023Erudite,
  title={An Erudite Fine-Grained Visual Classification Model},
  author={Chang, Dongliang and Tong, Yujun and Du, Ruoyi and Timothy, Hospedales and Song, Yi-Zhe and Ma, Zhanyu },
  booktitle = {Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- changdongliang@bupt.edu.cn
- mazhanyu@bupt.edu.cn
