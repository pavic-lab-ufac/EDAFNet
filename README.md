# EDAFNet: Efficient Dual Attention Fusion Network via Multi-Exposure Image for HDR Reconstruction

## Environment setup
To start, we prefer creating the environment using venv:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
(Python 3.8)

## Getting the data

The datasets we used are as follows:

- [Kalantari's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
- [Tel's dataset](https://drive.google.com/drive/folders/1CtvUxgFRkS56do_Hea2QC7ztzglGfrlB)
- [Hu's dataset](https://github.com/nadir-zeeshan/sensor-realistic-synthetic-data)


## Directory structure for the datasets

<details>
  <summary> (click to expand;) </summary>

    data_path
    └── data
        ├── Kal
        │   ├── Training
        |   |   ├── 001
        |   |   |   ├── 262A0898.tif
        |   |   |   ├── 262A0899.tif
        |   |   |   ├── 262A0900.tif
        |   |   |   ├── exposure.txt
        |   |   |   ├── HDRImg.hdr
        |   |   ├── 002
        |   |   ...
        |   |   └── 074
        │   └── Test
        │       └── Test-set
        │           ├── 001
        |           |   ├── 262A2615.tif
        |           |   ├── 262A2616.tif
        |           |   ├── 262A2617.tif
        |           |   ├── exposure.txt
        |           |   ├── HDRImg.hdr
        |           ├── 002
        |           |   ...
        |           └── 015
        ├── Tel
        │   ├── Training
        |   |   ├── scene_0001_1
        |   |   |   ├── input_1.tif
        |   |   |   ├── input_2.tif
        |   |   |   ├── input_3.tif
        |   |   |   ├── exposure.txt
        |   |   |   ├── HDRImg.hdr
        |   |   ├── scene_0001_2
        |   |   ...
        |   |   └── scene_0052_3
        │   └── Test
        |       ├── scene_0007_1
        |       |   ├── input_1.tif
        |       |   ├── input_2.tif
        |       |   ├── input_3.tif
        |       |   ├── exposure.txt
        |       |   ├── HDRImg.hdr
        |       ├── scene_0007_2
        |       |   ...
        |       └── scene_0042_3
        └── Hu
            ├── Training
            |   ├── 001
            |   |   ├── input_1_aligned.tif
            |   |   ├── input_2_aligned.tif
            |   |   ├── input_3_aligned.tif
            |   |   ├── input_exp.txt
            |   |   ├── ref_hdr_aligned_linear.hdr
            |   ├── 002
            |   ...
            |   └── 085
            └── Test
                ├── 086
                |   ├── input_1_aligned.tif
                |   ├── input_2_aligned.tif
                |   ├── input_3_aligned.tif
                |   ├── input_exp.txt
                |   ├── ref_hdr_aligned_linear.hdr
                ├── 087
                |   ...
                └── 100

</details>


## Running the model
### Training
1. Prepare the training dataset.
2. Modify `'--dataset_dir'` in the `train.py`, which contains the `../data/Kal`, `../data/Hu` and `../data/Tel`.
3. For different datasets, modify the arguments in `train.py` as follows: 
    - For Kalantari's dataset, modify arguments in the `train.py` as follows:
      - `'--test_pat'`: `'Test/Test-set'`
      - `'--ldr_prefix`: `''`
      - `'--exposure_file_name'`: `'exposure.txt'`
      - `'--label_file_name'`: `'HDRImg.hdr'`
    - For Tel's dataset, modify arguments in the `train.py` as follows:
      - `'--test_path'`: `'Test'`
      - `'--ldr_prefix'`: `''`
      - `'--exposure_file_name'`: `'exposure.txt'`
      - `'--label_file_name'`: `'HDRImg.hdr'`
    - For Hu's dataset, modify arguments in the `train.py` as follows:
      - `'--test_path'`: `'Test'`
      - `'--ldr_prefix'`: `'input'`
      - `'--exposure_file_name'`: `'input_exp.txt'`
      - `'--label_file_name'`: `'ref_hdr_aligned_linear.hdr'`
4. Run the following commands for training:
```bash
$ python train.py
```


### Testing
1. Prepare the testing dataset.
2. Modify `'--dataset_dir'` in the `test.py`, which contains the `../data/Kal`, `../data/Hu` and `../data/Tel`.
3. For different datasets, modify the arguments in `test.py` as follows: 
    - For Kalantari's dataset, modify arguments in the `test.py` as follows:
      - `'--test_path'`: `'Test/Test-set'`
      - `'--ldr_prefix`: `''`
      - `'--exposure_file_name'`: `'exposure.txt'`
      - `'--label_file_name'`: `'HDRImg.hdr'`
    - For Tel's dataset, modify arguments in the `test.py` as follows:
      - `'--test_path'`: `'Test'`
      - `'--ldr_prefix'`: `''`
      - `'--exposure_file_name'`: `'exposure.txt'`
      - `'--label_file_name'`: `'HDRImg.hdr'`
    - For Hu's dataset, modify arguments in the `test.py` as follows:
      - `'--test_path'`: `'Test'`
      - `'--ldr_prefix'`: `'input'`
      - `'--exposure_file_name'`: `'input_exp.txt'`
      - `'--label_file_name'`: `'ref_hdr_aligned_linear.hdr'`
4. Prepare the pretrained model.
5. Modify `'--pretrained_model'`, which corresponds to the path of the pretrained model.
6. Uncomment the following line to save the predicted HDR images:
```python
# save results
# cv2.imwrite(os.path.join(args.save_dir, '00{}_pred.hdr'.format(idx)), pred_hdr)
```
7. Run the following commands for tesing:
```bash
$ python test.py
```


## Results
Pretrained models can be find in the `./pretrain_model` folder.

<!-- ## Citation
If you find our work useful, please cite it as
```
@article{ni2025ssiu,
  title={Structural Similarity-Inspired Unfolding for Lightweight Image Super-Resolution},
	author={Ni, Zhangkai, and Zhang, Yang, and Yang, Wenhan, and Wang, Hanli, and Wang, Shiqi and Kwong, Sam},
	journal={IEEE Transactions on Image Processing},
	volume={},
	pages={},
	year={2025},
	publisher={IEEE}
}
``` -->


## Acknowledgments
This code is inspired by [AFUNet](https://github.com/eezkni/AFUNet/tree/main). We thank the authors for the nicely organized code!

