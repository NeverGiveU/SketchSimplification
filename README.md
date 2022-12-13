# Perceptual-Aware Sketch Simplification Based on Integrated VGG Layers

## Testing
```python
mkdir results
python test.py --img_fn ./dataset/val3 --out_fn ./results
```
Results of different scales `{0.50, 0.75, 1.0, 1.25}` can be found in `./results`.

## Training
```python
python main.py --dataset ./dataset 
```

## Cite
```sh
@ARTICLE{8771128,
  author={Xu, Xuemiao and Xie, Minshan and Miao, Peiqi and Qu, Wei and Xiao, Wenpeng and Zhang, Huaidong and Liu, Xueting and Wong, Tien-Tsin},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Perceptual-Aware Sketch Simplification Based on Integrated VGG Layers}, 
  year={2021},
  volume={27},
  number={1},
  pages={178-189},
  doi={10.1109/TVCG.2019.2930512}}
```