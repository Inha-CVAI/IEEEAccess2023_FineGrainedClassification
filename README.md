# **Fine-Grained Classification via Hierarchical Feature Covariance Attention Module**

--- 

## **DATASETS**

---

CUB_200_2011 (CUB) http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

Standford Cars (CAR) https://ai.stanford.edu/~jkrause/cars/car_dataset.html

FGVC-Aircraft (AIR) https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/


## **Train**

---

```
python train.py --dataset_name cub
```

## **Evaluation**

---

Run code:

```
python test.py --dataset_name cub --load cub_resnet50_4
```

