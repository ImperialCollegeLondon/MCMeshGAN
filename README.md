# Multimodal Conditional MeshGAN for Personalized Aneurysm Growth Prediction

This is the official implementation of MCMeshGAN for predicting aneurysm growth through mesh-to-mesh translation.

We provide the source code for MCMeshGAN, along with baseline models for comparison. Additionally, we include tools for converting longitudinal CT data of thoracic aortic aneurysms into 3D mesh representations.


The structure of the proposed MCMeshGAN
<div align=center>
<img src="MCMeshGAN.png" title="Overview of MCMeshGAN" width="800" height="300"/></div>


### Prerequisites

* Python 3.9
* PyTorch 2.3.1
* Pytroch3D 0.7.7
* Pytorch Geometric 2.5.3

## Conda Environment Setting
```
conda create --name MCMeshGAN python=3.9
conda activate MCMeshGAN
```
### 1. Pytorch Installation
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
### 2. Pytorch3D Installation
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.7/download/linux-64/pytorch3d-0.7.7-py39_cu121_pyt231.tar.bz2
```
### 3. Pytorch Geometric Installation
```
pip install torch_geometric
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

## Training
To start a training session run
``` 
python main.py --mode train --epochs 1000
```
The trained model will be saved in ```MCMeshGAN/trained_models/```
## Testing
To start a testing session run
``` 
python main.py --mode test --load MCMeshGAN/trained_models/2024_11_01/16_57/epochxxx/
```

## Comparison Baselines
All the Baselines and MCMeshGAN are constructed using the Pytorch Geometric.
The baselines, including GCNConv, GCN2Conv, SSGConv, and PMLP, are located in ``` MCMeshGAN/model.py``` 

## Tools for Mesh Data Generation from CT Data
### 1. Segmentaion Mask Generation for CT data 
1. Automatic mask generation using the public segmentation model TotalSegementor https://github.com/wasserth/TotalSegmentator
2. Manually correct the inaccurate segmentations using the 3D Slicer software https://www.slicer.org/

### 2. Generate Mesh from CT Segmentation Mask using the IRTK libray.

```
mcubes CT_Segmentation_Mask.nii.gz Mesh.vtk 0.5 -blur 2
```
For example,
```
mcubes Patient001_20191204_74_Female.nii.gz Patient001_20191204_74_Female.vtk 0.5 -blur 2
```
### 3. Mesh Alignment using IRTK https://www.doc.ic.ac.uk/~dr/software/usage.html
#### 3.1 Mesh Downsampling (Requires Open3D Installation)
(1) For each patient, choose one mesh from all their follow-up meshes as the target mesh (Target.vtk).
(2) Downs-sample the selected mesh to 10,000 vertices using:
```
python open3d_simplificcation.py
```
#### 3.2 Rigid Alignment to Align the Mesh Positions and Orientations (Requires IRTK Installation)
For each patient, align their other meshes to the target mesh (Target.vtk)
```
srreg Source.vtk Target.vtk -symmetric -dofout test.dof.gz
ptransformation Source.vtk Source_srreg.vtk -dofin test.dof.gz
```
#### 3.3 Non-Rigid Alignment for Vertex Correspondence Matching (Requires IRTK Installation)
```
snreg Target.vtk Source_srreg.vtk -symmetric -dofout test_snreg.dof.gz -epsilon 0.001 -ds 10
ptransformation Target.vtk Source_snreg.vtk -dofin test_snreg.dof.gz
```
Source_snreg.vtk is the final mesh after alignment.

## Practical Clinical Applicaion: Measurement of Maximum Aneurysm Diameter
If the maximum aneurysm diameter exceeds a specified threshold, surgical intervention or other treatment is required.
3D Slicer provides the function to measure the maximum aneurysm diameter and centerline of aorta.
<div align=center>
<img src="Centerline.png" width="500" height="300"/>   <img src="MaxDiameter.png" title="" width="500" height="300"/></div>

### Citation

This paper is under review now. If you use this code, please cite our paper later:


### License

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
These licenses allow you to use, modify and share the project for non-commercial use as long as you adhere to the conditions of the license above.
