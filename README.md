# Depth_Estimation

Learning depth information from image is a crucial topic in computer vision.  It is also a problem under the general topic [Geometry learning](http://geometricdeeplearning.com/)  This project target to explore and build machine learning model that can output depth or relative depth from input image either in a supervised or unsupervised manner.


## Benchmark datasets
1. Indoor scene: [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
2. Outdoor scene: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php).
3. 3D model related: [Make3D](http://make3d.cs.cornell.edu/data.html).

## Relevant papers for depth estimation/prediction

Depth map prediction networks:

| Paper | Description |
| --- | --- |
| [Learning Depth from Single Images with Deep Neural Network Embedding Focal Length](https://arxiv.org/abs/1803.10039) | Fully supervised method considering varying focal length |
| [Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image](https://arxiv.org/abs/1709.07492) | Depth prediction with sparse depth samples augmentation |
| [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283) | Coarse network + fine network (prior work for SOA on NYUv2) |
| [Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture](https://arxiv.org/abs/1411.4734) | SOA model on NYUv2 in 2016|
| [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373) | SOA model FRCN on NYUv2 using ResNet in 2016 |
| [Deep Ordinal Regression Network for Monocular Depth Estimation](https://arxiv.org/abs/1806.02446) | SOA model DORN on NYUv2 using ResNet in 2018 & 1st prize in Robust Vision Challange 2018 |

Global vs Local:

| Paper | Description |
| --- | --- |
| [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) | Layer structure designed for spatial/time interactions or correlations |
| [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719) | Proposed Global Convolutional Network for contradictory between classification and localization |
|[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)|Dilated convolution for dense prediction problem|




BerHu loss:

| Paper | Description |
| --- | --- |
| [A unified approach to model selection and sparse recovery using regularized least squares](https://arxiv.org/abs/0905.3573) | A smooth homotopy function between $L_0$ and $L_1$ norm as penalty |


Image Models (texture):

| Paper | Description |
| --- | --- |
| [Learning FRAME Models Using CNN Filters](https://arxiv.org/abs/1509.08379) | Markov random field model for texture|
|[A Theory of Generative ConvNet](https://arxiv.org/abs/1602.03264)|Theory and Intuitions of Generative CNN|
|[Generative Modeling of Convolutional Neural Networks](https://arxiv.org/abs/1412.6296)|Generative modeling CNN|
|[Filters, Random Fields and Maximum Entropy (FRAME): Towards a Unified Theory for Texture Modeling](https://link.springer.com/article/10.1023/A:1007925832420)|FRAME model for texture|
|[Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields](https://arxiv.org/abs/1502.07411)|CRF Model with CNN for Depth Y given Image X|
|[Deep Convolutional Neural Fields for Depth Estimation from a Single Image](https://arxiv.org/abs/1411.6387)|Basically same work on CRF model|
|[Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation](https://arxiv.org/abs/1704.02157)|CRF defined acorss multi-scale feature with attention gates inference by mean field approx|
|[Structured Attention Guided Convolutional Neural Fields for Monocular Depth Estimation](https://arxiv.org/abs/1803.11029)|Structured attention model jointly learned with CRF. Update: mean field approx|

Learning Energy based model (EBM) and Score Matching (SM):

| Paper | Description |
| --- | --- |
| [On Learning Non-Convergent Short-Run MCMC Toward Energy-Based Model](https://arxiv.org/abs/1904.09770) |Learning EBM using MCMC approach for approximating gradient|
|[Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126)|Energy based GAN, generator as a transformation sampler and discriminator as a energy function evaluator|
|[Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching](http://www.stat.ucla.edu/~ywu/CoopNets/doc/CoopNets_AAAI.pdf)|Jointly learning an EBM with a latent variable model|
|[Divergence Triangle for Joint Training of Generator Model, Energy-based Model, and Inference Model](https://arxiv.org/pdf/1812.10907.pdf)|Model 3 different joint distribution to avoid MCMC sampling|
|[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)|Learning data generative score function using Score Matching and generate samples by Langevin Dynamics|


Learning Density (including EBM) with other divergences:

| Paper | Description |
| --- | --- |
| [A Kernelized Stein Discrepancy for Goodness-of-fit Tests and Model Evaluation](https://arxiv.org/abs/1602.03253) |Kernelized Stein Discrepancy (KSD) as a computable measure of discrepancy between a sample of an unnormalized distribution|
|[Estimation of Non-Normalized Statistical Models by Score Matching](http://www.jmlr.org/papers/v6/hyvarinen05a.html)|Proposeed learning unormalized models by minimizing the expected squared distance between the gradient of the log-density from the model and the observed data.|
|[Information criteria for non-normalized models](https://arxiv.org/abs/1905.05976)|Information criteria for noise contrastive estimation (NCE) and score matching.|
|[Exponential Family Estimation via Adversarial Dynamics Embedding](https://arxiv.org/abs/1904.12083)|"We consider the primal-dual view of the MLE for the kinectics augmented model, which naturally introduces an adversarial dual sampler."|
