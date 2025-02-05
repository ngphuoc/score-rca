using TensorBoardLogger #import the TensorBoardLogger package
using Logging #import Logging package
using TestImages
using Flux
using MLDatasets
using LinearAlgebra

logger = TBLogger("embeddinglogs", tb_append) #create tensorboard logger

################log embeddings example: MNIST xtrain################
n_samples = 100
features = 100
xtrain, ytrain = MLDatasets.MNIST(:train)[1:100]
n = size(xtrain)[end]
imgs = xtrain
xtrain = permutedims(xtrain, (3, 1, 2))
xtrain = reshape(xtrain, n, 28*28)
xtrain = convert(Array{Float64,2}, xtrain)
svd_data = svd(xtrain)
u = svd_data.U
s = Array(Diagonal(svd_data.S))[:, 1:features]
vt = svd_data.Vt[1:features, :]
xtrain = u*s
metadata = ytrain

#using explicit function interface
log_embeddings(logger, "embedding/mnist/explicitinterface", xtrain, metadata = metadata, img_labels = TBImages(imgs, HWN), step = 1)

