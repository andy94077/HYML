# autoencoder
## autoencoder.h5 (X)
	build_autoencoder
	with image augmentation
### output
	output.npy

## autoencoder-2.h5
	build_autoencoder
### output
	output-2.npy

## autoencoder2.h5
	build_autoencoder2
### output
	output2.npy

## autoencoder3.h5
	build_autoencoder3
### output
	output3.npy

## autoencoder4.h5
	build_autoencoder4
### output
	output4.npy

## autoencoder5.h5
	build_autoencoder5
### output
	output5.npy


# Clustering

## clustering (X)

## clustering-2, clustering2
	self.seed = int(1e9+7)
	self.pca = KernelPCA(1024, 'rbf', n_jobs=-1, random_state=self.seed)
	self.pca2 = KernelPCA(64, 'rbf', n_jobs=-1, random_state=self.seed)
	KMeans(n_clusters=2, n_jobs=-1, random_state=seed)

## clustering-2_whiten
	improved_transform2

## clustering2_whiten
	improved_transform2

## clustering3_whiten
	improved_transform4

## clustering4_whiten
	improved_transform5

## clustering5_whiten
	improved_transform4
