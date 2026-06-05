import numpy as np
from scipy.cluster.vq import kmeans

def generate_codebook(mfcc_vectors, k=16):

    mfcc_vectors = mfcc_vectors.T

    centroids, _ = kmeans(mfcc_vectors, k)

    return centroids