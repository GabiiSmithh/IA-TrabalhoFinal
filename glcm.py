from skimage.feature import graycomatrix, graycoprops
from typing import List, Union
import skimage
import numpy as np


def glcm(img : np.ndarray, 
         distances : Union[List[int],np.ndarray] =[1,3,5], 
         angles : Union[List[float],np.ndarray] = np.deg2rad([0,90,180,270])):
    
    assert isinstance(img, np.ndarray) and len(img.shape) == 2
    hists = graycomatrix(img, distances=distances, angles=angles, normed=True, symmetric=True)
    prop_names = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    props = np.array([ graycoprops(hists, prop).flatten() for prop in prop_names]).flatten()
    
    return props
