import nibabel as nib
import numpy as np
from stl import mesh
from skimage import measure
import scipy
import sys
import matplotlib.pyplot as plt


# parameters

infile = "my_brain.nii"
outfile = "segment.stl"

thresholds = [[[70, 70], [70, 70], [120, 120]],
             [[160, 160], [150, 150], [230, 230]]]

#thresholds = [[[A, B], [C, D], [E, F]],
#             [[G, H], [I, J], [K, L]]]

xbounds = [72, 174]
ybounds = [70]
zbounds = [90]


data_sigma = 0.4
mask_sigma = 5


# load file 
nifti = nib.load(infile)
data = nifti.get_fdata()

# normalize data 
# data = data/data.max()

# swap the axes so it is easier to process
data = np.swapaxes(data, 0, 2)

# prepare mask 

mask = np.zeros(data.shape)

ybounds_use = [0] + ybounds + [data.shape[0]]
xbounds_use = [0] + xbounds + [data.shape[1]]
zbounds_use = [0] + zbounds + [data.shape[2]]

for yidx in range (1, len(ybounds_use)):
    ymin = ybounds_use[yidx-1]
    ymax = ybounds_use[yidx]
    for xidx in range (1, len(xbounds_use)):
        xmin = xbounds_use[xidx-1]
        xmax = xbounds_use[xidx]
        for zidx in range (1, len(zbounds_use)):
            zmin = zbounds_use[zidx-1]
            zmax = zbounds_use[zidx]
            t = thresholds[yidx-1][xidx-1][zidx-1]
            mask[ymin:ymax, xmin:xmax, zmin:zmax] = t

#blur the mask
mask = scipy.ndimage.gaussian_filter(mask, sigma = mask_sigma)
        
# apply mask
data[data<mask] = 0

# blur the data

data = scipy.ndimage.gaussian_filter(data, sigma = data_sigma)

# apply mask again
data[data<mask] = 0


# swap indices back

data = np.swapaxes(data, 0, 2)

# save data as stl
verts, faces, normals, values = measure.marching_cubes(data, 0.0)
obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    obj.vectors[i] = verts[f]

obj.save(outfile)

# Optional step to apply blender smoothing
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.import_mesh.stl(filepath=outfile)
obj_name = bpy.data.objects.keys()[0]
obj = bpy.data.objects[obj_name]

bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_add(type="CORRECTIVE_SMOOTH")

bpy.context.object.modifiers["CorrectiveSmooth"].scale = 0
bpy.ops.export_mesh.stl(filepath=outfile, use_selection=True)

