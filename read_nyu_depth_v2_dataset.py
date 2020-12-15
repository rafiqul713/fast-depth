import skimage.io as io
import numpy as np
import h5py
import cv2


path="/home/rafiqul/Documents/Thesis/Experiment/DepthImage"

def skimage_to_opencv(skiimage):
    cvimage = skiimage*255
    cvimage.astype(int)
    cv2.cvtColor(skiimage, cv2.COLOR_RGB2BGR)
    return cvimage

def read_rgb_image(image, i):
    rgb_image = np.empty([480, 640, 3])
    rgb_image[:,:,0] = image[0, :, :].T
    rgb_image[:,:,1] = image[1, :, :].T
    rgb_image[:,:,2] = image[2, :, :].T
    rgb_image_ = rgb_image.astype('float32')
    rgb_image_=rgb_image_ / 255.0
    cv_img = skimage_to_opencv(rgb_image_)
    path_rgb=path+"/rgb/rgb_"+str(i)+".jpg"
    cv2.imwrite(path_rgb,cv_img)


def read_depth_image(depth,i):
    depth_ = np.empty([480, 640, 3])
    depth_[:, :, 0] = depth[:,:].T
    depth_[:, :, 1] = depth[:,:].T
    depth_[:, :, 2] = depth[:,:].T
    depth_ = depth_.astype('float32')
    depth_=depth_ / 4.0
    cv_depth = skimage_to_opencv(depth_)
    path_depth = path + "/depth/depth_" + str(i) + ".jpg"
    cv2.imwrite(path_depth,cv_depth)

def save_rgb_depth_image(path,n):
    f = h5py.File(path)
    for i in range(n):
        img = f['images'][i]
        depth = f['depths'][i]
        read_rgb_image(img,i)
        read_depth_image(depth,i)


path = './nyu_depth_v2_labeled.mat'
save_rgb_depth_image(path,100)