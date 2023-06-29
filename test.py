import os
import cv2
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    cracks=0
    wm=0
    mixed=0
    bg=0
    masks_path = '/home/lean-ai-yaniv/projects/for_segmentation/results/masks'
    files = [os.path.join(masks_path,f) for f in os.listdir(masks_path)]
    for file in files:
        im =  cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        cracks += (1 in np.unique(im)) and (2 not in np.unique(im))
        wm += (1 not in np.unique(im)) and (2 in np.unique(im))
        mixed += (1 in np.unique(im)) and (2 in np.unique(im))
        bg += (1 not in np.unique(im)) and (2 not in np.unique(im))
    print(f'cracks only:{cracks}, wm only:{wm}, mixes:{mixed}, background:{bg}')
    # impath = '/home/lean-ai-yaniv/projects/for_segmentation/results/images'
    # maskpath = '/home/lean-ai-yaniv/projects/for_segmentation/results/masks'
    # imgs = [os.path.join(impath,f) for f in os.listdir(impath)]
    # masks = [os.path.join(maskpath,f) for f in os.listdir(maskpath)]
    # for m in masks:
    #     mask = cv2.imread(m,cv2.IMREAD_GRAYSCALE)
    #     if all(np.unique(mask) == [0]):
    #         basefilename = Path(m).stem
    #         path_to_image = os.path.join(impath,basefilename.rsplit('_',1)[0]+'.png')
    #         path_to_mask = os.path.join(maskpath,basefilename+'.png')
    #         os.remove(path_to_image)
    #         os.remove(path_to_mask)
