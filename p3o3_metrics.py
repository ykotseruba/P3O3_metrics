"""
The code for metrics described in the paper:

I. Kotseruba, C. Wloka, A. Rasouli, J.K. Tsotsos "Do Saliency Models Detect Odd-One-Out Targets? New Datasets and Evaluations", BMVC, 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
import cv2
import random
import skimage.morphology as morph
from scipy.misc import imread
import matplotlib.pyplot as plt

'''
Compute global saliency index (GSI) as defined in Soltani&Koch, 2010:
GSI = (R_target - R_dist)/(R_target + R_dist)
where R_target is the average saliency within the target mask 
and R_distr is the average saliency within the distractor masks.
The values are between -1 and 1.
GSI < 0 means that the target is not salient, the closer GSI is to 1
the more salient is the target compared to the distractors.

:param salmap: path to saliency map or saliency map image
:param targmap: path to target mask or target mask image
:param distmap: path to distractor mask or distractor mask image
:param add_eps: add small constant to saliency map to break the ties
'''


def GSI(salmap, targmap, distmap, add_eps=False):
    if isinstance(salmap, str):
        # we only want the grayscale version, since saliency maps should all be grayscale
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE)
    if isinstance(targmap, str):
        # assume that this is a grayscale binary map with white for target and black for non-target
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE)
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE)

    if add_eps:
        randimg = [random.uniform(0,1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_normalized = targmap / 255
    distmap_normalized = distmap / 255

    R_target = np.sum(np.multiply(salmap, targmap_normalized))/np.sum(targmap_normalized)
    R_distr = np.sum(np.multiply(salmap, distmap_normalized))/np.sum(distmap_normalized)

    if (R_target + R_distr) > 0:
        score = (R_target-R_distr)/(R_target + R_distr)
    else:
        score = -1

    return score

'''
Compute the ratio of the maximum saliency within target mask
and the maximum saliency within distractor masks
:param salmap: path to saliency map or saliency map image
:param targmap: path to target mask or target mask image
:param distmap: path to distractor mask or distractor mask image
:param dilate: number of pixels to dilate the target and distractor maps to allow for saliency bleed
:param add_eps: add small constant to saliency map to break the ties
:return: MSR_targ score, or -1 if the maximum distractor saliency is 0
'''


def MSR_targ(salmap, targmap, distmap, dilate=0, add_eps=False):
    if isinstance(salmap, str):
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE) # assume that this is a grayscale binary map with white for target and black for non-target
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE) # assume that this is a grayscale binary map with white for distractors and black for non-distractors

    if add_eps:
        randimg = [random.uniform(0, 1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_copy = targmap.copy()
    distmap_copy = distmap.copy()

    # dilate the target and distractor maps to allow for saliency bleed
    if dilate > 0:
        targmap_copy = morph.dilation(targmap_copy.astype(np.uint8), morph.disk(dilate))
        distmap_copy = morph.dilation(distmap_copy.astype(np.uint8), morph.disk(dilate))

    # convert the target and distractor masks into arrays with 0 and 1 for values
    targmap_normalized = targmap_copy / 255
    distmap_normalized = distmap_copy / 255
    salmap_normalized = salmap/255

    maxt = np.max(np.multiply(salmap_normalized, targmap_normalized))
    maxd = np.max(np.multiply(salmap_normalized, distmap_normalized))

    if maxd > 0:
        score = maxt/maxd
    else:
        score = -1

    return score

'''
Compute the ratio of the maximum saliency within target mask
and the maximum saliency within background mask
:param salmap: path to saliency map or saliency map image
:param targmap: path to target mask or target mask image
:param distmap: path to distractor mask or distractor mask image
:param dilate: number of pixels to dilate the target and distractor maps to allow for saliency bleed
:param add_eps: add small constant to saliency map to break the ties
:return: MSR_targ score, or -1 if the maximum distractor saliency is 0
'''


def MSR_bg(salmap, targmap, distmap, dilate=0, add_eps=False):
    if isinstance(salmap, str):
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE) # a grayscale binary map with white for target and black for non-target
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE) # a grayscale binary map with white for distractors and black for non-distractors

    if add_eps:
        randimg = [random.uniform(0,1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_copy = targmap.copy()
    distmap_copy = distmap.copy()

    # dilate the target and distractor maps to allow for saliency bleed
    if dilate > 0:
        targmap_copy = morph.dilation(targmap_copy.astype(np.uint8), morph.disk(dilate))
        distmap_copy = morph.dilation(distmap_copy.astype(np.uint8), morph.disk(dilate))

    # convert the target and distractor masks into arrays with 0 and 1 for values
    targmap_normalized = targmap_copy / 255
    distmap_normalized = distmap_copy / 255
    salmap_normalized = salmap / 255
    # compute background mask from the target and distractor masks
    bgmap_normalized = 1 - np.logical_or(targmap_normalized > 0.5, distmap_normalized > 0.5)

    maxt = np.max(np.multiply(salmap_normalized, targmap_normalized))
    maxb = np.max(np.multiply(salmap_normalized, bgmap_normalized))

    if maxt > 0:
        score = maxb/maxt
    else:
        score = -1

    return score


'''
Count the number of fixations necessary to find the target.
Target is found if the maximum saliency value lands within the target map and 
number of fixations does not exceed max_tries.
:param salmap: the input saliency map being tested or its path
:param targmap: binary target mask or a string path
:param iorrec: (width, height) of the rect. region to blot out around saliency maxima 
:param gaussian: add gaussian decay towards the edges of the iorrec
:param dilate: the number of pixels of dilation to apply to the target map (defaults to 0 for no dilation)
:param max_tries: the maximum number of fixations to try; if 0 or less continue until the target is found
:param add_eps: add a small value to the salmap to break any pixel value ties
:returns: tries - the number of fixations needed to find the target, found - true or false
'''


def numfix2find(image, salmap, targmap, iorrec,
                find_box=(1, 1),
                gaussian=False,
                dilate=0,
                max_tries=0,
                add_eps=False,
                visualize=False):

    if isinstance(image, str) and visualize:
        image = imread(image)
    if isinstance(salmap, str):
        salmap = imread(salmap, flatten=True) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = imread(targmap, flatten=True) # assume that this is a grayscale binary map with white for target and black for non-target

    searchmap = salmap / 255
    targmap_normalized = targmap / 255

    # add small perturbations to the saliency map to avoid ties
    if add_eps:
        randimg = [random.uniform(0,1/100000) for _ in range(searchmap.size)]
        randimg = np.reshape(randimg, searchmap.shape)
        searchmap = searchmap + randimg

    # dilate the target map to allow for near hits
    if dilate > 0:
        targmap_normalized = morph.dilation(targmap_normalized, morph.disk(dilate))

    # iteratively grab the maximum point and inhibit until the maximum falls
    # within the target map or the maximum number of tries have been reached
    tries = 0
    found = False
    out_of_tries = False
    (h, w) = searchmap.shape

    if visualize:
        plt.ion()
        plt.axis('off')

    while not found and not out_of_tries:
        (x, y) = np.unravel_index(np.argmax(searchmap), searchmap.shape)
        tries += 1

        found = targmap_normalized[x, y] > 0 

        if not found:
            # we haven't found the target
            # inhibit the region around our fixation and retry
            x_min = max(0, int(x-iorrec[0]/2))
            x_max = min(h, int(x+iorrec[0]/2))
            y_min = max(0, int(y-iorrec[1]/2))
            y_max = min(w, int(y+iorrec[1]/2))

            if gaussian:
                [xv, yv] = np.meshgrid(range(x_min, x_max), range(y_min, y_max), indexing='ij')
                iorrec_patch = np.sqrt(np.power(xv-x, 2)+np.power(yv-y, 2))
                iorrec_patch = iorrec_patch/np.max(iorrec_patch)
                searchmap[x_min:x_max, y_min:y_max] = np.multiply(searchmap[x_min:x_max, y_min:y_max], iorrec_patch)
            else:
                searchmap[x_min:x_max, y_min:y_max] = 0

        if max_tries > 0 and tries > max_tries:
            out_of_tries = True

        if visualize:
            plt.subplot(131)
            plt.title('Image')
            plt.imshow(image)
            plt.plot(y, x, 'r+', linewidth=3)
            plt.axis('off')
            plt.subplot(132)
            plt.title('Search Map\nfixations: {}'.format(tries))
            plt.imshow(np.power(searchmap, 3))
            plt.axis('off')
            plt.subplot(133)
            plt.title('Target Map')
            plt.imshow(targmap, cmap="gray")
            plt.axis('off')
            plt.show()
            plt.pause(0.0001)

    if visualize:
        plt.pause(5)
        plt.close()

    return tries, found
