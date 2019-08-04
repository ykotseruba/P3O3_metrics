"""
Code showing how to use the metrics described in the paper:

I. Kotseruba, C. Wloka, A. Rasouli, "Do Saliency Models Detect Odd-One-Out Targets? New Datasets and Evaluations", BMVC, 2019.

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

from metrics import GSI, numfix2find, MSR_targ, MSR_bg
import numpy as np
from skimage.draw import circle


################## P3 ###################

# the code below shows how to compute number of fixations to find the target
# and GSI score for any stimuli in the P3 dataset

P3_img_name = 'image_circle_90_20_36_16.png'

P3_img = 'examples/images/'+P3_img_name
P3_distmap = 'examples/dist_images/'+P3_img_name
P3_targmap = 'examples/targ_images/'+P3_img_name
P3_targ_x, P3_targ_y = (790, 663) #target coordinates can be found in image_properties.txt
targ_size = 75 # target and distractor sizes can be found in image_properties.txt
dist_size = 75

P3_salmap_DGII = 'examples/salmap_DGII/'+P3_img_name

# create circular mask with circle of diameter=targ_size around the target center
targmap_circle = np.zeros((1024, 1024), dtype=np.uint8) 
rr, cc = circle(P3_targ_y, P3_targ_x, int(targ_size/2))
for r, c in zip(rr, cc):
    if r >=0 and c >=0 and r < targmap_circle.shape[0] and c < targmap_circle.shape[1]:
        targmap_circle[r, c] = 255


# this will show how number of fixation to find the target is computed
tries, found = numfix2find(P3_img, P3_salmap_DGII, targmap_circle, (dist_size, dist_size), max_tries=99, visualize=True)
print('Target found: {}  num_fixations: {}'.format(found, tries))

gsi = GSI(P3_salmap_DGII, P3_targmap, P3_distmap)
print('GSI: {}'.format(gsi))


############## O3 ####################

# the code below shows how to compute MSR_targ score for target vs distractor and
# and MSR_bg score for target vs background

O3_img_name = 'C4kfsNTXa2Irj.jpg'
O3_img = 'examples/images/'+O3_img_name
O3_distmap = 'examples/dist_images/'+O3_img_name
O3_targmap = 'examples/targ_images/'+O3_img_name

O3_salmap_DGII = 'examples/salmap_DGII/'+O3_img_name

msr_targ = MSR_targ(O3_salmap_DGII, O3_targmap, O3_distmap)
msr_bg = MSR_bg(O3_salmap_DGII, O3_targmap, O3_distmap)

print('MSR_targ: {} MSR_bg: {}'.format(msr_targ, msr_bg))
