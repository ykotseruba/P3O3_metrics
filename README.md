### Code for metrics and SMILER experiment files used in the paper Kotseruba et al. "Do Saliency Models Detect Odd-One-Out Targets? New Datasets and Evaluations", BMVC, 2019.

### Computing metrics

In our paper we propose global saliency index (GSI) and number of fixations to find the target to measure the performance of the saliency algorithms on stimuli in the Psychophysical Patterns (P<sup>3</sup>) dataset.

For the Odd-One-Out (O<sup>3</sup>) dataset we compute the ratios of maximum saliency values within the target and the distractors (MSR<sub>targ</sub>) and within the target and the background (MSR<sub>bg</sub>) areas. These metrics measure how well the algorithms are able to discriminate the target.

The code for GSI and MSR metrics is defined in `metrics.py`. 

Run `python demo.py` to see these how these scores are computed for samples from P<sub>3</sub> and O<sub>3</sub> datasets.


### Computing the saliency maps

To compute saliency maps for images in P<sup>3</sup> and O<sup>3</sup> datasets:

1. Download the datasets manually from <http://data.nvision2.eecs.yorku.ca/P3O3/> or using the script in the `data` folder:

```
cd data
sh download_data.sh
```

2. Install SMILER. Follow the instructions in the official repository <https://github.com/TsotsosLab/SMILER>.

3. Run the models using the `yaml` files in the `SMILER_experiments` folder (update the paths to the P<sup>3</sup> and O<sup>3</sup> images if needed) as follows:

```
./smiler run -e SMILER_O3.yaml
```

Note that depending on the system running all 20 models on P<sup>3</sup> and O<sup>3</sup> datasets may take several days. It is not recommended to run several experiments concurrently.

### P<sup>3</sup> image properties
There is a csv text file `image_properties.txt` associated with each set of `colors`, `orientations` and `sizes` images. It lists the following set of properties for each image:
+ `path` - type of the image (colors, orientations or sizes)
+ `name` - image name (e.g. image_rectangle_-90_-30_45_15.png)
+ `bg_color` - hex representation of the background color
+ `t_pos` - target position in the 7x7 grid (from 1 to 49)
+ `t_x`,` t_y` - pixel coordinates of the target center
+ `t_ori` - target orientation in degrees
+ `d_ori` - distractor orientation in degrees
+ `t_color` - target color in hex representation
+ `d_color` - distractor color in hex representation
+ `t_shape` - target shape (e.g. rectangle, circle)
+ `d_shape` - distractor shape
+ `t_height` - target height in pixels
+ `d_height` - distractor height in pixels
+ `t_height2`, `d_height2` - optional height parameter for sume shapes
+ `t_width` - target width in pixels
+ `d_width` - distractor width in pixels

### O<sup>3</sup>image properties
There is a csv text file `image_properties.txt` listing the following properties for each image in O<sup>3</sup> dataset:
+ `image_name` - file name
+ `num_distractors` - number of distractors in the image
+ `target_type` - object category for the target (to be ignored as most categories are empty)
+ `target_subtype` - object sub-category (e.g. tulip, dress, pea)
+ `target_size` - largest dimension of the target in pixels
+ `target_x`, `target_y` - pixel coordinates of the target center
+ `orientation`, `color`, `focus`, `shape`, `size`, `location`, `pattern` - feature dimensions where target differs from the distractors (each can be set to 0 or 1)

Note: `focus` refers to camera focus (e.g. target may be in focus and the rest of the scene not), `pattern` is a catch-all feature for differences in texture, material or patterns on the objects, 'location' refers to grouping effects (e.g. distractors are close to one another and the target object is relatively far away).
