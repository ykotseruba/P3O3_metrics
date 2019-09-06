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
