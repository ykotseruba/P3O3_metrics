### Code for metrics and SMILER experiment files used in the paper I. Kotseruba, C. Wloka, A. Rasouli, "Do Saliency Models Detect Odd-One-Out Targets? New Datasets and Evaluations", BMVC, 2019.


In our paper we compute global saliency index (GSI) and number of fixations to find the target (numfix2find) for stimuli in the Psychophysical Patterns (P<sup>3</sup>) dataset.

For the Odd-One-Out (O<sup>3</sup>) dataset we compute the ratios of maximum saliency values within the target and the distractors (MSR<sub>targ</sub>) and within the target and the background (MSR<sub>bg</sub>) areas.

The code for these metrics is defined in `metrics.py`. 

Run `python demo.py` to see these how these scores are computed for samples from P<sub>3</sub> and O<sub>3</sub> datasets.


### SMILER experiment files

To compute saliency maps for images in P<sub>3</sub> and O<sub>3</sub> datasets:

1. Download the datasets manually from <http://data.nvision2.eecs.yorku.ca/P3O3/> or using the script in the `data` folder:

```
cd data
sh download_data.sh
```

2. Install SMILER. Follow the instructions in the official repository <https://github.com/TsotsosLab/SMILER>.

3. Run the models using the `yaml` files in the `SMILER_experiments` folder (updated the paths to the P<sub>3</sub> and O<sub>3</sub> images if needed):

```
./smiler run -e SMILER_O3.yaml
```

Note that depending on the system running all 18 models on P<sub>3</sub> and O<sub>3</sub> datasets may take several days. It is not recommended to run several experiments concurrently.