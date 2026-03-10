# How to test age-related macular degeneration (AMD) model based on deep learning (DL)
----
Author: Dr. Waziha Kabir
Last modified: January 24, 2022
## I. Requirements
You would require the following Python scripts:
1. prepro_AMD.py
2. test_binary_AMD.py

You would require the following directories and files:
1. **AMDmodels**>>v1.0.0.0>>cycle_14_mAUC_95.33_MCC_89.73_F1_86.23>>*model_checkpoint.pth*
The trained AMD DL model is saved as *model_checkpoint.pth*.
2. **data**>>images, test_amd.csv
3. **models**>>bit_models_MOD.py, bit_models.py, BiT-M-R101x1.npz, get_model.py and repvgg.py
4. **utils**>>combo_loader.py, evaluation.py, get_loaders.py, get_mask.py, model_saving_loading.py and reproducibility.py

## II. Pre-processing
The test image is first pre-processed using the following steps.
1. Save your test image in the following location:
*data/images*
For example, *fundus_1.jpg* is the name of the test image. Then, you should find it after the saving in the following location as 
*data/images/fundus_1.jpg*
2. Save *test_amd.csv* that has the name of your test image under *image_id* in the 1st column of the *.csv* in the following location:
*data/images*
You should find it after the saving in the following location as 
*data/images/test_amd.csv*
3. Run the following command to pre-process the test image.
*python3 prepro_AMD.py*
The pre-processed image will be saved in the following location.
*data/cropped_images*

## III. Testing
The pre-processed test image is now tested using the following step.
1.  Run the following command to predict AMD of the pre-processed test image.
*python3 test_binary_AMD.py*
The predicted AMD for the test image will be displayed on command screen as well as it will be saved in the *‘results’* directory.
The predicted AMD result would be shown on the screen as follows.
*DATA_TAG:=AMD=True; File=fundus_1.jpg*
