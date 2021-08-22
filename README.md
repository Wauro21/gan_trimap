# Trimap Synthesis with a Conditional Adversarial Network

Trimap generation through a cGAN net, based on the architecture [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), given an input image containing a person. This trimap could be use as an input for the Information Flow Matting algorithm, thus obtaining an alpha channel for the person.

## Results obtained for the synthetic composite test set

Test dataset based on the [Adobe Dataset](https://sites.google.com/view/deepimagematting) for background-matting. The results showed were obtained for the 200th epoch.

Input | Output| Groundtruth|
:----:|:-----:|:-----------:
![](test_results/synth/0_input_label.jpg)|![](test_results/synth/0_synthesized_image.jpg)| ![](test_results/synth/0.png)
![](test_results/synth/2_input_label.jpg)|![](test_results/synth/2_synthesized_image.jpg)| ![](test_results/synth/2.png)
![](test_results/synth/3_input_label.jpg)|![](test_results/synth/3_synthesized_image.jpg)| ![](test_results/synth/3.png)
![](test_results/synth/5_input_label.jpg)|![](test_results/synth/5_synthesized_image.jpg)| ![](test_results/synth/5.png)
![](test_results/synth/6_input_label.jpg)|![](test_results/synth/6_synthesized_image.jpg)| ![](test_results/synth/6.png)
![](test_results/synth/8_input_label.jpg)|![](test_results/synth/8_synthesized_image.jpg)| ![](test_results/synth/8.png)
![](test_results/synth/9_input_label.jpg)|![](test_results/synth/9_synthesized_image.jpg)| ![](test_results/synth/9.png)
![](test_results/synth/10_input_label.jpg)|![](test_results/synth/10_synthesized_image.jpg)| ![](test_results/synth/10.png)

## Results obtained for the natural composite test set

This dataset was tailored for this test. The background of each image is not a product of a computational composition. Images copyright goes to their respective owners. The results showed were obtained for the 200th epoch.

Input | Output| Groundtruth|
:----:|:-----:|:-----------:
![](test_results/natural/1_input_label.jpg)|![](test_results/natural/1_synthesized_image.jpg)| ![](test_results/natural/1.png)
![](test_results/natural/2_input_label.jpg)|![](test_results/natural/2_synthesized_image.jpg)| ![](test_results/natural/2.png)
![](test_results/natural/3_input_label.jpg)|![](test_results/natural/3_synthesized_image.jpg)| ![](test_results/natural/3.png)
![](test_results/natural/4_input_label.jpg)|![](test_results/natural/4_synthesized_image.jpg)| ![](test_results/natural/4.png)

### Prerequisites
 - 
