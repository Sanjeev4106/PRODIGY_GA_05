# PRODIGY_GA_05
Neural Style Transfer
This project implements Neural Style Transfer (NST) to combine the artistic style of one image (for example, a famous painting) with the content of another image (such as a photograph). The result is a new image that retains the content of the original photo but appears as if it were painted in the chosen artistic style.

Features
Apply the style of famous artworks to custom photos.

Generate high-quality stylized outputs using pre-trained deep neural networks.

Adjustable weights to control the balance between style and content.

Example
Content Image: A photograph of a landscape
Style Image: The Starry Night by Vincent van Gogh
Output: The landscape rendered in the swirling, colorful style of Van Gogh’s painting.

How It Works
The project uses a pre-trained VGG19 convolutional neural network to:

Extract content representations from the content image.

Extract style representations from the style image using Gram matrices.

Iteratively optimize a generated image to match the content of the first image and the style of the second.
