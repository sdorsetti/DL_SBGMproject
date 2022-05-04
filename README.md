# DL_SBGMproject
## Jeremy Berloty | Stanislas d'Orsetti

This repository contains all the necessary tools to generate a Mozart's Sonata, based on the work of Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole : Score-based generative modeling through stochastic differential equations, 2021. 

It contains the following folders: 
- data : contains the Mozart's piano sonatas in midi format, the compressed encoded data, pretrained checkpoint for VAE and SBGM models and the very final output of the project, knowing generated "mozart's-sonatas-like" melodies.
- MidiFile : contains all the intelligence to parse and encode midi file as dummies (piano rolls) with pretty-midi
- preprocessing : contains utils function for parsing Midi Files
- utils : contains utils function to plot results, display audio in notebooks are save piano rolls to midi
- VAE : contains the intelligence to train a VAE to map a sample from a sonate in a latent space. 

A notebook : final_dorsetti_berloty.ipynb : explains and shows the main outputs that this repository give, when using SBGM.

A pdf : final_dorsetti_berloty.pdf : The report of our project, with litterature review and further explications of the SBGM, the VAEs and the implementation of them.

Finally, a structure.py file contains the main path to the data, if one wants to use heavier data in a cloud environnement such a google drive. Just clone the repository and change the path. The repository is "colab-friendly" and can be ran easaly on any google collab. 

**********************************************
PLEASE VISIT OUR SHARED GOOGLE COLAB AT: 
https://colab.research.google.com/drive/1UlHGcRHQDY5uSIO50JA-jtRYaueu2KSv?usp=sharing
**********************************************


