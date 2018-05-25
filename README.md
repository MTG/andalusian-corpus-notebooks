# Arab-Andalusian Corpus Analysis 

## Description

This repository contains a docker-compose file to run a Jupyter server and the notebooks to download and to analyze, data and metadata from the Arab Andalusian Corpus of Dunya (http://dunya.compmusic.upf.edu/). This repository contains four notebooks:

*  **Corpus.ipynb**:  to download data and metadata from the Corpus, and to compute the pitch profile, the pitch distribution and the tonic frequency of each recording;
* **Metadata.ipynb**: to group, to visualize and to analize metadata;
* **NawbaPitchAnalysis.ipynb**: to visualize pitch distribution and note/class distribution of a single recording or of a group of them.
* **NawbaRecognition.ipynb**: to compute several experiments to evaluate the performance on nawba recognition of algorithms based on templates derived from scores.


## Install docker
To run the notebooks, you need to first install docker. Here you can find the links to installation instructions for different operative systems:

### Windows
https://docs.docker.com/docker-for-windows/install/

### Mac
https://docs.docker.com/docker-for-mac/install/

### Ubuntu
https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce

## Usage
As first step, run the Jupyter server.
Then, in terminal, cd to your local folder for this repository and run the following command:
```
sudo docker-compose up
```

This would install a docker image (first time, it would download the image of size ~2Gb) and provide a web link (http://0.0.0.0:8888). Clicking or copy-pasting this link to a browser, one can access ipython notebooks (Windows users: http://{YourIPaddress}:8888). The password required is mir.

Then, you can access the notebooks from the browser and run them. All the notebooks contain the user instructions.

#### NB: the computation of the data and metadata can require a couple of days. For this reason the Nawba recognition experiment include a zip file with the necessary pre-computed files.

## Credits
This work is based on a collaboration between Niccolò Pretto, Barış Bozkurt, Rafael Caro Repetto and Xavier Serra as part of https://www.upf.edu/web/musicalbridges. The notebooks use the Arab Andalusian Corpus in Dunya, created during the CompMusic project (http://compmusic.upf.edu/).

These notebooks is based on **MIR-docker-extension** (https://github.com/MTG/MIR-docker-extension).
The repository includes snippets of code and algorithms from the following repositories:
* **pycompmusic**: https://github.com/MTG/pycompmusic
* **dunya**: https://github.com/MTG/dunya
* **tomato**: https://github.com/sertansenturk/tomato
* **morty**: https://github.com/altugkarakurt/morty
* **dunya** desktop: https://github.com/MTG/dunya-desktop


## References
[1] Pretto, N; Bozkurt, B; Caro Repetto, R; and Serra, X (2018) "Nawba Recognition for Arab-Andalusian Music using Templates from Music Scores", in Proceedings of the 15th Sound and Music Computing Conference (SMC2018), (in press).
