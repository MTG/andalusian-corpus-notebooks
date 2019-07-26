# Arab-Andalusian Corpus Analysis

## Description

This repository contains a docker-compose file to run a Jupyter server and the notebooks to download and analyse data and metadata from the Arab Andalusian Corpus of [Dunya](http://dunya.compmusic.upf.edu/). This repository contains four notebooks:

* **Corpus.ipynb**:  to download data and metadata from the Corpus, to compute the pitch profile, distribution and the tonic frequency of each recording;
* **Metadata.ipynb**: to group, visualize and analyse metadata;
* **NawbaPitchAnalysis.ipynb**: to visualize pitch distribution and note/class distribution of a single recording or of a group of them.
* **NawbaRecognition.ipynb**: to compute several experiments to evaluate the performance on nawba recognition of algorithms based on templates derived from scores.

## Installation
To run the notebooks, you need to first install docker. Here you can find the links to installation instructions for different operative systems:
* **Windows**: https://docs.docker.com/toolbox/overview/

* **Mac**: https://docs.docker.com/docker-for-mac/install/

* **Ubuntu**: https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce

## Usage

In a terminal/console window, change to this directory

On MacOS or Windows, run:

    docker-compose up

On Linux, run the following (this command ensures that any files you create are owned by your own user):

    JUPYTER_USER_ID=$(id -u) docker-compose up

The first time you run this command it will download the required docker images (about 2GB in size).

Then accesss http://localhost:8888 with your browser and when asked for a
password use the default password ***mir***

Then, you can access the notebooks from the browser and run them. All the notebooks contain their user guides.
The use of Dunya data and metadata requires that you register with [Dunya](http://dunya.compmusic.upf.edu/). After the registration, a personal token is provided. This token has to be added in the `utilities/constants.py` file.

#### NB: the computation of the data and metadata can require a couple of days. For this reason, the nawba recognition experiment includes a zip file with the necessary pre-computed files.

## Credits
This work is based on a collaboration between [*Niccolò Pretto*](http://www.dei.unipd.it/~prettoni/), *Barış Bozkurt*, *Rafael Caro Repetto* and *Xavier Serra*, as part of the project [**Musical Bridges**](https://www.upf.edu/web/musicalbridges). The notebooks use the Arab Andalusian Corpus in Dunya, created during the [**CompMusic**](http://compmusic.upf.edu/) project.

These notebooks are based on **MIR-docker-extension** (https://github.com/MTG/MIR-docker-extension).
The repository includes snippets of code and algorithms from the following repositories:
* **pycompmusic**: https://github.com/MTG/pycompmusic
* **dunya**: https://github.com/MTG/dunya
* **tomato**: https://github.com/sertansenturk/tomato
* **morty**: https://github.com/altugkarakurt/morty
* **dunya** desktop: https://github.com/MTG/dunya-desktop


## References
[1] Pretto, N; Bozkurt, B; Caro Repetto, R; and Serra, X (2018) "Nawba Recognition for Arab-Andalusian Music using Templates from Music Scores", in Proceedings of the 15th Sound and Music Computing Conference (SMC2018), (in press).
