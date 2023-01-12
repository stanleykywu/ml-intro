# Example code for Stanley Wu's Intro to ML (NLP) Workshop

## Running Locally
After cloning, please make sure to have the Anaconda or Miniconda package management software installed.

Local environments and common ML packages are tricky when there are M1, M2, Intel, ... etc. processors out there that support different things. This process has been tested on the following systems and are known to work:
- Apple Silicon Macs
### Create Conda Environment
1. `conda update conda -y`
2. `conda create --name ml-workshop python=3.8`
3. `conda activate ml-workshop`
4. `conda install --file requirements.txt`

If step 4 fails, try the pip variant: `pip install -r requirements.txt`. Otherwise, the Google Colab Online Mirror option below is a safe alternative.

## Google Colab Online Mirror
If you are having trouble getting this repository up and running, the jupyter notebook code is also hosted [here](https://colab.research.google.com/drive/1f5RgezXaV30o1ByGVjEUUKr-Sa0Wci4q?usp=sharing) on Google Colab. Make a copy and you are free to run the code without having to install anything on your local machine.

_This code is adapted from [James McCaffrey's NLP Blog Post]([https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html](https://jamesmccaffrey.wordpress.com/2021/09/22/natural-language-question-answering-using-hugging-face/)_
