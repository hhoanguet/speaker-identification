# Speaker-Identification using Gaussian Mixture Models (GMM)

Speaker identification system can automatically identify the person speaking in an audio file given a group of predefined speakers. For testing, an unseen audio is compared against the provided group of speakers, and in the case there is a match found. The speaker's identity is returned.
## Installation
1. Create a new environment with **python 3.6** version because python 3.7 or higher does not support **pyaudio** package.
``` 
conda create -n test python=3.6
conda activate test 
```

2. Clone this project:

```
git clone https://github.com/hhoanguet/speaker-identification.git
cd speaker-identification
```

3. Install all dependencies in requirements.txt, it may take a few minutes.

`pip install -r requirements.txt`

4. Our speaker identification app is ready to run:

`streamlit run app.py`

## Project report

You can download our project report from [here](Project_report.pdf)

## Team member:
**Hoang Huy Nguyen**

Github: [hhoanguet](https://github.com/hhoanguet)

**Hoang Xuan Nguyen**

Github: [hoangngx](https://github.com/hoangngx)
