# EDM-subgenre-classifier
This repository contains the code for "[Deep Learning Based EDM Subgenre Classification using Mel-Spectrogram and Tempogram Features](https://arxiv.org/abs/2110.08862)" arXiv:2110.08862, 2021. Wei-Han Hsu, Bo-Yu Chen, Yi-Hsuan Yang

## Prerequisites

* Python  == 3.6.13
* librosa == 0.8.1
* torch   == 1.3.0
* numpy   == 1.16.0
* pandas  == 1.1.2

## Running the application

You could run
  <pre><code>pip3 install -r requirements.txt</code></pre>
to make the part of installing dependencies.

Then run
  <pre><code>python3 main.py</code></pre>
could predict the song's genre directly.

Here are the steps to run the code:
* Step 1 : Preparing the audio (mp3, wav) and put it under "./data/audio/"
* Step 2 : Extracting the feature, and the feature will under the ./data/{feature_folder}" (feature_folder : mel-spectrogram, fourier-tempogram and auto-tempogram)
* Step 3 : Classifying the audio by feature
* Step 4 : The result will be in the "./result.csv"
* All the "step" could complete by main.py

We only put the "late-fusion model" in this repo.

## Note
This project was revised from [sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models).
Many thanks for all authors of the paper ("Evaluation of CNN-based Automatic Music Tagging Models", SMC 2020)

## License
<pre><code>
MIT License

Copyright (c) 2020 Music Technology Group, Universitat Pompeu Fabra. Code developed by Minz Won.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</code></pre>

## Contact
Please feel free to contact [Wei-Han Hsu](https://github.com/ddman1101) if you have any questions.
