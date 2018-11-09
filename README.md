# DenseFlow
For Two-Stream code, create it

This is a python port of denseflow, which extract the videos' frames and **optical flow images** with **TVL1 algorithm** as default.

---

### Requirements:
- numpy
- cv2
- PIL.Image
- multiprocess
- scikit-video (optional)
- scipy

## Installation
#### Install the requirements:
```
pip install -r requirements.txt

```

---

## Usage
The denseflow.py contains two modes including '**run**' and '**debug**'.


here 'debug' is built for debugging the video paths and video-read methods. ([IPython.embed](http://ipython.org/ipython-doc/dev/interactive/reference.html#embedding) suggested)

Just simply run the following code:

```
python my_denseflow_v1.py --data_root /path/to/dataset --dataset DatasetName --num_workers 1 --step 1 --bound 20 --mode debug

```
While in 'run' mode, here we provide multi-process as well as multi-server with manually s_/e_ IDs setting.

for example:  server 0 need to process 3000 videos with 4 processes parallelly working:

```
python my_denseflow_v1.py --data_root /path/to/dataset --dataset DatasetName --num_workers 4 --step 1 --bound 20 --mode run
```

---

Just feel free to let me know if any bugs exist.

This code can process dataset like UCF-101，HMDB-51 and etc.

This code refrence to this [link](https://github.com/qijiezhao/py-denseflow.git), and make some change.
