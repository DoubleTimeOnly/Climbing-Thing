# Climbing-Thing
## Ideas
* Climb recommender. Suggest similar climbs based on input
  * Similarity based on beta, holds, ???
  * How do we determine if two climbs are similar to each other?
* Body position / beta / keypoint detection thing
  * Determine a person's beta based on a video
  * Maybe show where and how two betas differ
  * Can probably use a pretrained keypoint detector
* Climbing hold segmentation. Given an image of a climb show which holds belong to which routes
  * Shouldn't be too bad. Most annoying part is collecting data
* Recommend beta on a route based on an image
  * Use keypoint data taken from different images to learn how routes should be climbed?

## Development
### Virtual Environment
To edit the project, a virtual environment is recomended. To setup your virtual environment, while in the project directory, run:
```shell
$ python -m venv venv
```
To activate the virtual environment run:
```shell
$ source ./venv/Scripts/activate
```
Or in windows
```powershell
> ./venv/Scripts/activate.bat
```
### Installing the Package
To install the package in editable mode run:
```shell
$ python -m pip install -e . 
```
