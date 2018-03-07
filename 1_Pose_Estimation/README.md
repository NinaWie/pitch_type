# Pose estimation

Simple test version: video input and images with plotted skeletons as output: [here](ice_hockey.py)

### Process videos to joint coordinates (saved as json files)
* For multiple files in folders by date (for example to run on new videos), use [folder script](from_video_to_joints.py)
* for a single file, or just example files in one folder, use [file script](real_time_localize.py)

Note: The scripts works both for low quality and for high quality videos, but for the high quality videos, instead of the bounding box, the coordinates of the first hip position of the target player must be specified.

The scripts read a video frame by frame and use the pose estimation network in to output the skeletons of all detected persons. Then the target person is localized and the region of interest for the next frame is calculated. After processing all frames, the output trajectories are smoothed and interpolated.

Pose estimation, localization, smoothing and interpolating functions from [Functions](Functions.py)

A [notebook](Player localization.ipynb) is used to test localization and other pose estimation related outputs.

### Unit Tests

Run unittests by simply calling

```bash
python -m unittest discover tests
```
