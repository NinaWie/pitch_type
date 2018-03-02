# Pose estimation

### Process videos to joint coordinates
* for multiple files in folders by date (for example to run on new videos), use [folder script](from_video_to_joints.py)
* for a single file, or just example files in one folder, use [file script](from_video_to_joints.py)
  * works both for low quality and for high quality videos, only other input: coordinate of center of target person in the first frame
  * for old videos however, a bounding box for the first frame is recommended
* Uses localization, smoothing and interpolating functions from [Functions](Functions.py)

### Unit Tests

Run unittests by simply calling

```bash
python -m unittest discover tests
```
