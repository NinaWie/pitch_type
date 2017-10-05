from video_to_pitchtype_directly import VideoProcessor

vid = VideoProcessor(path_input="/scratch/nvw224/videos/atl", resize_width = 220, resize_height = 220)

bsp_date = '2017-06-17'

from os import listdir

for f in listdir(bsp_date):
    res_p = []
    res_b = []
    if f[-4:]==".mp4":
        print(f)
        p = vid.path_input+"/"+bsp_date+"/center\ field/"
        pitcher, batter = vid.get_pitcherAndBatter_array(p+f)
        res_p.append(pitcher)
        res_b.append(batter)

np.save(np.array("pitcher_first_move", res_p))
np.save(np.array("batter_first_move", res_b))
