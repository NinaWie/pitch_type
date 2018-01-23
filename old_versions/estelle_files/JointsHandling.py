import cv2
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
from Functions import handle_one,df_coordinates
import ast

parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_dir', metavar='DIR',
                    help='folder where merge.csv are')
parser.add_argument('output_dir', metavar='DIR',
                    help='folder where merge.csv are')
parser.add_argument('extension', metavar='DIR',
                    help='-')
# example usage: python JointsHandling.py atl/ output/ .mp4

dates = ["2017-04-14", "2017-04-18", "2017-05-02", "2017-05-06"] # , "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16", "2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07", "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17"]



for date in dates:
    args = parser.parse_args()
    input_dir= args.input_dir+"/"+date+"center\ field/"
    output_folder=args.output_dir
    ext=args.extension
    #input_dir="/scratch/ea1921/videos/2017-07-15/center_field"
    #output_folder='/scratch/ea1921/pytorch_Realtime_Multi-Person_Pose_Estimation/'
    outputdir=[]
    for f in listdir(output_folder):
        outputdir.append(str(f))

    dir_videos_dat=[]
    for f in listdir(input_dir):
        dir_videos_dat.append(input_dir+str(f))
    ##print dir_videos_dat

    if __name__ == "__main__":
        j=0
        #print output_folder
        #print outputdir
        #print len(dir_videos_dat)
        for f in dir_videos_dat:
            center_dic={}
            if f.endswith(ext):
                #print 2
                tic=time.time()
                #print '---------------------------Processing video #' +str(j)+ '---------------------------------'

                path_input_vid=f
    #                if args.output_dir+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_video_keypoints_pitcher.mov" in dir_out:
    #                    #print "already here"

    #                else:('/scratch/ea1921/pytorch_Realtime_Multi-Person_Pose_Estimation/'
    #                    #print 'start with j = ' + str(j)
                if path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_df.csv" not in outputdir:
                    path_input_dat=f+'.dat'
                    #print 3
                    #print path_input_dat
                    if path_input_dat in dir_videos_dat:
                        #print 4
                        video_capture = cv2.VideoCapture(path_input_vid)
                        for i in open(path_input_dat).readlines():
                            #print i
                            datContent=ast.literal_eval(i)
                        ##print dic

                        ##print dic.keys()
             #           ret, frame = video_capture.read()
              #          fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                        #datContent = [i.strip().split() for i in open(path_input_dat).readlines()]
                        #datContent=ast.literal_eval(datContent[0][0])
                        bottom_p=datContent['Pitcher']['bottom']
                        left_p=datContent['Pitcher']['left']
                        right_p=datContent['Pitcher']['right']
                        top_p=datContent['Pitcher']['top']
                        bottom_b=datContent['Batter']['bottom']
                        left_b=datContent['Batter']['left']
                        right_b=datContent['Batter']['right']
                        top_b=datContent['Batter']['top']
                        center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])
                        center_dic['Batter']=np.array([abs(top_b-bottom_b)/2., abs(left_b-right_b)/2.])
                 #       NEWVID_pitcher = cv2.VideoWriter(output_folder+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_video_keypoints_pitcher.mov", fourcc, 30 , (frame[top_p:bottom_p, left_p:right_p].shape[1], frame[top_p:bottom_p, left_p:right_p].shape[0]),1) #2 frames / sec
                #        NEWVID_batter = cv2.VideoWriter(output_folder+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_video_keypoints_batter.mov", fourcc, 30 ,(frame[top_b:bottom_b, left_b:right_b].shape[1], frame[top_b:bottom_b, left_b:right_b].shape[0]),1) #2 frames / sec


                        df = pd.DataFrame(columns=['Frame','Pitcher','Batter'])
                        p=0
                        while True:
                    # Capture frame-by-frame

                            ret, frame = video_capture.read()
                            if frame == None:
                                break


                            pitcher = frame[top_p:bottom_p, left_p:right_p]

                            batter = frame[top_b:bottom_b, left_b:right_b]

             #               canvas_pitcher,coordinates_pitcher2=handle_one(pitcher)
              #              canvas_batter,coordinates_batter2= handle_one(batter)
               #             NEWVID_pitcher.write(canvas_pitcher)
                #            NEWVID_batter.write(canvas_batter)
               #             coordinates_pitcher2=handle_one(pitcher)
               #             coordinates_batter2= handle_one(batter)

                    #title_pitcher=path_output+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+'_pitcher'+str(j)
                    #title_batter=path_output+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+'_batter'+str(j)

                 #           coordinates_batter[key]=coordinates_batter2.tolist()
                 #           coordinates_pitcher[key]=coordinates_pitcher2.tolist()
                            df.loc[p]=[int(p),handle_one(pitcher),handle_one(batter) ]
                            #print 'frame'+str(p)

                            #print '-------------------------------------------------------------------------------'

                            p+=1
                        toc=time.time()
                        #df.to_csv(output_folder+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_temp.csv")

                        #print f
                        try:
                            df_res=df_coordinates(df,center_dic)

                            df_res.loc[p]=["# of last frame",p-1,p-1]
                            if 'first_movement_frame_index' in datContent.keys():

                                df_res.loc[p+1]=["first_movement_frame_index",datContent['first_movement_frame_index'],datContent['first_movement_frame_index']]
                            else:
                                 df_res.loc[p+1]=["first_movement_frame_index",'NA','NA']

                            if 'first_movement_frame_offset_in_msec' in datContent.keys():
                                df_res.loc[p+2]=["first_movement_frame_offset_in_msec",datContent['first_movement_frame_offset_in_msec'],datContent['first_movement_frame_offset_in_msec']]
                            else:
                                df_res.loc[p+2]=["first_movement_frame_offset_in_msec",'NA','NA']
                            if "pitch_frame_index" in datContent.keys():
                                df_res.loc[p+3]=["pitch_frame_index",datContent['pitch_frame_index'],datContent['pitch_frame_index']]
                            else:
                                df_res.loc[p+3]=["pitch_frame_index",'NA','NA']
                            if "pitch_frame_offset_in_msec" in datContent.keys():
                                df_res.loc[p+4]=["pitch_frame_offset_in_msec",datContent['pitch_frame_offset_in_msec'],datContent['pitch_frame_offset_in_msec']]
                            else:
                                df_res.loc[p+4]=["pitch_frame_offset_in_msec",'NA','NA']
                            if "strike_zone_frame_url" in datContent.keys():

                                df_res.loc[p+5]=["strike_zone_frame_url",datContent['strike_zone_frame_url'],datContent['strike_zone_frame_url']]
                            else:
                                df_res.loc[p+5]=["strike_zone_frame_url",'NA','NA']
                            df_res.loc[p+6] =["center ROIs",center_dic['Pitcher'],center_dic['Batter']]
                            df_res.loc[p+7]=["Game",path_input_vid[path_input_vid.rfind('/')+1:][:-4],path_input_vid[path_input_vid.rfind('/')+1:][:-4]]
                            df_res.loc[p+8]=['Player','Pitcher','Batter']
                            #print 5
                            df_res.to_csv(output_folder+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_df.csv")
                        except (UnboundLocalError,IndexError) as e:
                            df.to_csv(output_folder+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_temp.csv")

                            continue
                #title_pitcher=path_output+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+'_pitcher'+str(j)


            #title_batter=path_output+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+'_batter'+str(j)
                 #       NEWVID_pitcher.release()
                 #       NEWVID_batter.release()
            #            video_capture.release()
            #            with open(output_folder+path_input_vid[path_input_vid.rfind('/')+1:][:-4]+"_df.csv", 'wb') as csv_file:
                        toctoc=time.time()
            #                writer = csv.writer(csv_file)
            #                writer.writerow(['Frame','Pitcher_Crop','Batter_Crop'])
                    #writer.writerow(['ROI :[ Bottom, Top, Left, Right ]',[bottom_p,top_p,left_p,right_p],[bottom_b,top_b,left_b,right_b]])
            #                for key in coordinates_pitcher.iterkeys():
             #                   writer.writerow([key, coordinates_pitcher[key],coordinates_batter[key]])
            #


                        #print 'time spent for finding joints= ' ,toc - tic
                        #print 'time spent for handling df= ', toctoc-toc
