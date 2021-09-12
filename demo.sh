#!/bin/bash
cd /home/guy/Projects/Seminar
source /home/guy/Softwares/anaconda3/etc/profile.d/conda.sh
conda activate vid2e
valid=true
while [ $valid ]
do
printf "What would you like to show :\n"
printf "1 : Stereo matching using SAD\n" #done
printf "2 : Show depth map\n"
printf "3 : Show saccades video\n"
printf "4 : show stream of events - polarity view\n"
printf "5 : Event based Stereo Matching using SAD\n"
printf "6 : Show event based depth map\n"
printf "7 : exit\n"
read -r answer
case $answer in
1)
  python match_finder.py &
;;
2)
  python show_depth_map.py &
;;
3)
  python show_saccades_video.py &
;;
4)
  python show_events_streams.py &
;;
5)
  python event_match_finder.py &
;;
6)
 python show_event_depth_map.py &
;;
7)
  printf "Bye Bye!"
  exit
  python vid2e/upsampling/upsample.py --input_dir vid2e/example/original/ --output_dir Data/Images/Upsampled --device cuda:0
;;
8)
  printf "____________Left events ________________\n"
  python vid2e/esim_torch/generate_events.py --input_dir=Data/Images/Upsampled/Left/ --output_dir=Data/Events/Generated/Left -cn=0.35 -cp=0.45
  printf "________________ Right events ________________\n"
  python vid2e/esim_torch/generate_events.py --input_dir=Data/Images/Upsampled/Right/ --output_dir=Data/Events/Generated/Right -cn=0.35 -cp=0.45
;;

esac
done

