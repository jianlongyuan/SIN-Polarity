# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:08:24 2022

@author: JIANLONGYUAN_DESKTOP
"""

import os


os.system( "python run.py --mode=pred --model_dir=model/190703-214543\
                --data_dir=demo_Yangbi/mseed --data_list=demo_Yangbi/fname.csv\
                --output_dir=output --batch_size=1 --input_mseed\
                --plot_figure --save_result --input_length=3000" )

# os.system( "python run.py --mode=train --train_dir=dataset/waveform_train \
#               --train_list=dataset/waveform.csv --batch_size=20" )

# os.system( "python run.py --mode=valid --model_dir=model/190703-214543 --data_dir=dataset/waveform_train \
#                 --data_list=dataset/waveform.csv --plot_figure --save_result --batch_size=20 " )


# os.system( 'python run.py --mode=train --model_dir=model/190703-214543 \
#               --train_dir=dataset/waveform_train --train_list=dataset/waveform.csv --batch_size=20' )