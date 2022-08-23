# -*- coding: utf-8 -*-

import os
import sys
import shutil
from datetime import datetime,timedelta
import random
import numpy as np

sys.path.append('./modules')

from MyLSTM.v191015.bl_lstm     import BL_LSTM
from MyLSTM.v191015.timeseries  import MyLSTMTimeSeries
from utilities.v190422.config   import CConfig


if __name__ == "__main__" :
    
    root    = CConfig(sys.argv[1],target="execute_test").get()
    
    INPUT_FILE_PATH         = root.get("inputFilePath",default=".")
    OUTPUT_DIRECTORY_ROOT   = root.get("outputDirectoryRoot",default=".")
    IN_VARIABLES            = root.get("inputVariables",default=[])
    OUT_VARIABLES           = root.get("outputVariables",default=[])
    HIDDEN_UNITS            = root.get("hiddenUnits",default=10)
    SERIES_LENGTH           = root.get("seriesLength",default=6)
    EPOCH                   = root.get("epoch",default=100)
    RANDOM_SEED             = root.get("randomSeed",default=0)
    MAX                     = root.get("max",default=[])
    MIN                     = root.get("min",default=[])
    TRAINING_DATES          = root.get("trainingDates",default=[])
    TEST_DATES              = root.get("testDates",default=[])
    IGNORE_HOURS            = root.gets("ignoreHours",default=[])
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    
    # 出力フォルダー作成
    OUTPUT_DIRECTORY_PATH   = OUTPUT_DIRECTORY_ROOT + "/"+os.path.basename(sys.argv[1]).split(".")[0]
    if os.path.exists( OUTPUT_DIRECTORY_PATH ) :
        shutil.rmtree( OUTPUT_DIRECTORY_PATH )
    os.mkdir( OUTPUT_DIRECTORY_PATH )
    shutil.copy( sys.argv[1], OUTPUT_DIRECTORY_PATH+"/"+os.path.basename(sys.argv[1]) )
    
    
    # 訓練データと評価データを読み込む
    with open( INPUT_FILE_PATH,'r' ) as fin :
        columns = fin.readline().strip().split(",")
        
        data    = dict(zip( columns,[[] for c in columns] ))
        
        for line in fin :
            items   = line.strip().split(",")
            for (c,clm) in enumerate(columns) :
                try :
                    data[clm].append( float(items[c]) )
                except ValueError :
                    data[clm].append( items[c] )
    
    print "success to loading"
    
    """
    with open( OUTPUT_DIRECTORY_PATH + "/nomalized_data.csv",'w' ) as fout :
        fout.write( "TIME," + ",".join(IN_VARIABLES + OUT_VARIABLES) + "\n" )
        for i in range(len(data["TIME"])) :
            fout.write( "%s"%(data["TIME"][i]) )
            for (j,valiable) in enumerate(IN_VARIABLES + OUT_VARIABLES) :
                normalized = (data[valiable][i] - MIN[j]) / (MAX[j] -MIN[j])
                fout.write( ",%.08f"%(normalized) )
            fout.write( "\n" )
    
    with open( OUTPUT_DIRECTORY_PATH + "/data.csv",'w' ) as fout :
        fout.write( "TIME," + ",".join(IN_VARIABLES + OUT_VARIABLES) + "\n" )
        for i in range(len(data["TIME"])) :
            fout.write( "%s"%(data["TIME"][i]) )
            for valiable in (IN_VARIABLES + OUT_VARIABLES) :
                fout.write( ",%.08f"%(data[valiable][i]) )
            fout.write( "\n" )
    exit()
    """
    
    # LSTM入力用時系列データ作成
    sequence_data   = dict()
    for (i,t) in enumerate( data["TIME"] ):
        sequence_data[t] = [ data[v][i] for v in IN_VARIABLES+OUT_VARIABLES ]
    train_series    = dict()
    test_series     = dict()
    
    with open( OUTPUT_DIRECTORY_PATH+"/train_series.csv",'w' ) as fout :
        fout.write( "seriesID,target_frame\n" )
    
    with open( OUTPUT_DIRECTORY_PATH+"/test_series.csv",'w' ) as fout :
        fout.write( "seriesID,target_frame\n" )
    
    for i in range( len(sequence_data) ) :
        dt_tmp      = datetime.strptime( data["TIME"][i], "%Y-%m-%d %H:%M:%S" )
        series_dts  = [ datetime.strftime( dt_tmp+timedelta(minutes=d*5), "%Y-%m-%d %H:%M:%S" ) for d in range(-SERIES_LENGTH+1,1) ]
        
        if True in [ h.time()<=dt_tmp.time() and dt_tmp.time()<t.time() for (h,t) in IGNORE_HOURS ] :
            continue
        
        sequences   = [ sequence_data[t] if t in sequence_data else None for t in series_dts ]
        if None in sequences :
            continue
        
        series  = MyLSTMTimeSeries( sequences, in_area=(0,len(IN_VARIABLES)), out_area=(len(IN_VARIABLES), len(IN_VARIABLES)+len(OUT_VARIABLES)) )
        series.MaxMinScaling( MAX,MIN )
        
        if True in [ d.date()==dt_tmp.date() for d in TRAINING_DATES ] :
            train_series[data["TIME"][i]] = series
            with open( OUTPUT_DIRECTORY_PATH+"/train_series.csv",'a' ) as fout :
                fout.write( "%d,"%(len(train_series))+",".join(series_dts)+"\n" )
        elif True in [ d.date()==dt_tmp.date() for d in TEST_DATES ] :
            test_series[data["TIME"][i]] = series
            with open( OUTPUT_DIRECTORY_PATH+"/test_series.csv",'a' ) as fout :
                fout.write( "%d,"%(len(test_series))+",".join(series_dts)+"\n" )
    
    print "success to create time series, start learning..."
    
    
    # LSTM学習
    lstm    = BL_LSTM( in_units=len(IN_VARIABLES), hidden_units=HIDDEN_UNITS, out_units=len(OUT_VARIABLES), seed=RANDOM_SEED )
    
    shuffled_keys   = random.sample( train_series.keys(), len(train_series.keys()) )
    shuffled_vals   = [ train_series[k] for k in shuffled_keys ]
    
    with open( OUTPUT_DIRECTORY_PATH+"/loss.csv",'w' ) as fout :
        for epoch in range(EPOCH) :
            total_avg_loss  = lstm.Train( shuffled_vals, epoch, OUTPUT_DIRECTORY_PATH)
            print epoch,total_avg_loss
            fout.write( "%d,%.08f\n"%(epoch,total_avg_loss) )
    
    predicted   = lstm.Predict( shuffled_vals, OUTPUT_DIRECTORY_PATH+"/train_val.csv" )
    with open( OUTPUT_DIRECTORY_PATH+"/train_result_W15.csv",'w' ) as fout :
        fout.write( "id,time,predicted,answer,m_predicted,m_answer\n" )
        
        errors      = []
        m_errors    = []
        al      = MAX[(IN_VARIABLES+OUT_VARIABLES).index("W15n")] - MIN[(IN_VARIABLES+OUT_VARIABLES).index("W15n")]
        bt      = MIN[(IN_VARIABLES+OUT_VARIABLES).index("W15n")]
        
        for (i,p,a) in predicted["result"] :
            print i,p,a
            fout.write( "%d,%s,%.08f,%.08f,%.08f,%.08f\n"%(i,shuffled_keys[i],p[0],a[0],p[0]*al+bt,a[0]*al+bt) )
            errors.append( (p[0]-a[0])**2 )
            m_errors.append( ((p[0]*al+bt)-(a[0]*al+bt))**2 )
        RMSE    = np.sqrt(np.mean( errors ))
        mRMSE   = np.sqrt(np.mean( m_errors ))
        fout.write( "\n\nRMSE,%.08f\nmRMSE[kWh],%.08f\n"%(RMSE,mRMSE) )
    
    with open( OUTPUT_DIRECTORY_PATH+"/train_result_TSA15.csv",'w' ) as fout :
        fout.write( "id,time,predicted,answer,m_predicted,m_answer\n" )
        
        errors      = []
        m_errors    = []
        al      = MAX[(IN_VARIABLES+OUT_VARIABLES).index("TSA15n")] - MIN[(IN_VARIABLES+OUT_VARIABLES).index("TSA15n")]
        bt      = MIN[(IN_VARIABLES+OUT_VARIABLES).index("TSA15n")]
        
        for (i,p,a) in predicted["result"] :
            print i,p,a
            fout.write( "%d,%s,%.08f,%.08f,%.08f,%.08f\n"%(i,shuffled_keys[i],p[1],a[1],p[1]*al+bt,a[1]*al+bt) )
            errors.append( (p[1]-a[1])**2 )
            m_errors.append( ((p[1]*al+bt)-(a[1]*al+bt))**2 )
        RMSE    = np.sqrt(np.mean( errors ))
        mRMSE   = np.sqrt(np.mean( m_errors ))
        fout.write( "\n\nRMSE,%.08f\nmRMSE[kWh],%.08f\n"%(RMSE,mRMSE) )
    
    print "end learning, start evaluating..."
    
    
    # LSTM評価
    test_keys       = [ key for key in sorted(test_series.keys()) if datetime.strptime(key,"%Y-%m-%d %H:%M:%S").minute%15==14 ]
    shuffled_keys   = test_keys
    shuffled_vals   = [ test_series[k] for k in test_keys ]
    
    predicted   = lstm.Predict( shuffled_vals, OUTPUT_DIRECTORY_PATH+"/test_val.csv" )
    
    with open( OUTPUT_DIRECTORY_PATH+"/test_result_W15.csv",'w' ) as fout :
        fout.write( "id,time,predicted,answer,m_predicted,m_answer\n" )
        
        errors      = []
        m_errors    = []
        al      = MAX[(IN_VARIABLES+OUT_VARIABLES).index("W15n")] - MIN[(IN_VARIABLES+OUT_VARIABLES).index("W15n")]
        bt      = MIN[(IN_VARIABLES+OUT_VARIABLES).index("W15n")]
        
        for (i,p,a) in predicted["result"] :
            print i,p,a
            fout.write( "%d,%s,%.08f,%.08f,%.08f,%.08f\n"%(i,shuffled_keys[i],p[0],a[0],p[0]*al+bt,a[0]*al+bt) )
            errors.append( (p[0]-a[0])**2 )
            m_errors.append( ((p[0]*al+bt)-(a[0]*al+bt))**2 )
        RMSE    = np.sqrt(np.mean( errors ))
        mRMSE   = np.sqrt(np.mean( m_errors ))
        fout.write( "\n\nRMSE,%.08f\nmRMSE[kWh],%.08f\n"%(RMSE,mRMSE) )
    
    with open( OUTPUT_DIRECTORY_PATH+"/test_result_TSA15.csv",'w' ) as fout :
        fout.write( "id,time,predicted,answer,m_predicted,m_answer\n" )
        
        errors      = []
        m_errors    = []
        al      = MAX[(IN_VARIABLES+OUT_VARIABLES).index("TSA15n")] - MIN[(IN_VARIABLES+OUT_VARIABLES).index("TSA15n")]
        bt      = MIN[(IN_VARIABLES+OUT_VARIABLES).index("TSA15n")]
        
        for (i,p,a) in predicted["result"] :
            print i,p,a
            fout.write( "%d,%s,%.08f,%.08f,%.08f,%.08f\n"%(i,shuffled_keys[i],p[1],a[1],p[1]*al+bt,a[1]*al+bt) )
            errors.append( (p[1]-a[1])**2 )
            m_errors.append( ((p[1]*al+bt)-(a[1]*al+bt))**2 )
        RMSE    = np.sqrt(np.mean( errors ))
        mRMSE   = np.sqrt(np.mean( m_errors ))
        fout.write( "\n\nRMSE,%.08f\nmRMSE[kWh],%.08f\n"%(RMSE,mRMSE) )
    
    print "success, exit"