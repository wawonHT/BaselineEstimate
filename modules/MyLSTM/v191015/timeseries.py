class TimeSeries(object) :
    def __init__(self,series=None,note="") :
        self.__series = series if series!=None else []
        self.__note   = note
    
    def Add(self,vector) :
        self.__series.append(vector)
    
    def Get(self) :
        return self.__series
    
    def At(self,time) :
        try:
            return self.__series[time]
        except IndexError :
            return None
    
    def Length(self) :
        return len(self.__series)
    
    def LengthAt(self,time) :
        try:
            return len(self.__series[time])
        except IndexError :
            return -1
    
    def Note(self) :
        return self.__note
    
    def Show(self) :
        for series in self.__series :
            for s in series :
                print "%.03f"%(s),
            print ""
        print "Length : %4d\n"%(self.Length())
        return



class LearningTimeSeries(TimeSeries) :
    def __init__(self, series=None, answer=None, note="") :
        super(LearningTimeSeries,self).__init__(series,note)
        self.__answer = answer if answer!=None else []
    
    def Add(self,vector,answer) :
        super(LearningTimeSeries,self).Add(vector)
        self.__answer.append( answer )
    
    def Get(self) :
        return super(LearningTimeSeries,self).Get(),self.__answer
    
    def AnswerLength(self) :
        return len(self.__answer)
    
    def AnswerLengthAt(self,time) :
        try:
            return len(self.__answer[time])
        except IndexError :
            return -1
    
    def Show(self) :
        super(LearningTimeSeries,self).Show()
        print "-------------------"
        for ans in self.__answer :
            for a in ans :
                print "%.03f"%(a),
            print ""
        print "answer range is",self.AnswerLength()
        return


import sys
import numpy as np


class MyLSTMTimeSeries(LearningTimeSeries) :
    def __init__(self, series=None, in_area=(0,4), out_area=(4,14), note="") :
        super(MyLSTMTimeSeries,self).__init__(series=None,answer=None,note=note)
        self.__IN_AREA  = in_area
        self.__OUT_AREA = out_area
        
        for vector in series :
            self.Add( vector )
    
    def Add(self,vector) :
        length = len(vector)
        if length < self.__OUT_AREA[1] :
            print "vector length is short : %d, need %d"%(length,self.__OUT_AREA[1])
            return
        v   = vector[ self.__IN_AREA[0]:self.__IN_AREA[1] ]
        a   = vector[ self.__OUT_AREA[0]:self.__OUT_AREA[1] ]
        super(MyLSTMTimeSeries,self).Add(v,a)
    
    def MaxMinScaling(self,max,min) :
        max_v   = np.array( max[ self.__IN_AREA[0]:self.__IN_AREA[1] ] )
        min_v   = np.array( min[ self.__IN_AREA[0]:self.__IN_AREA[1] ] )
        max_a   = np.array( max[ self.__OUT_AREA[0]:self.__OUT_AREA[1] ] )
        min_a   = np.array( min[ self.__OUT_AREA[0]:self.__OUT_AREA[1] ] )
        
        v_den   = np.array( [(x-n) if x!=n else x for (x,n) in zip(max_v,min_v)] )
        a_den   = np.array( [(x-n) if x!=n else x for (x,n) in zip(max_a,min_a)] )
        
        series,answer  = super(MyLSTMTimeSeries,self).Get()
        for t in range(len( series )) :
            series[t]    = list( (np.array(series[t])-min_v)/v_den )
            answer[t]    = list( (np.array(answer[t])-min_a)/a_den )
        
        return True


if __name__ == "__main__" :
    t = TimeSeries()
    t.Add( [1,1,1] )
    t.Add( [2,1,1] )
    t.Add( [3,1,1] )
    t.Show()
    t = TimeSeries()
    t.Add( [1,1,1] )
    t.Add( [2,1,1] )
    t.Add( [3,1,1] )
    t.Show()