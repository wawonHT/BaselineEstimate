class Logger(object) :
    def __init__(self,enable=True) :
        self.__log = []
        self.__enable = enable
    
    @property
    def IsEnable(self) :
        return self.__enable
    
    def Logging(self,val) :
        if self.IsEnable :
            self.__log.append( val )
    
    def Output( self,file_path ) :
        if self.IsEnable :
            fout = open( file_path,'w' )
            for l in self.__log :
                fout.write(l+"\n")
            fout.close()
    
    def Clear(self) :
        if self.IsEnable :
            self.__log = []
    
    def Enable(self) :
        self.__enable = True
    
    def Disable(self) :
        self.__enable = False
