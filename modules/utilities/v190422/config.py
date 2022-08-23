

import os
import xml.etree.ElementTree as ET
import copy
from datetime import datetime,timedelta

class CConfig :
    def __init__( self,ifp,target=None ) :
        self.ifp  = ifp
        self.root = None
        self.__read(target)
    
    def __read(self,target=None):
        self.root = ET.parse(self.ifp).getroot()
        if not target is None :
            text    = self.root.find("target").text
            if not text is None and text != target :
                print "!!!!!     <target> not match : source->",target,"config->",text,"     !!!!!"
        return self
    
    def get(self) :
        return CElement( self.root )
    
    @staticmethod
    def create(key,attr={}) :
        elm = ET.Element(key)
        for key in attr : elm.set(key,attr[key])
        return CElement( elm )

class CElement :
    def __init__(self,elm) :
        self.__element  = elm
        self.__list     = list(elm)
        self.__tags     = []
        for e in self.__list : self.__tags.append( e.tag )
    
    def __iadd__(self,elm) :
        if not isinstance(elm,CElement) :
            print "elm is not CElement"
            return self
        
        for e in elm.list() :
            if self.__element.find(e.tag) is None :
                self.__element.append( e )
                #print str(e.tag),"is not exist, append!"
            else :
                #print str(e.tag),"is exist in self"
                self.__itrappend( e, self.__element.findall(e.tag) )
            self.__update()
        
        return self
    
    def __itrappend(self,e_from,e_tos) :
        for e_to in e_tos :
            #print "e_to keys are",list(e_to)
            for ef_child in list(e_from) :
                #print "ef_child.tag is ",ef_child.tag
                if e_to.find(ef_child.tag) is not None :
                    #print str(ef_child.tag),"is exist in",e_to.tag
                    self.__itrappend( ef_child, e_to.findall(ef_child.tag) )
                else :
                    e_to.append(ef_child)
                    #print str(ef_child.tag),"is not exist, append! (child)"
                    #self.show()
                self.__update()
    
    def __update(self) :
        self.__list     = list(self.__element)
        self.__tags     = []
        for e in self.__list : self.__tags.append( e.tag )
    
    def __trans( self,elm ) :
        form = elm.get( "form" )
        var  = elm.text
        try:
            if form == None :
                return var
            elif form == "int":
                return int(var)
            elif form == "float":
                return float(var)
            elif form == "string":
                return str(var)
            elif form == "bool":
                return False if var.lower()=="false" else True
            elif form == "time":
                return datetime.strptime(var,'%H:%M:%S')
            elif form == "date":
                return datetime.strptime(var,'%Y%m%d')
            elif form == "dtime":
                t = datetime.strptime(var,'%H:%M:%S')
                return timedelta( hours=t.hour, minutes=t.minute, seconds=t.second )
            elif form == "dday":
                return timedelta( days=int(var) )
            elif form == "sarray":
                return var.split(" ")
            elif form == "iarray":
                items = var.split(" ")
                ret = []
                for i in items : ret.append( int(i) )
                return ret
            elif form == "farray":
                items = var.split(" ")
                ret = []
                for i in items : ret.append( float(i) )
                return ret
            elif form == "tarray":
                items = var.split(" ")
                ret = []
                for i in items : ret.append( datetime.strptime(i,'%H:%M:%S') )
                return ret
            elif form == "darray":
                items = var.split(" ")
                ret = []
                for i in items : ret.append( datetime.strptime(i,'%Y%m%d') )
                return ret
            elif form == "path":
                return os.path.normpath( str(var) )
            else :
                return None
        except ValueError:
            return None
        except TypeError:
            return None
    
    
    def find(self,url,default=None) :
        found_elm   = self.__element.find(url)
        if found_elm is None :
            return None
        else :
            return CElement(found_elm)
    
    def update(self,url,txt=None) :
        found_elm   = self.__element.find(url)
        if found_elm is not None :
            found_elm.text = txt
        return self
    
    
    def get(self,key,attr={},default=None) :
        ret = self.gets( key,attr,default )
        if ret is None :
            return default
        
        if len(ret) is not 0 :
            return ret[0]
        else :
            return default
    
    def gets(self,key,attr={},default=None) :
        if not self.has_key(key) :
            return [default] if default is not None else []
        
        elms = self.__element.findall(key)
        tmp = None
        
        if   len(elms) == 0  :   # key not found
            return None
        elif len(attr) == 0 :   # key not found
            tmp = elms
        else :
            tmp = []
            for e in elms :
                flg = True
                for k in attr :
                    flg &= e.get(k)==attr[k]
                if flg :
                    tmp.append(e)
        
        if tmp == None : return None
        
        ret = []
        for t in tmp :
            if len(t.getchildren()) is not 0 :
                ret.append( CElement(t) )
            else :
                ret.append( self.__trans(t) )
        return ret
    
    def has_key(self,key) :
        return key in self.__tags
    
    def keys(self) :
        return self.__tags
    
    def list(self) :
        return self.__list
        
    def element(self) :
        return self.__element
    
    def attribute(self,attr) :
        return self.__element.get(attr)
    
    def items(self) :
        return self.__element.items()
    
    def add( self,key,val,attr={} ) :
        elm = None
        if isinstance(val,ET.Element) :
            elm = val
        elif isinstance(val,CElement) :
            elm = val.element()
        else :
            elm = ET.Element(key)
            elm.text = val
        for key in attr : elm.set(key,attr[key])
        self.__element.append( elm )
        self.__update()
        return self
    
    def set( self,key,val,attr={} ) :
        elm = self.__element.findall(key)
        
        if len(elm) == 0 :
            self.add( key,val,attr )
            return self
        
        obj = None
        if len(attr) == 0 :
            obj = elm[0]
        else :
            for e in elm :
                if obj != None : break
                flg = True
                for k in attr :
                    flg &= e.get(k)==attr[k]
                if flg :
                    obj = e
                    break
        if obj != None :
            if isinstance(val,ET.Element) :
                self.__element.remove(obj)
                self.__element.append(val)
            elif isinstance(val,CElement) :
                self.__element.remove(obj)
                self.__element.append(val.element())
            else :
                obj.text = val
            self.__update()
        else :
            self.add(key,val,attr)
        return self
    
    def has_child(self) :
        return len( self.__element.getchildren() ) is not 0
    
    def show(self,indent="") :
        print indent+"######"
        print indent+"Element :", self.__element.tag
        print indent+"attributes : ", self.items()
        print indent+"elements : "
        for elm in list(self.__element) :
            e = self.get( elm.tag,elm.attrib )
            if not isinstance(e,CElement) : print indent+" ",elm.tag,":",e
            else : e.show("  "+indent )
        print indent+"######"
