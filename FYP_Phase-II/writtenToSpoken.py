
# coding: utf-8

# # Text Processing functions
# ## for generating normalised output for text of the following classes:
# 
# 1. Cardinal
# 2. Digit
# 3. Ordinal
# 4. Letters
# 5. Address
# 6. Telephone
# 7.  Electronic
# 8. Fractions
# 9. Money
# 
# The idea is to first create a dictionary of all the training input strings and their corresponding normalised text. For normalising the test data, we first look it up in the dictionary and return the corresponding 'after' value. If the string is not in the dictionary, we use these functions to generate normalised text.

# ## Import modules and data sets

# In[108]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import inflect
from num2words import num2words 
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from datetime import datetime
p = inflect.engine()

import string
# Any results you write to the current directory are saved as output.


# ## CARDINAL

# In[109]:


def cardinal(x):
    try:
        if re.match('.*[A-Za-z]+.*', x):
            return x
        x = re.sub(',', '', x, count = 10)

        if(re.match('.+\..*', x)):
            x = p.number_to_words(float(x))
        elif re.match('\..*', x): 
            x = p.number_to_words(float(x))
            x = x.replace('zero ', '', 1)
        else:
            x = p.number_to_words(int(x))
        x = x.replace('zero', 'o')    
        x = re.sub('-', ' ', x, count=10)
        x = re.sub(' and','',x, count = 10)
        return x
    except:
        return x


# In[110]:


def is_num(key):
    if is_float(key) or re.match(r'^-?[0-9]\d*?$', key.replace(',','')): return True
    else: return False

def is_float(string):
    try:
        return float(string.replace(',','')) and "." in string # True if string is a number contains a dot
    except ValueError: 
        return False
    
def num2word(key):
    bag_res = bag2word(key, digit_trained)
    if bag_res != key: return bag_res
    if re.match(r'^-?\d+$', key.replace(',','')):
        return cardinal(key)
    if is_float(key):
        return float2word(key)


# # Measure

dict_m = {'"': 'inches',
          "'": 'feet',
          'km/s': 'kilometers per second',
          'AU': 'units', 'BAR': 'bars',
          'CM': 'centimeters',
          'mm': 'millimeters',
          'FT': 'feet',
          'G': 'grams', 
          'GAL': 'gallons',
          'GB': 'gigabytes',
          'GHZ': 'gigahertz',
          'HA': 'hectares',
          'HP': 'horsepower', 
          'HZ': 'hertz',
          'KM':'kilometers',
          'km3': 'cubic kilometers',
          'KA':'kilo amperes',
          'KB': 'kilobytes',
          'KG': 'kilograms',
          'KHZ': 'kilohertz',
          'KM²': 'square kilometers',
          'KT': 'knots',
          'KV': 'kilo volts',
          'M': 'meters',
          'KM2': 'square kilometers',
          'Kw':'kilowatts',
          'KWH': 'kilo watt hours',
          'LB': 'pounds',
          'LBS': 'pounds',
          'MA': 'mega amperes',
          'MB': 'megabytes',
          'KW': 'kilowatts',
          'MPH': 'miles per hour',
          'MS': 'milliseconds',
          'MV': 'milli volts',
          'kJ':'kilojoules',
          'km/h': 'kilometers per hour',
          'V': 'volts',
          '%':'percent',
          'M2': 'square meters', 'M3': 'cubic meters', 'MW': 'megawatts', 'M²': 'square meters', 'M³': 'cubic meters', 'OZ': 'ounces',  'MHZ': 'megahertz', 'MI': 'miles',
     'MB/S': 'megabytes per second', 'MG': 'milligrams', 'ML': 'milliliters', 'YD': 'yards', 'au': 'units', 'bar': 'bars', 'cm': 'centimeters', 'ft': 'feet', 'g': 'grams', 
     'gal': 'gallons', 'gb': 'gigabytes', 'ghz': 'gigahertz', 'ha': 'hectares', 'hp': 'horsepower', 'hz': 'hertz', 'kWh': 'kilo watt hours', 'ka': 'kilo amperes', 'kb': 'kilobytes', 
     'kg': 'kilograms', 'khz': 'kilohertz', 'km': 'kilometers', 'km2': 'square kilometers', 'km²': 'square kilometers', 'kt': 'knots','kv': 'kilo volts', 'kw': 'kilowatts', 
     'lb': 'pounds', 'lbs': 'pounds', 'm': 'meters', 'm2': 'square meters','m3': 'cubic meters', 'ma': 'mega amperes', 'mb': 'megabytes', 'mb/s': 'megabytes per second', 
     'mg': 'milligrams', 'mhz': 'megahertz', 'mi': 'miles', 'ml': 'milliliters', 'mph': 'miles per hour','ms': 'milliseconds', 'mv': 'milli volts', 'mw': 'megawatts', 'm²': 'square meters',
     'm³': 'cubic meters', 'oz': 'ounces', 'v': 'volts', 'yd': 'yards', 'µg': 'micrograms', 'ΜG': 'micrograms', 'kg/m3': 'kilograms per meter cube'}

def measure(key):
    if "%" in key:
        unit = "percent";
        val = key[:len(key)-1]
    elif "/" in key and key.split("/")[0].replace(".","").isdigit():
        try:
            unit = "per " + dict_m[key.split("/")[-1]]
        except KeyError:
            unit = "per " + key.split("/")[-1].lower()
        
        val = key.split("/")[0]
    else:
        v = key.split()
        if len(v)>2:
            try:
                unit = " ".join(v[1:-1])+" "+dict_m[v[-1]]
            except KeyError:
                unit = " ".join(v[1:-1])+" "+v[-1].lower()
        else:
            try:
                unit = dict_m[v[-1]]
            except KeyError:
                unit = v[-1].lower()
        val = v[0]
    if is_num(val):
        val = p.number_to_words(val,andword='').replace("-"," ").replace(',','')
        text = val + ' ' + unit
    else: text = key
    return text


# # Verbatim

def verbatim(key):

    dict_verb = {"#":"number","&":"and","α":"alpha","Α":"alpha","β":"beta","Β":"beta","γ":"gamma","Γ":"gamma",
                 "δ":"delta","Δ":"delta","ε":"epsilon","Ε":"epsilon","Ζ":"zeta","ζ":"zeta","η":"eta","Η":"eta",
                 "θ":"theta","Θ":"theta","ι":"iota","Ι":"iota","κ":"kappa","Κ":"kappa","λ":"lambda","Λ":"lambda",
                 "Μ":"mu","μ":"mu","ν":"nu","Ν":"nu","Ξ":"xi","ξ":"xi","Ο":"omicron","ο":"omicron","π":"pi","Π":"pi",
                 "ρ":"rho","Ρ":"rho","σ":"sigma","Σ":"sigma","ς":"sigma","Φ":"phi","φ":"phi","τ":"tau","Τ":"tau",
                 "υ":"upsilon","Υ":"upsilon","Χ":"chi","χ":"chi","Ψ":"psi","ψ":"psi","ω":"omega","Ω":"omega",
                 "$":"dollar","€":"euro","~":"tilde","_":"underscore","ₐ":"sil","%":"percent","³":"cubed"}
    if key in dict_verb: 
        return dict_verb[key]
    if len(key)==1 or not(key.isalpha()):
        return key
    if key=='-':
        return "to"
    return letters(key)


# # Decimal

# In[115]:


def decimal(key):
    if "million" in key or "billion" in key:
        return money(key)
    if key.find(" ") != (-1):
        return measure(key)
    try:
        key = float(key.replace(',',''))
    except ValueError:
        return measure(key)
    try:
        key = float(key.replace(',',''))
        key = p.number_to_words(key,decimal='point',andword='', zero='o')
        if 'o' == key.split()[0]:
            key = key[2:]
        key = key.replace('-',' ').replace(',','')
        return key.lower()
    except:
        try:
            m=key.split('.')
            if len(m)>2 and m[2]!='':
                return date(key)
        except:
            return key


# # DATE

# In[123]:


dict_mon = {'jan': "January", 
            "feb": "February", "mar ": "march", "apr":
            "april", "may": "may ","jun": "june",
            "jul": "july", "aug": "august",
            "sep": "september",
            "oct": "october","nov": "november",
            "dec": "december", "january":"January",
            "february":"February", "march":"march",
            "april":"april", "may": "may", 
            "june":"june","july":"july", "august":"august",
            "september":"september", "october":"october",
            "november":"november", "december":"december"}
def date(key):
    try:
        v =  key.split('-')
        if len(v)==1:
            text=cardinal(v[0][:-2])+" "+cardinal(v[0][-2:])
            return text
        if len(v)==3:
            if v[1].isdigit():
                try:
                    date = datetime.strptime(key , '%Y-%m-%d')
                    text = 'the '+ p.ordinal(p.number_to_words(int(v[2]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                    if int(v[0])>=2000 and int(v[0]) < 2010:
                        text = text  + ' '+cardinal(v[0])
                    else: 
                        text = text + ' ' + cardinal(v[0][0:2]) + ' ' + cardinal(v[0][2:])
                except:
                    text = key
                return text.lower()    
        else:   
            v = re.sub(r'[^\w]', ' ', key).split()

            if v[0].isalpha():
                try:
                    if len(v)==3:
                        text = dict_mon[v[0].lower()] + ' '+ p.ordinal(p.number_to_words(int(v[1]))).replace('-',' ')
                        if int(v[2])>=2000 and int(v[2]) < 2010:
                            text = text  + ' '+cardinal(v[2])
                        else: 
                            text = text + ' ' + cardinal(v[2][0:2]) + ' ' + cardinal(v[2][2:])   
                    elif len(v)==2:

                        if int(v[1])>=2000 and int(v[1]) < 2010:
                            text = dict_mon[v[0].lower()]  + ' '+ cardinal(v[1])
                        else: 
                            if len(v[1]) <=2:
                                text = dict_mon[v[0].lower()] + ' ' + cardinal(v[1])
                            else:
                                text = dict_mon[v[0].lower()] + ' ' + cardinal(v[1][0:2]) + ' ' + cardinal(v[1][2:])
                    else: text = key
                except: text = key
                return text.lower()
            else: 
                key = re.sub(r'[^\w]', ' ', key)
                v = key.split()

                try:

                    date = datetime.strptime(key , '%d %b %Y')

                    text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                    if int(v[2])>=2000 and int(v[2]) < 2010:
                        text = text  + ' '+cardinal(v[2])
                    else: 
                        text = text + ' ' + cardinal(v[2][0:2]) + ' ' + cardinal(v[2][2:])
                except:

                    try:

                        date = datetime.strptime(key , '%d %B %Y')

                        text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                        if int(v[2])>=2000 and int(v[2]) < 2010:
                            text = text  + ' '+cardinal(v[2])
                        else: 
                            text = text + ' ' + cardinal(v[2][0:2]) + ' ' + cardinal(v[2][2:])
                    except:

                        try:
                            date = datetime.strptime(key , '%d %m %Y')

                            text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                            if int(v[2])>=2000 and int(v[2]) < 2010:
                                text = text  + ' '+cardinal(v[2])
                            else: 
                                text = text + ' ' + cardinal(v[2][0:2]) + ' ' + cardinal(v[2][2:])
                        except:
                            try:
                                date = datetime.strptime(key , '%d %m %y')

                                text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                                v[2] = datetime.date(date).strftime('%Y')
                                if int(v[2])>=2000 and int(v[2]) < 2010:
                                    text = text  + ' '+cardinal(v[2])
                                else: 
                                    text = text + ' ' + cardinal(v[2][0:2]) + ' ' + cardinal(v[2][2:])
                            except:

                                text = key
                return text.lower()
    except:
        return key


# In[214]:


def time(key):
    v=key.split()
    try:
        if ":" in key:
            if len(v)==1:
                v=key.split(":")
                if len(v)==3:
                    text=cardinal(v[0])+" hours "+cardinal(v[1])+" minutes and "+cardinal(v[2])+" seconds"
                elif len(v)==2:
                    s=v[1]
                    if "." in key:
                        t=v[1].split('.')
                        text=cardinal(v[0])+" hours "+cardinal(t[0])+" minutes and "+cardinal(t[1])+" seconds"
                    
                    elif len(s)==4:
                        m=s
                        if (m[:-2]=="0" or m[:-2]=="00"):
                            text=cardinal(v[0])+" "+letters(m[-2:])
                        elif m[0]!="0":
                            text=cardinal(v[0])+" "+cardinal(m[:-2])+" "+letters(m[-2:])
                        else:
                            text=cardinal(v[0])+" "+cardinal(m[0])+" "+cardinal(m[1])+" "+letters(s[-2:])
                    else:
                        if (v[0]=="0"):
                            v[0]="zero"
                        if (s[0]=="0"):
                            text=cardinal(v[0])+" "+cardinal(s[0])+" "+cardinal(s[1])
                        else:
                            text=cardinal(v[0])+" "+cardinal(s)
            elif len(v)==2:
                m=v[1]
                t=v[0].split(":")
                if (t[1]=="0" or t[1]=="00"):
                    try:
                        text=cardinal(t[0])+" "+letters(m)
                    except:
                        j=key.split(':')
                        text=cardinal(j[0])+" "+cardinal(j[1])
                else:
                    s=t[1]
                    if (s[0]=="0"):
                        text=cardinal(t[0])+" "+cardinal(s[0])+" "+cardinal(s[1])+" "+letters(m)
                    else:
                        text=cardinal(t[0])+" "+cardinal(s)+" "+letters(m)
        else:
            if len(v)==2:
                if "." in key and len(key)>=7:
                    t=v[0].split(".")
                    m=t[1]
                    if (m[:-2]=="0" or m[:-2]=="00"):
                        text=cardinal(t[0])+" "+letters(m[-2:])
                    else:
                        text=cardinal(t[0])+" "+cardinal(m[:-2])+" "+letters(m[-2:])+" "+letters(v[1])
                else:
                    text=cardinal(v[0])+" "+letters(v[1])
            elif "." in key:
                t=v[0].split(".")
                m=t[1]
                if (len(m)==4):
                    if (m[:-2]=="0" or m[:-2]=="00"):
                        text=cardinal(t[0])+" "+letters(m[-2:])
                    else:
                        text=cardinal(t[0])+" "+cardinal(m[:-2])+" "+letters(m[-2:])
                else:
                    text=cardinal(t[0])+" "+cardinal(t[1])
            else:
                if (len(key)==3 or len(key)==4):
                    text=cardinal(key[:-2])+" "+letters(key[-2:])
            
        
        return text        
    except:
        return key
    
            
            
            


# ## DIGIT

# In[24]:


def digit(x): 
    try:
        x = re.sub('[^0-9]', '',x)
        result_string = ''
        for i in x:
            result_string = result_string + cardinal(i) + ' '
        result_string = result_string.strip()
        return result_string
    except:
        return(x)


# ## LETTERS

# In[35]:


def letters(x):
    try:
        x = re.sub('[^a-zA-Z]', '', x)
        x = x.lower()
        result_string = ''
        for i in range(len(x)):
            result_string = result_string + x[i] + ' '
        return(result_string.strip())  
    except:
        return x


# ## ORDINAL

# In[72]:


#Convert Roman to integers
#https://codereview.stackexchange.com/questions/5091/converting-roman-numerals-to-integers-and-vice-versa
def rom_to_int(string):
    table=[['M',1000], ['CM',900], ['D',500], ['CD',400], ['C',100], ['XC',90],
           ['L',50], ['XL',40], ['X',10], ['IX',9], ['V',5], ['IV',4], ['I',1]]
    returnint=0
    for pair in table:
        continueyes=True
        while continueyes:
            if len(string)>=len(pair[0]):
                if string[0:len(pair[0])]==pair[0]:
                    returnint+=pair[1]
                    string=string[len(pair[0]):]
                else: continueyes=False
            else: continueyes=False
    return returnint

def ordinal(x):
    try:
        result_string = ''
        x = x.replace(',', '')
        x = x.replace('[\.]$', '')
        if re.match('^[0-9]+$',x):
            x = num2words(int(x), ordinal=True)
            return(x.replace('-', ' '))
        if re.match('.*V|X|I|L|D',x):
            if re.match('.*th|st|nd|rd',x):
                x = x[0:len(x)-2]
                x = rom_to_int(x)
                result_string = re.sub('-', ' ',  num2words(x, ordinal=True))
            else:
                x = rom_to_int(x)
                result_string = 'the '+ re.sub('-', ' ',  num2words(x, ordinal=True))
        else:
            x = x[0:len(x)-2]
            result_string = re.sub('-', ' ',  num2words(float(x), ordinal=True))
        return(result_string)  
    except:
        return x


# ## ADDRESS

# In[221]:


def address(key):
    try:
        text = re.sub('[^a-zA-Z]+', '', key)
        num = re.sub('[^0-9]+', '', key)
        result_string = ''
        if len(text)>0: result_string = ' '.join(list(text.lower()))
        if num.isdigit():
            if int(num)<1000:
                result_string = result_string + " " + cardinal(num)
            else:
                result_string = result_string + " " + telephone(num)
        return(result_string.strip())        
    except:    
        return(key)


# ## TELEPHONE

# In[37]:


def telephone(x):
    try:
        result_string = ''
        for i in range(0,len(x)):
            if re.match('[0-9]+', x[i]):
                result_string = result_string + cardinal(x[i]) + ' '
            else:
                result_string = result_string + 'sil '
        return result_string.strip()    
    except:    
        return(x)


# ## ELECTRONIC

# In[41]:


def electronic(x):
    try:
        replacement = {'.' : 'dot', ':' : 'colon', '/':'slash', '-' : 'dash', '#' : 'hash tag', }
        result_string = ''
        if re.match('.*[A-Za-z].*', x):
            for char in x:
                if re.match('[A-Za-z]', char):
                    result_string = result_string + letters(char) + ' '
                elif char in replacement:
                    result_string = result_string + replacement[char] + ' '
                elif re.match('[0-9]', char):
                    if char == 0:
                        result_string = result_string + 'o '
                    else:
                        number = cardinal(char)
                        for n in number:
                            result_string = result_string + n + ' ' 
            return result_string.strip()                
        else:
            return(x)
    except:    
        return(x)


# ## FRACTIONS

# In[45]:


def fraction(x):
    if x.find("½") != -1:
        x = x.replace("½","").strip()
        if len(x) != 0: 
            return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and a half"
        else: 
            return "one half"
    elif x.find("¼") != -1:
        x = x.replace("¼","").strip()
        if len(x) != 0: 
            return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and a quarter"
        else: 
            return "one quarter"
    elif x.find("⅓") != -1:
        x = x.replace("⅓","").strip()
        if len(x) != 0: 
            return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and a third"
        else:
            return "one third"
    elif x.find("⅔") != -1:
        x = x.replace("⅔","").strip()
        if len(x) != 0:
            return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and two thirds"
        else:
            return "two third"
    elif x.find("⅞") != -1:
        x = x.replace("⅞","").strip()
        if len(x) != 0:
            return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and seven eighths"
        else:
            return "seven eighth"
    elif x.find(" ") != -1:
        v = x.split(" ")
        res = " and ".join([fraction(val) for val in v])
        return res.replace("and one","and a")
    
    try:
        y = x.split('/')
        result_string = ''
        if len(y)==1 and y[0].isdigit(): return p.number_to_words(y[0],andword='').replace("-"," ").replace(',','')
        y[0] = p.number_to_words(y[0],andword='').replace("-"," ").replace(',','')
        y[1] = ordinal(y[1]).replace("-"," ").replace(" and "," ").replace(',','')
        if y[1] == "first":
            return y[0]+" over one"
        if y[1] == 'fourth':
            if y[0]=='one': result_string = y[0] + ' quarter'
            else: result_string = y[0] + ' quarters'
        elif y[1] == 'second':
            if y[0]=='one': result_string = y[0] + ' half'
            else: result_string = y[0] + ' halves'
        else:
            if y[0]=='one': result_string = y[0] + " "+ y[1]
            else: result_string = y[0] + ' ' + y[1] + 's'
        return(result_string)
    except:    
        return(x)


# ## MONEY

# In[20]:


def money(x):
    try:
        if re.match('^\$', x):
            x = x.replace('$','')
            if len(x.split(' ')) == 1:
                if re.match('.*M|m$',x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' million dollars'
                elif re.match('.*b|B$', x):
                    x = x.replace('B', '')
                    x = x.replace('b', '')
                    text = cardinal(x)
                    x = text + ' million dollars'
                else:
                    text = cardinal(x)
                    x = text + ' dollars'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' million dollars'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion dollars'
                return x.lower()
        if re.match('^US\$', x):
            x = x.replace('US$','')
            if len(x.split(' ')) == 1:
                if re.match('.*M|m$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' million dollars'
                elif re.match('.*b|B$', x):
                    x = x.replace('b', '')
                    x = x.replace('B', '')
                    text = cardinal(x)
                    x = text + ' million dollars'
                else:
                    text = cardinal(x)
                    x = text + ' dollars'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' million dollars'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion dollars'
                return x.lower()
        elif re.match('^£', x):
            x = x.replace('£','')
            if len(x.split(' ')) == 1:
                if re.match('.*M|m$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' million pounds'
                elif re.match('.*b|B$', x):
                    x = x.replace('b', '')
                    x = x.replace('B', '')
                    text = cardinal(x)
                    x = text + ' million pounds'
                else:
                    text = cardinal(x)
                    x = text + ' pounds'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' million pounds'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion pounds'
                return x.lower()            
        elif re.match('^€', x):
            x = x.replace('€','')
            if len(x.split(' ')) == 1:
                if re.match('.*M|m$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' million euros'
                elif re.match('.*b|B$', x):
                    x = x.replace('B', '')
                    x = x.replace('b', '')
                    text = cardinal(x)
                    x = text + ' million euros'
                else:
                    text = cardinal(x)
                    x = text + ' euros'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' million euros'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion euros'
                return x.lower()  
    except:    
        return(x)

