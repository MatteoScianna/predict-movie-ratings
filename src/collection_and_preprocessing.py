#%%
## IMPORT LIBRARIES
import os
from urllib.parse import quote
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
import imdb
from imdb import Cinemagoer
import json
import stanza
from stanza.utils.conll import CoNLL
import stanza.resources.common

#%%
url = 'https://imsdb.com/all-scripts.html' #URL OF THE SITE
SCRIPTS_DIR = '/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/imsdb_download_all_scripts/scripts_saved' #DIRECTORY IN WHICH WE'LL STORE THE SCRIPTS NB IF NOT PRESENT YOU HAVE TO CREATE ONE

#%%
#Definition of functions

def get_script(relative_link):
    tail = relative_link.split('/')[-1]
    print('fetching %s' % tail)
    script_front_url = url + quote(relative_link)
    front_page_response = requests.get(script_front_url)
    front_soup = BeautifulSoup(front_page_response.text, "html.parser")

    try:
        script_link = front_soup.find_all('p', align="center")[0].a['href']
    except IndexError:
        print('%s has no script :(' % tail)
        return None, None

    if script_link.endswith('.html'):
        title = script_link.split('/')[-1].split(' Script')[0]
        script_url = url + script_link
        script_soup = BeautifulSoup(requests.get(script_url).text, "html.parser")
        script_text = script_soup.find_all('td', {'class': "scrtext"})[0].get_text()
        script_text = clean_script(script_text)
        return title, script_text
    else:
        print('%s is a pdf :(' % tail)
        return None, None

def clean_script(text):
    text = text.replace('<pre><html>', '')
    lines = text.split('\n')
    t = ''
    switch = False
    for l in lines:
        if '<pre>' in l:
           switch = True
        if '</pre>' in l:
           switch = False
        if switch:
            t += l+'\n' 
    return t.replace(r'\r', '')

def remove_non_ascii(string):

    return ''.join(char for char in string if ord(char) < 128)

def remove_outlier_IQR(df):
    Q1=np.quantile(df,0.25)
    Q3=np.quantile(df,0.75)
    IQR=Q3-Q1
    l = [Q1, Q3, IQR]
    #df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return l

def remove_dates(sentence):
    """remove the dates like 30 Mar 2013"""
    sentence = re.sub('\d{2}\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}', '', sentence)
    return sentence

#%%
#EXTRACTION AND STORING OF RAW SCRIPTS
if __name__ == "__main__":
    response = requests.get(url)
    html = response.text

    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all('p')

    for p in paragraphs:
        relative_link = p.a['href']
        print(relative_link)
        if relative_link == 'https://store.steampowered.com/app/1856130': #adv gave problems
            continue
        title, script = get_script(relative_link)
        if not script:
            continue

        with open(os.path.join(SCRIPTS_DIR, title.strip('.html') + '.txt'), 'w', encoding = 'utf-8') as outfile:
            outfile.write(script)

#%%
#NOW THAT WE HAVE OUR RAW SCRIPTS, WE PRE-PROCESS THEM, 
# ELIMINATE OUTLIERS AND
# STORE THEM IN A DIRECTORY 

# Import scripts text from the files in the directory and create a movies_names list
SCRIPTS_DIR = '/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/Computational_linguistic_project/scripts_saved' #DIRECTORY IN WHICH WE'LL STORE THE SCRIPTS NB IF NOT PRESENT YOU HAVE TO CREATE ONE

with open('/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/Computational_linguistic_project/dict_titles_scripts.json') as json_file:
    dict_titles_scripts = json.load(json_file)

dizionariuccio = {}
conta_buoni = 0 
conta_cattivi = 0 #to see how many give problems
lista_cattivi = []
conta =  0  
for item in dict_titles_scripts.items():
    try:
        with open(SCRIPTS_DIR+"/"+item[1], 'r', encoding = 'utf-8') as f:
            text = f.read()
            lines = re.split("\n", text)
            script_text = r""
            switch = False
            for l in lines:
                if '</b>' in l:
                    switch = True 
                if '<b>' in l:
                    switch = False 
                if '<br/>' in l:
                    continue    
                if switch:
                    l = l.replace(r'</b>','')
                    l = remove_non_ascii(l)
                    l = l.strip()
                    if l != '':
                        script_text += l + " "
                script_text = re.sub(r'<[^>]*>[^>]*<[^>]*>', '',script_text)
                script_text = re.sub(r'<[^>]*>', '',script_text)
            dizionariuccio[item[0]] = {}
            dizionariuccio[item[0]]["Name"] = item[1]
            dizionariuccio[item[0]]["script"] = script_text
        conta_buoni +=1 
        print(conta_buoni)
    except:
        conta_cattivi +=1
        lista_cattivi.append(item[0])
        pass

#%% 
# Further preprocessing of sripts to obtain a clean text
length_list = []
for k in dizionariuccio.keys():
    s = dizionariuccio[k]['script']
    s = re.sub(r"\'", "'", s)
    s = re.sub(r"\\", "", s)
    s = re.sub(r"/", " ", s)
    s = re.sub(r',+',',',s)
    s = re.sub('--','',s)
    s = re.sub(' - ',' ',s)
    s = re.sub(r'\*','',s)
    s = re.sub(r'\.{2,}',' ',s)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'http.*', '', s)
    s = re.sub('\n+',' ',s)
    s = re.sub('\s+',' ',s)
    s = s.lower()
    dizionariuccio[k]['script'] = s
    dizionariuccio[k]["length"] = len(s)
    length_list.append(len(s))

#%% # Remove from the dictionary all the movie that has a charachter lenght inferior to 1.5*IQR
IQR = remove_outlier_IQR(length_list)
pre = len(dizionariuccio.keys())
out_names = []
removed_movies_dict = {}
for key in dizionariuccio.keys():
    if dizionariuccio[key]["length"] < (IQR[0]-1.5*IQR[2]):
        out_names.append(key)

for key in out_names:
    v = dizionariuccio.pop(f'{key}')
    removed_movies_dict[f'{key}'] = v
post = len(dizionariuccio.keys())

print('The total number of movies is:', post,'\nThe total number of outlier was:',pre-post)
#tot 1051
#outliers 54


#%% 
for k in removed_movies_dict.keys():
    print(f'\n\n{k}\n\n')
    print(removed_movies_dict[f'{k}'])

#%%
# Save the dictionary with the preprocessed movies' scripts
with open('dict_movies_script.json', 'w', encoding="utf-8") as f:
  json.dump(dizionariuccio, f)

#%%
# Save the preprocessed scripts
for k in dizionariuccio.values():
    nome = k["Name"]
    with open(f'preprocessed_scripts/{nome}','w', encoding="utf-8") as f:
        f.write(k['script'])


# %%
#########################

#NOW WE ONLY CONSIDER SCRAPED MOVIES (LESS THAN BEFORE)

SCRIPTS_DIR = "/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/Computational_linguistic_project/preprocessed_scripts"

dict_titles_scripts = {} #CREATE DITIONARY IWITH REAL TITLE AND NAME OF TXT FILE
conta = 0
substring = ", The" # PROBLEM WITH MOVIES STARTING WITH "THE", HERE WE FIX IT 

directory =  SCRIPTS_DIR
if __name__ == '__main__':
# assign directory
# iterate over files in
# that directory
    for filename in os.listdir(directory):
        try:
            filename1 = filename[:-4]
            filename_new = filename1.replace("-", " ")
            if substring in filename_new:
                filename_new1 = filename_new.replace(", The", "")
                filename_new1 = "The " + filename_new1
                dict_titles_scripts[filename_new1] = filename
            else:
                dict_titles_scripts[filename_new] = filename
            conta += 1
        except:
     
            print("The following file gave problems: " + filename) #So we can see when the program did not work
print("The total number of Scripts is: " + str(conta))


#%%
#Now we save the dictionary with titles in a txt file
file = json.dumps(dict_titles_scripts)
f = open("dict_titles_scripts.json","w")
f.write(file)
f.close()

#%%
#Read dictionary with movie names 

with open('/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/Computational_linguistic_project/dict_titles_scripts.json') as json_file:
    dict_titles_scripts = json.load(json_file)
 
#%%
ia = imdb.IMDb()
dict_movies_id = {}

#%%

#CREATING A DICTIONARY WITH MOVIE NAMES AND IMDb ID (takes a while)
conta = 0
conta_buoni = 0 
conta_cattivi = 0
lista_problemi = []
for key in dict_titles_scripts.keys(): 
    try:
        movie = ia.search_movie_advanced(key)[0]
        dict_movies_id[key] = movie.getID()
        conta_buoni +=1
        print(str((conta_buoni/len(dict_titles_scripts))*100) + " percent of movies correctly matched with their id")
    except:
        lista_problemi.append(key)
        conta_cattivi +=1
        print(str((conta_cattivi/len(dict_titles_scripts))*100) + " percent of movies not matched with their id") #here we highlights movies that raise errors (don't know why). Need to be manually insered in the dictionary
        pass
    conta+=1
    print(str(conta) + " movies processed")

print("The total number of movies correctly matched with their ID is:" + str(conta_buoni))
print("The total number of movies not matched with their ID is:" + str(conta_cattivi))
#lista_problemi
#%%

#Save the dictionary in json

file = open('list_bad.txt','w')
for title in lista_problemi:
	file.write(title+"\n")
file.close()

###### USEFUL COMMANDS TO EXTRACT MOVIE FEATURES ###########
#ia.get_movie(dict_movies_id["movie name"]).get_current_info()[3] returns the gross income 
#ia.get_movie(dict_movies_id["movie name"]).getID() returns IMDB id of the movie 
#ia.get_movie(dict_movies_id["movie name"]).values()[3] returns genres
#ia.get_movie(dict_movies_id["movie name"]).values()[4] returns duration
#ia.get_movie(dict_movies_id["movie name"]).values()[11]["Budget"] Returns budget
#ia.get_movie(dict_movies_id["movie name"]).values()[11]["Opening Weekend United States"] Returns gross of opening weekend in USA 
# *** NB DON'T KNOW IF IT IS ALWAYS USA OR DEPENDS ***
#ia.get_movie(dict_movies_id["movie name"]).values()[11]['Cumulative Worldwide Gross'] Returns total gross
#ia.get_movie(dict_movies_id["movie name"]).values()[-7] Returns a long plot of the movie
#ia.get_movie(dict_movies_id["movie name"]).values()[14] Returns IMDb rating

#%%
#PIANO PIANO QUA SI SISTEMANO GLI ID DI TUTTI I FILM 
def sistema_IDs(lista, dizionario):
    i = 0
    lista1 = lista
    for problema in lista1:
        ID = input(problema)
        dizionario[problema] = ID 
        lista.remove(problema)
        i +=1
        print(str(i)+ " di " + str(len(lista1)) + " id sistemati")
    

sistema_IDs(lista_problemi,dict_movies_id)

#%%
dict_movie1 = dict_movies_id
for key, value in dict_movie1.items():
    if value == "":
        del dict_movies_id[key]

#%%
substring = "\""

for key, value in dict_movies_id.items():
    if substring in value:
        k[key] = value.replace(substring,"")



#%%

movies_id = json.dumps(dict_movies_id)
f = open("dict_movies_id.json","w")
f.write(movies_id)
f.close()


#%%
with open('/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/Computational_linguistic_project/dict_movies_id.json') as json_file:
    dict_movies_id = json.load(json_file)


#with open('/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/Computational_linguistic_project/dict_movies_scripts.json') as json_file:
#    dict_movies_scripts = json.load(json_file)
# %%
#Create dictionary with movie information, like gross revenue, imdb rating, movie genres and budget

i = 0
j = 0
dict_movies_features = {}
lista_brutti = ["$", "GBP", "(estimated)", ","]
lista_problemi = []

for key, value in dict_movies_id.items():
    if key in dict_movies_scripts.keys():
        dict_movies_features[key] = {}
        info = ia.get_movie(value).values()
        if info[0] == info[1]:
            info.pop(0)
        genres = info[2]
        rating = info[13]
        dict_movies_features[key]["Genres"] = genres
        dict_movies_features[key]["Rating"] = rating
        for info1 in info:
            if type(info1) == dict:
                try:
                    budget = info1["Budget"]
                    budget_def = budget
                    for simbolo in lista_brutti:
                        if simbolo in budget:
                            budget_def = budget_def.replace(simbolo, "")
                    dict_movies_features[key]["Budget"] = budget_def
                    gross = info1["Cumulative Worldwide Gross"]
                    gross_def = gross
                    for simbolo in lista_brutti:
                        if simbolo in gross:
                            gross_def = gross_def.replace(simbolo, "")
                    try:
                        gross_def = remove_dates(gross_def)
                    except:
                        pass
                    dict_movies_features[key]["Gross"] = (gross_def)
                    i +=1 
                    print(str((i/len(dict_movies_id))*100)+" percent movies went right, this one was: " + key)


                except: 
                    j+=1
                    print(str((j/len(dict_movies_id))*100)+" percent movies went wrong, this one was: " + key)
                    lista_problemi.append(key)
 

#%%

for key, value in dict_movies_features.items():
    if re.search(r"\d.\d", value["Rating"]):
        pass
    else: 
        print(key)
        print(value["Rating"])

#%%
#Now we save the dictionary with features in a txt file
file = json.dumps(dict_movies_features)
f = open("dict_movies_features.json","w")
f.write(file)
f.close()

#%%

with open('/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/dict_movies_features_vocabulary_richness.json') as json_file:
    dict_movies_features = json.load(json_file)

# %%
#PIANO PIANO QUA SI SISTEMANO I GROSS DI TUTTI I FILM 
def sistema_grosses(lista, dizionario):
    i = 0
    for problema in lista:
        gross = input(problema)
        dizionario[problema]["Gross"] = gross
        i+=1
        print(str(i)+" of " + str(len(lista))+ " processed")

sistema_grosses(l, dict_movies_features)

#%%

# opening the file in read mode
my_file = open("/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/list_bad_gross.txt", "r")

# reading the file
data = my_file.read()

# replacing end of line('/n') with ' ' and
# splitting the text it further when '.' is seen.
lista_problemi = data.split("\n")

# printing the data
print(lista_problemi)
my_file.close()


#%% MO VOGLIAMO METTERE TUTTO IN CONLL


# Initializing stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse') # Info descritte nel paper, che dovrebbero essere necessarie per calcolare gli indici
movie_values = dizionariuccio.values()
i = 0 
for n in movie_values["Name"]:
    if i == 1: #TO BE REMOVED
        break
    with open(f'preprocessed_scripts/{n}.txt','r', encoding="utf-8") as f:
        text = f.read()
        doc = nlp(text)
        # Converting in conll format
        dicts = doc.to_dict()
        conll = CoNLL.convert_dict(dicts)
        # Saving in a folder
        with open(f'conll_scripts/{n}' + ".conllu", mode="w", encoding="utf-8") as out:
            for s in conll:
                out.write("\n".join(("\t".join(token) for token in s)))
                out.write("\n\n")
    i += 1


#%%
male = 0
bene = 0
lista_male = []
for movie in dict_movies_features.keys():
    movie1 = dict_movies_features[movie]["Rating"]
    if re.search(r"\d\.\d", str(movie1)):
        bene+= 1
        pass
    else:
        lista_male.append(movie)
        male+=1 

#%%
def sistema_rating(lista, dizionario):
    i = 0
    for problema in lista:
        rating = input(problema)
        dizionario[problema]["Rating"] = rating
        i +=1 
        print("Sistemati "+str(i)+" ratings su "+ str(len(lista)))
sistema_rating(lista_male,dict_movies_features)


#%%

file = open('list_bad_rating.txt','w')
for title in lista_male:
	file.write(title+"\n")
file.close()

file = open('list_bad_gross.txt','w')
for title in lista_problemi:
	file.write(title+"\n")
file.close()

# %%
