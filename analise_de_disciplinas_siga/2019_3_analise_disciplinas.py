# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pandas as pd
import numpy as np
import seaborn as sns
import re
import pylab as pl
import glob
import os
#%%
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
#%% 
def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e4):
        return '%1.2f' % x   
      else:
        return '%1.4g' % x
  
def fstat(x):
  #m,s= '{:1.4g}'.format(np.mean(x)), '{:1.4g}'.format(np.std(x))
  m,s, md= fmt(np.mean(x)), fmt(np.std(x)), fmt(np.median(x)) 
  #text=str(m)+'$\pm$'+str(s)
  text=str(m)+' ('+str(s)+')'#+' ['+str(md)+']'
  return text

from unicodedata import normalize
def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
  
#%%
pl.rc('text', usetex=True)
pl.rc('font', family='serif',  serif='Times')    
sns.set_style(style="white", rc={
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
sns.set_context("paper", font_scale=1.2, 
        rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":12,
            'xtick.labelsize':14,'ytick.labelsize':14,
            'font.family':"Times New Roman", }) 
#%%
  
path='./data/'

csv={
     #'Docentes'     :'./data/2019_1_analise_disciplinas_ead_discentes.csv',
     #'Discentes'    :'./data/2019_1_analise_disciplinas_ead_docentes.csv',
     'Docentes'     :'./data/RelatorioRespostasAluno_2019_3_Presencial.csv',
     'Discentes'    :'./data/RelatorioRespostasDocente_2019_3_PRESENCIAL_modificado.csv',     
}

filelist=glob.glob(path+'*.csv')

X,Y = [],[]
for c in csv:
    f=csv[c]
    print(c,f)
    df = pd.read_csv(f, sep=';', decimal=',', encoding = 'latin1')
    #df['Ano']=[int(i) for i in df['Ano']]
    s=df['Tipo Avaliacao'].dropna().unique()
    if s[0]==u'ALUNO_TURMA':
            X.append(df)
    else:
            Y.append(df)
            
X = pd.concat(X)    
Y = pd.concat(Y)    

X=X.reindex()
Y=Y.reindex()


#X[u'Período']=X['Período']
#Y[u'Período']=Y['Perï¿½odo']
#Y.drop(['Perï¿½odo'], axis=1, inplace=True)

#W = pd.concat([X,Y], sort=False)

#%%
# Limpeza das entradas NaN ou #

X.dropna(axis=0, subset=['Curso Aluno'], inplace=True)
Y.dropna(axis=0, subset=['Professor'], inplace=True)

X.dropna(axis=0, subset=['Departamento'], inplace=True)
Y.dropna(axis=0, subset=['Departamento'], inplace=True)

X=X[X['Ano']!='#']
Y=Y[Y['Ano']!='#']

X=X[X['Curso Aluno']!='00A']
   
#%%
#print('-'*80+'\n'+'Lista dos professores que responderam os questionarios e também \nforam avaliados pelo alunos'+'\n'+'-'*80)

dic_prof={}
for p in Y['Professor'].unique():
    #print(p, end=' -- ')
    if p in X['Professor'].unique():
        aux=X[X['Professor']==p]
        c_ = '|'.join(aux['Curso Aluno'].unique())
        #print(c_, end='\n')
        dic_prof[p]=c_
    else:
        #print(None)
        dic_prof[p]=None

Y['Cursos'] = [ dic_prof[p] for p in Y['Professor'] ]
#%%

#lista_cursos=['77A', '65A', '34A', '04A', '23A', '65B', '65D','15A', '64A', 
#                                                     '08GV', '87A']
#lista_cursos=[u'08GV', u'63E', u'88A', '73BL', '63B', '71A']
#lista_cursos=['71A', '29A', '24A']
#X.dropna(subset=['Curso Aluno'], inplace=True)
#lista_cursos=[i.replace('/','-') for i in X['Curso Aluno'].unique()]
#

# Todos os cursos
lista_cursos=X['Curso Aluno'].dropna().unique()
lista_cursos=[i.replace('/','-') for i in lista_cursos]
lista_cursos.sort()
#
#%%

#
# Analise Discente
#


#cabecalho = ['Ano','Periodo','Departamento','Disciplina','Professor','Tipo'] 

cabecalho = [u'Ano', u'Período', u'Tipo Avaliacao', u'Professor', u'Departamento',
       u'Disciplina', u'Turma', u'Cursos', u'Aluno', u'Ano Ingresso', u'Tipo Ingresso',
       u'Curso Aluno', u'IRA', u'Escola 2G', u'Bolsa Apoio', u'Tipos Bolsa',
       u'Outras Bolsas', u'Cidade', u'Estado', u'Nota Final', u'Perï¿½odo',
       'ind_disc']

idx = X.columns.isin(cabecalho)

perguntas = X.columns[~idx]
questoes = [ 'Q'+'{:02d}'.format(i+1) for i in range(len(perguntas))]

lista_questoes_alunos = pd.DataFrame(np.c_[questoes, perguntas])
lista_questoes_alunos.to_csv('lista_questoes_alunos.csv', encoding='latin-1',
                             header=False,index=False, sep=';')

questoes_alunos={}
for i,j in zip(questoes,perguntas): 
    questoes_alunos[i]=j

#%%
#    
# Elimina alunos com RI e TM
#
C = X[X.columns[idx]]
Q = X[X.columns[~idx]]
Q.columns = [ 'Q'+'{:02d}'.format(i+1) for i in range(len(Q.columns))]
X = pd.concat([C,Q], axis=1)

#
try:
    X = X[ X['Nota Final'] !='RI' ]
except:
    pass

try:
    X = X[ X['Nota Final'] !='TM' ]
except:
    pass

#%%

X['ind_disc']=X['Disciplina'].values+X['Turma'].values
Y['ind_disc']=[ str(i)+str(j) for (i,j) in  zip(Y['Disciplina'].values, Y['Turma'].values)]

X=X[X['Ano']!='REFERENCIA']
Y=Y[Y['Ano']!='REFERENCIA']
X['Ano'] = [int(i) for i in X['Ano'].values ] 
Y['Ano'] = [int(i)  for i in Y['Ano'].values ] 

X['Ano Ingresso'] = [int(i) for i in X['Ano Ingresso'].values ] 
#Y['Ano Ingresso'] = [int(i) for i in Y['Ano Ingresso'].values ] 

#%%

aux=[]
for i in lista_cursos:
    #print(i)
    aux.append(X[X['Curso Aluno']==i])

X=pd.concat(aux)    
#%%

idx = Y.columns.isin(cabecalho)

cabecalho_docente=Y.columns[idx]
perguntas_docentes = Y.columns[~idx]
questoes_docentes = [ 'Q'+'{:02d}'.format(i+1) for i in range(len(perguntas_docentes))]


lista_questoes_docentes = pd.DataFrame(np.c_[questoes_docentes, perguntas_docentes])
lista_questoes_docentes.to_csv('lista_questoes_docentes.csv', encoding='latin-1',
                             header=False,index=False, sep=';')


C = Y[Y.columns[idx]]
Q = Y[Y.columns[~idx]]
Q.columns = [ 'Q'+'{:02d}'.format(i+1) for i in range(len(Q.columns))]
Y = pd.concat([C,Q], axis=1)
#%%

#Z=[]
#for c,df in Y.groupby(['Cursos']):
#    aux=df.copy()
#    for i in c.split('|'):
#        aux['Cursos']=i
#        Z.append(aux)
#
#Z=pd.concat(Z)
#
#Z['Curso']=Z['Cursos']
#
#idx = Z['Curso'].isin(lista_cursos)
#Z = Z[idx]

#%% 
#list(X['Disciplina'].unique())
#
#G = []
#for a, df in X.groupby(['Curso Aluno']):
#    num_alunos=df['Aluno'].unique().shape[0]
#    lista_disciplinas=list(df['Disciplina'].unique())
#    lista_disciplinas = np.unique(lista_disciplinas)
#    ind_disc = df['ind_disc'].unique()
#    num_disciplinas = len(lista_disciplinas)
#    G.append({'Curso':a, 'Num. Alunos':num_alunos, 'Disciplinas':lista_disciplinas, 
#              'Num. Disciplinas':num_disciplinas,
#              'ind_disc':ind_disc})
#
#G1=pd.DataFrame(G); G1.index=G1['Curso'].values
#writer = pd.ExcelWriter('quantitativo_info_cursos.xlsx')
#G1.drop(['ind_disc','Disciplinas', 'Num. Disciplinas'], axis=1).to_excel(writer,'Alunos por Disciplina', index=False)
#
#H=[]
#for c in G1['Curso']:
#    for i in G1['ind_disc']:
#        for j in i:
#            df1=Z[(Z['ind_disc']==j) & (Z['Curso']==c)]
#            if len(df1)>0:
#                print(c,df1.Curso.unique(), len(df1))
#                H.append(df1)
#                aux=df1
#
#H=pd.concat(H)
#
#J=[]
#for a, df in H.groupby(['Cursos']):
#    dic={'Departamento':'Num. Departamentos','Professor':'Num. Professores', 
#         'Disciplina':'Num. Disciplinas', 'ind_disc':'Num. Turmas'}
#    aux={'Curso':a}
#    for i in dic:
#        print(a,dic[i],df[i].unique().shape[0])
#        aux[dic[i]]=df[i].unique().shape[0]
#
#    J.append(aux)
#    
#J1=pd.DataFrame(J); J1.index=J1['Curso'].values
#
#J1.to_excel(writer,'Quantitativos por Curso', index=False)
#writer.save()
#%%
fn1 = './aux/'+'departamentos_nomes_siglas.csv'
dataframe_departamentos= pd.read_csv(fn1, encoding = 'latin-1')

dic={}
for i in range(len(dataframe_departamentos)): 
    d, s = dataframe_departamentos.iloc[i]['Nome'], dataframe_departamentos.iloc[i]['Sigla']
    dic[d]=s#.encode('utf-8').decode('latin-1')


#X['Departamento'] = [i.encode('utf-8').decode('latin-1') for i in X['Departamento']] 
dep_list=[]    
for i in X['Departamento']:
    s=dic[i.encode('utf-8').decode('latin-1')]        
    dep_list.append(s)

X['Departamento'] = dep_list   
#%%
dep_list=[]    
for i in Y['Departamento']:
    s=dic[i.encode('utf-8').decode('latin-1')]        
    dep_list.append(s)

Y['Departamento'] = dep_list   

#%% 
#dep_list=[]    
#for i in Z['Departamento']:
#    dep_list.append(dic[i])
#
#Z['Departamento'] = dep_list    
#%% 
fn2 = './aux/'+'cursos_nomes_siglas.csv'
dataframe_cursos= pd.read_csv(fn2)

dic={}
for i in range(len(dataframe_cursos)): 
    d, s = dataframe_cursos.iloc[i]['CODIGO'], dataframe_cursos.iloc[i][u'NOME']
    dic[d]=s
    
dicionario_cursos=dic

#%%
#
# trtamento do código e-MEC
#
# https://medium.com/capivarapython/itera%C3%A7%C3%A3o-em-pandas-dataframes-e-desempenho-72d2d12522e1

fn3 = './aux/'+'referencial-de-cursos-ufjfe-mec.csv'
dataframe_emec= pd.read_csv(fn3, sep='\t')

codigo_emec=[]
for index, row in dataframe_emec.iterrows():
    c=row['Código SIGA'].split(';')
    #row.drop(['Código SIGA'],inplace=True)
    for i in c:
        j=i.replace(' ','').replace(' ','').replace(' ','')
        row['Código SIGA']=str(j)
        codigo_emec.append(dict(row))
        #print(j,c,row.values)


codigo_emec=pd.DataFrame(codigo_emec)
    
#%%
dic_codigo_emec={}
dic_nome_emec={}

for index, row in codigo_emec.iterrows():
    dic_codigo_emec[row['Código SIGA']]=row['Código e-MEC']
    dic_nome_emec[row['Código SIGA']]=row['Curso']
    
    
#%%
dic={}
for i in range(len(dataframe_cursos)): 
    d, s = dataframe_cursos.iloc[i]['CODIGO'], dataframe_cursos.iloc[i][u'NOME']
    dic[d]=s
    
dicionario_cursos=dic


X['Código e-MEC']=[dic_codigo_emec[i] for i in X['Curso Aluno']]
X['Curso Nome']=[dic_nome_emec[i] for i in X['Curso Aluno']]
X['Curso SIGA']=X['Curso Aluno'].values
#%%
aux=[]
for l in lista_cursos:
    df = X[X['Curso Aluno']==l]
    aux.append(df)
        
X = pd.concat(aux)
X = X.reindex()
lista_dep = X['Departamento'].dropna().unique(); lista_dep.sort()

#%%

#dic_cursos={i:dicionario_cursos[i]+' ('+i+')' for k,i in enumerate(lista_cursos)}
#nomes_cursos={}
#for c in dic_cursos:
#    nomes_cursos[c] = dic_cursos[c].replace('-','').replace('  ',' ').replace(' (',' (').replace(' ','\n').replace('/','-')
#    nomes_cursos[c] = nomes_cursos[c].replace('EM\n','EM ').replace('DA\n','DA ')
#
#tab_cursos=pd.DataFrame( [{' Sigla':i, 'Nome do Curso':dicionario_cursos[i]} for k,i in enumerate(lista_cursos)] )
#print (tab_cursos)
#tab_cursos.to_csv(path_or_buf='cursos_participantes.csv',index=False)

#%% 
tab=[]
info_aluno=[]
for i, df in X.groupby(['Código e-MEC']):
    n= df['Aluno'].unique().shape[0]
    aluno=[]
    for j, df1 in df.groupby('Aluno'):
        aux={h:df1[h].unique().shape[0] for h in df1}
        aux['Aluno']=j
        aux={'Aluno':j, 'NProf':df1['Professor'].unique().shape[0],
             'NDspln':df1['Disciplina'].unique().shape[0],
             'Apoio':df1['Bolsa Apoio'].unique()[0], 
             'Ano Ingresso':df1['Ano Ingresso'].unique()[0],
             'Tipo Ingresso':df1['Tipo Ingresso'].unique()[0],
             'Bolsa Apoio':df1['Bolsa Apoio'].unique()[0],
             'Sigla Curso':df1['Curso Aluno'].unique()[0],
             'Estado':df1['Estado'].unique()[0],
             #'Curso':nomes_cursos[i],             
             'Curso EMEC':df1['Código e-MEC'].unique()[0],             
             'Curso Nome':df1['Curso Nome'].unique()[0],             
             'Curso':df1['Curso Nome'].unique()[0],             
             'Curso SIGA':df1['Curso Aluno'].unique()[0],             
             'IRA':df1['IRA'].unique()[0]}
        aluno.append(aux)
        info_aluno.append(aux)

    # alguns valores estavam com o separador "," no lugar do ponto "."
    v=np.array([str(i).replace(',','.') for i in df['IRA']]).astype(float)        
    dic={'Curso':i, 'Curso SIGA':df1['Curso Aluno'].unique()[0], 'NA':n, 'IRA':fstat(v)}    
    
    aluno = pd.DataFrame(aluno)
    
    #print('='*80,'\n','Curso: '+str(i)+'\n'+'='*80)
    print(dic)
    tab.append(dic)

    
tab=pd.DataFrame(tab)
tab.to_csv('tabela_informacoes_1'+'.csv')
print(tab)

info_aluno = pd.DataFrame(info_aluno)
info_aluno['Ano Ingresso']=[int(i) for i in info_aluno['Ano Ingresso']]

#%% 
##info_aluno=info_aluno[info_aluno['Sigla Curso']=='08GV']
##sns.set()    
#ct=pd.crosstab(info_aluno['Curso'], info_aluno['Bolsa Apoio']); 
#ct = ct.T/ct.sum(axis=1).values
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Bolsas de Apoio')
#g.set_ylabel(u'Porcentagem de discentes\ncom bolsa de apoio')
#g.legend(title='Bolsa de Apoio', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('quantitativos_bolsa_de_apoio_por_curso.png', dpi=300,bbox_inches='tight')
#print(X['Tipos Bolsa'].unique())
#
##sns.set()
#n_colors=info_aluno['Ano Ingresso'].unique().shape[0]
##sns.set_palette("RdBu", n_colors=n_colors,)  
#ct=pd.crosstab(info_aluno['Curso'], info_aluno['Ano Ingresso']); 
#ct = ct.T/ct.sum(axis=1).values
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Distribuição do Ano de Ingresso')
#g.set_ylabel(u'Porcentagem de discentes participantes')
#g.legend(title='Ano de Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('quantitativos_ano_de_ingresso_por_curso.png', dpi=300,bbox_inches='tight')
#
##sns.set()
#n_colors=info_aluno['Tipo Ingresso'].unique().shape[0]
#sns.set_palette("cubehelix", n_colors=n_colors,)   
#ct=pd.crosstab(info_aluno['Curso'], info_aluno['Tipo Ingresso']); 
#ct = ct.T/ct.sum(axis=1).values
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Forma de ingresso dos discentes')
#g.set_ylabel(u'Porcentagem de discentes participantes')
#g.legend(title='Ano de Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('quantitativos_tipo_de_ingresso_por_curso.png', dpi=300,bbox_inches='tight')
#
##sns.set()
#n_colors=info_aluno['NDspln'].unique().shape[0]
#sns.set_palette("cubehelix", n_colors=n_colors,)   
#ct=pd.crosstab(info_aluno['Curso'], info_aluno['NDspln']); 
#ct = ct.T/ct.sum(axis=1).values
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Número de disciplinas')
#g.set_ylabel(u'Porcentagem de discentes participantes')
#g.legend(title='Número de\nDisciplinas Avaliadas', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('quantitativos_no_de_disciplinas_por_curso.png', dpi=300,bbox_inches='tight')
#
##sns.set()
#n_colors=info_aluno['Estado'].unique().shape[0]
#sns.set_palette("Set1", n_colors=n_colors,)   
#ct=pd.crosstab(info_aluno['Curso'], info_aluno['Estado']); 
#ct = ct.T/ct.sum(axis=1).values
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Estado de origem')
#g.set_ylabel(u'Porcentagem de discentes participantes')
#g.legend(title='Estado', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('quantitativos_estado_de_origem_por_curso.png', dpi=300,bbox_inches='tight')

#%% 
#
#for c,df in info_aluno.groupby('Curso'):
#    print (c)
#    df.reindex()
#    sns.set()
#    n_colors=df['Tipo Ingresso'].unique().shape[0]
#    sns.set_palette("cubehelix", n_colors=n_colors,)   
#    ct=pd.crosstab(df['Tipo Ingresso'],df['Ano Ingresso'], ); 
#    #ct = ct.T/ct.sum(axis=1).values
#    g = ct.T.plot.bar(stacked=True, )
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_title(c)
#    g.set_ylabel(u'Porcentagem de discentes participantes')
#    g.legend(title='Ano de Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    #pl.savefig('quantitativos_tipo_de_ingresso_por_curso.png', dpi=300,bbox_inches='tight')

#%% 

print ("Número  médio de disciplinas avaliadas por aluno (por curso):")
tab_disciplinas=[]
for i, df in X.groupby(['Curso Nome']):
    n_disciplinas = df.groupby(['Aluno']).agg(len)['Disciplina'].mean()
    #n_alunos        = df['Aluno'].unique().shape[0]
    #n_professores   = df['Disciplina'].unique().shape[0]
    print (i,'\t',n_disciplinas)
    tab_disciplinas.append({
            'ND':n_disciplinas,
            #'NA':n_alunos, 'NP':n_professores,
            'Participante':'Docente',
            'Curso':i})   
    
tab_disciplinas=pd.DataFrame(tab_disciplinas)
tab_disciplinas['Cursos']=[i for i in tab_disciplinas['Curso']]

#
#pl.figure()
#g=sns.catplot(x='Cursos', y='ND', data=tab_disciplinas, kind='bar',palette='Greens_d',
#               aspect=1, order=tab_disciplinas['Cursos'])
#g.set_xticklabels(rotation=90)
##g.set_xlabels('Curso (sigla)')
#g.set_xlabels('')
#g.set_ylabels(u'Número de disciplinas avaliadas por aluno')
#for p in g.ax.patches:
#        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
#             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars
#
#g.savefig('numero_disciplinas_avaliadas_por_aluno.png', dpi=300,bbox_inches='tight')


#%%

print ("Número de discentes participantes por curso:")
tab=[]
for i, df in X.groupby(['Curso Nome']):
    n= df['Aluno'].unique().shape[0]
    print (i,'\t',len(df)/n)
    tab.append({'NA':n, 'NF':len(df),
                'Participante':'Aluno',
                #'MRA':len(df)/(1.0*n),
                'Curso':i})   
    
tab=pd.DataFrame(tab)
tab.to_csv('tabela_informacoes_2'+'.csv')
tab['Cursos']=[i for i in tab['Curso']]

#
#pl.figure()
#g=sns.catplot(x='Cursos', y='NA', data=tab, kind='bar', palette='Blues_d',
#               aspect=2, order=tab['Cursos'])
#g.set_xticklabels(rotation=90)
#g.set_xlabels('')#'Curso (sigla)')
#g.set_ylabels(u'Número de participantes')
#for p in g.ax.patches:
#        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
#             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars
#
#g.savefig('numero_participantes_curso.png', dpi=300,bbox_inches='tight')
#
##tab['Curso*'] = [dic_cursos[i]+' ('+i+')'  for i in tab['Curso']]
##pl.figure()
##g=sns.catplot(x='Cursos', y='NA', data=tab, kind='bar',palette='Blues_d',
##               aspect=2, errwidth=0,)
##g.set_xticklabels(rotation=90)
##g.set_xlabels('')#'Curso (sigla)')
##g.set_ylabels(u'Número de participantes')
##for p in g.ax.patches:
##        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
##             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars
##
##g.savefig('numero_participantes_curso_nome.png', dpi=300,bbox_inches='tight')
#
#
#pl.figure()
#ct = pd.crosstab(X['Aluno'], X['Disciplina'])
#g=sns.distplot(ct.sum(), rug=True, kde=False, bins=30)
#g.set_xlabel(u'Número de participantes por disciplina')
#g.set_ylabel(u'Ocorrência')
#g.set_xlim([1,ct.sum().max()])
#pl.savefig('numero_participantes_disciplina_histograma.png', dpi=300,bbox_inches='tight')

#%%
#tab_participantes = pd.concat([tab, tab_disciplinas])
#
#pl.figure()
#g=sns.catplot(x='Cursos', y='NA', data=tab_participantes, kind='bar', 
#              hue='Participante',legend=False,
#              #palette='Blues_d',
#              aspect=2, order=tab['Cursos'])
#g.set_xticklabels(rotation=90)
#g.set_xlabels('')#'Curso (sigla)')
#g.set_ylabels(u'Número de participantes')
#pl.legend(loc='upper right', fontsize=16)
#for p in g.ax.patches:
#        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
#             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars
#
#g.savefig('numero_participantes_curso.png', dpi=300,bbox_inches='tight')

#%%

#pl.figure()
#ct = pd.crosstab(X['Disciplina'], X['Aluno'])
#g=sns.distplot(ct.sum(), rug=True, kde=False, bins=30)
#g.set_xlabel(u'Número de participantes por disciplina')
#g.set_ylabel(u'Ocorrência')
#g.set_xlim([1,ct.sum().max()])
#pl.savefig('numero_participantes_disciplina_histograma.png', dpi=300,bbox_inches='tight')
#
#pl.figure()
#ct = pd.crosstab(X['Disciplina'], [X['Aluno'], X['Curso Nome']])
#tab = ct.sum().unstack().mean()
#print(tab)
#
#pl.figure()
#sns.set_palette("Blues_d", len(lista_cursos))
#print(tab)
#g=tab.plot(kind='bar')
#g.set_ylabel(u'Número médio de formulários\nrespondidos por discente')  
#g.set_xlabel('Curso')
#pl.legend('')
#for p in g.patches:
#        g.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
#             ha='right', va='center', rotation=90, xytext=(0, 15), textcoords='offset points')  #vertical bars
#
#pl.savefig('numero_formularios_respondidos_discente_curso.png', dpi=300,bbox_inches='tight')
#
#pl.figure()
#sns.set_palette("Blues_d", len(lista_cursos))


#pl.figure()
#g=sns.catplot(x='Curso Aluno', data=X, kind='count', palette='Blues_d',
#                 aspect=2, order=lista_cursos)
#g.set_xticklabels(rotation=90)
#g.set_xlabels('Curso (sigla)')
#g.set_ylabels(u'Número de formulários')
#
#for p in g.ax.patches:
#        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
#             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars
#
#g.savefig('numero_formularios_curso.png', dpi=300,bbox_inches='tight')
#    
#pl.figure() 
#g=sns.catplot(x='Departamento', data=X, kind='count', palette='Blues_d',
#                 aspect=3, order=lista_dep)
#g.set_xticklabels(rotation=90)
#g.set_xlabels('Departamento (sigla)')
#g.set_ylabels('')
#
#for p in g.ax.patches:
#        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
#             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars
#
#g.savefig('numero_participantes_departamento.png', dpi=300,bbox_inches='tight')
#%%    
X[questoes]=X[questoes].fillna(0.)
X[questoes]=X[questoes].astype(int)
A = []    
for i in range(len(X)):
    df = X.iloc[i]
    for q in questoes:
        #df[q]= 0 if np.isnan(float(df[q])) else float(df[q])
        dic = dict(df)#[cabecalho])
        dic['Questao']  = q#.decode('latin1').encode('utf8')
        dic['Resposta'] = float(df[q])
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else str(df[q])
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else int(df[q])
        #print(df[q])
        dic['Resposta'] = int(df[q])#0 if np.isnan(df[q]) else int(df[q])
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else int(df[q])
        #dic['Resposta'] = 'NA' if df[q]==0 else int(df[q])
        A.append(dic)
    
A = pd.DataFrame(A)
lista_dep = A['Departamento'].dropna().unique(); lista_dep.sort()

#A['Questao (Alunos)']=A['Questao']
#A['Questão (Alunos)']=A['Questao']

#%%    
Y[questoes]=Y[questoes].fillna(0.)
Y[questoes]=Y[questoes].astype(int)
B = []    
for i in range(len(Y)):
    df = Y.iloc[i]
    for q in questoes_docentes:
        #df[q]= 0 if np.isnan(float(df[q])) else float(df[q])
        dic = dict(df)#[cabecalho_docente])
        dic['Questao']  = q#.decode('latin1').encode('utf8')
        #dic['Resposta'] = df[q]
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else str(df[q])
        dic['Resposta'] = int(df[q])#0 if np.isnan(df[q]) else int(df[q])
        #dic['Resposta'] =  'NA' if np.isnan(df[q]) else int(df[q])
        B.append(dic)
    
B = pd.DataFrame(B)
#B['Questao (Docentes)']=B['Questao']
#B['Questão (Docentes)']=B['Questao']

#A.dropna(inplace=True)
#A['Cursos Atendidos'] = [len(i.split('|')) for i in A['Cursos']]

#B['Resposta'].replace(['NA',0], inplace=True)

#aux_2=[]
#for i in A['Curso Aluno']:
#    aux_2.append(nomes_cursos[i])
#A['Curso Discente'] = aux_2
lista_dep = B['Departamento'].dropna().unique(); lista_dep.sort()
#%%    
#
# Ajuste da escala Likert
#
max_likert = 5
colnames=[str(i) for i in range(max_likert+1)]

#%%    

#fn1 = path+'departamentos_nomes_siglas.csv'
#lista_departamentos= pd.read_csv(fn1)
#
#for i in range(len(lista_departamentos)):
#    print lista_departamentos.iloc[i]['Nome']
#
#dep_list=[]    
#for i in A['Departamento']:
#    dep_list.append(dic[i])
#
#A['Departamento'] = dep_list 
#    
#%%
#pl.figure()
#sns.catplot(x='Questao', y='Resposta', data=A, kind='bar', aspect=2, 
#               color='seagreen', errwidth=1, capsize=0.1,)
#pl.savefig('resposta_questoes_geral.png', dpi=300,bbox_inches='tight')
#%%
##sns.catplot(x='Questao', y='Resposta', hue='Departamento', data=A, kind='bar', aspect=2.5, orient='v')
#
##A=A[A['Curso Aluno']=='08GV']
##sns.set(palette='Blues_d',)
#sns.set_palette("Set1", 15, .99)
#pl.figure()
#ct = pd.crosstab(A['Curso Nome'], A['Ano Ingresso'].astype('str'))
#ct = ct.T/ct.sum(axis=1).values
#
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
##g.set_title(u'Total de Questões respondidas')
#g.set_ylabel(u'Porcentagem de discentes participantes')
#g.legend(title='Ano de\nIngresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('ano_de_ingresso_geral.png', dpi=300,bbox_inches='tight')
##pl.show()
#%%
##sns.catplot(x='Questao', y='Resposta', hue='Departamento', data=A, kind='bar', aspect=2.5, orient='v')
##sns.set(palette='Blues_d',)
#sns.set_palette("Set1", 15, .99)
#pl.figure()
#ct = pd.crosstab(A['Curso Nome'], A['Tipo Ingresso'].astype('str'))
#ct = ct.T/ct.sum(axis=1).values
#g = ct.T.plot.bar(stacked=True, )
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
##g.set_title(u'Total de Questões respondidas')
#g.set_ylabel(u'Porcentagem de discentes participantes')
#g.legend(title='Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('tipo_de_ingresso_geral.png', dpi=300,bbox_inches='tight')
##pl.show()
#%%
#for d, df in A.groupby(['Código e-MEC']):
#    id_curso=str(d)#str(df['Código e-MEC'].unique())
#    print(d, id_curso)
#    
#    #id_curso=id_curso.replace('/','-')
#    
#    pl.figure()
#    df1=df.drop_duplicates(subset=['Aluno', 'Tipo Ingresso', 'Ano Ingresso'])
#    n_alunos=len(df1)
#    ct = pd.crosstab(df1['Tipo Ingresso'], df1['Ano Ingresso'].astype('str'))
#    n_colors=df1['Tipo Ingresso'].unique().shape[0]
#    sns.set_palette("Set1", n_colors=n_colors,)   
#    g = ct.plot.bar(stacked=True)
#    g.set_ylabel(u'Número de alunos')
#    if g.get_yticks().max()>=2:
#        g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
#    tit=re.sub('\n', ' ', d)+' - Total de alunos:'+str(n_alunos)
#    g.set_title(tit)
#    g.legend(title='Ano', )#loc='center left', bbox_to_anchor=(1.0, 0.5))
#    pl.savefig('ingresso_discentes_curso_ano_'+id_curso+'.png', dpi=300,bbox_inches='tight')
#    #ct = ct.T/ct.sum(axis=1).values
#    #g = ct.plot.bar(stacked=True)
#    #g.set_ylim([0,1])
#    ct = pd.crosstab(df1['Tipo Ingresso'], df1['Ano Ingresso'].astype('str'))
#    g = ct.T.plot.bar(stacked=True)
#    g.set_ylabel(u'Número de alunos')
#    if g.get_yticks().max()>=2:
#        g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
#    tit=re.sub('\n', ' ', d)+' - Total de alunos:'+str(n_alunos)
#    g.set_title(tit)
#    g.legend(title='Tipo de ingresso',)#loc='center left', bbox_to_anchor=(1.0, 0.5))
#    pl.savefig('ingresso_discentes_curso_tipo_'+id_curso+'.png', dpi=300,bbox_inches='tight')
#    #
#    n_colors=df1['Estado'].unique().shape[0]
#    #sns.set_palette("Set1", n_colors=n_colors,)   
#    ct=pd.crosstab(df1['Ano Ingresso'], df1['Estado']); 
#    g = ct.plot.bar(stacked=True, )
#    g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
#    tit=u'Estado de origem - Total de alunos:'+str(n_alunos)
#    g.set_title(tit)
#    g.set_ylabel(u'Número de discentes respondentes')
#    g.legend(title='Estado',)# loc='center left', bbox_to_anchor=(1.0, 0.5))
#    pl.savefig('quantitativos_estado_de_origem_'+id_curso+'.png', dpi=300,bbox_inches='tight')
#
#    ct=pd.crosstab(df1['Ano Ingresso'], df1['Bolsa Apoio']); 
#    g = ct.plot.bar(stacked=True, )    
#    g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
#    tit=u'Bolsas de Apoio - Total de alunos:'+str(n_alunos)
#    g.set_title(tit)
#    g.set_ylabel(u'Número de discentes com bolsa de apoio')
#    g.legend(title='Bolsa de Apoio',)# loc='center left', bbox_to_anchor=(1.0, 0.5))
#    pl.savefig('quantitativos_bolsa_de_apoio_'+id_curso+'.png', dpi=300,bbox_inches='tight')
#
#    pl.show()
#%%
#sns.set()

#n_colors=A['Questao'].unique().shape[0]
#sns.set_palette("Set1", n_colors=n_colors,)   
#
#pl.figure()
#ct = pd.crosstab(A['Questao'], A['Resposta'].astype('str'))
##ct.drop(['NA'], axis=1, inplace=True)
#ct = ct/(ct.sum(axis=1).mean())
##
#for i in colnames:
#    if not i in ct.columns:
#        ct[i]=0
#        
#ct=ct[np.sort(ct.columns)]
##
#g = ct.plot.bar(stacked=True)
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
##g.set_title(u'Total de Questões respondidas')
#g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
#g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('resposta_questoes_geral.png', dpi=300,bbox_inches='tight')
##pl.show()
    
#%%
#sns.set()
# 
#pl.figure()
#ct = pd.crosstab(B['Questao'], B['Resposta'].astype('str'))
##try:
##    ct.drop(['NA'], axis=1, inplace=True)
##except:
##    pass
#ct = ct/(ct.sum(axis=1).mean())
#g = ct.plot.bar(stacked=True)
##
#for i in colnames:
#    if not i in ct.columns:
#        ct[i]=0
#        
#ct=ct[np.sort(ct.columns)]
##
#g.set_ylim([0,1])
#g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
##g.set_title(u'Total de Questões respondidas')
#g.set_ylabel(u'Porcentagem de questões respondidas\npelos docentes')
#g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#pl.savefig('resposta_questoes_geral_docentes.png', dpi=300,bbox_inches='tight')
##pl.show()
    
#%%
#for d, df in A.groupby(['Código e-MEC']):
#    #d=d.replace('/','-')
#    print(d)
#    #df=df[df['Resposta']!='NA']
#    n_colors=df['Questao'].unique().shape[0]
#    sns.set_palette("Set2", n_colors=n_colors,)   

#    pl.figure()
#    ct = pd.crosstab(df['Questao'], df['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\n pelos alunos')
#    g.set_xlabel(u'')
#    #g.set_title(dic_cursos[d])
#    g.set_title(df['Curso Nome'].unique()[0])
#    pl.savefig('resposta_alunos_questoes_curso_'+str(d)+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
#    
#    #disciplinas_curso = df['Disciplina'].unique()
#    #l=[ disciplinas_curso[i] in B['Disciplina'] for i in range(len(disciplinas_curso)) ]
#
#    #professor_curso = df['Professor'].unique()
#    #l=[ professor_curso[i] in B['Professor'] for i in range(len(professor_curso)) ]
#    #print(l)
#    
#    lista_professores_curso = df['Professor'].unique()
#    df1 = pd.DataFrame()
#    for p in lista_professores_curso:
#        df1=df1.append(B[B['Professor'].values==p])
#        
#        
#    n_colors=df1['Questao'].unique().shape[0]
#    pl.figure()
#    ct = pd.crosstab(df1['Questao'], df1['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\n pelos docentes')
#    g.set_xlabel(u'')
#    #g.set_title(dic_cursos[d])
#    g.set_title(df['Curso Nome'].unique()[0])
#    pl.savefig('resposta_docentes_questoes_curso_'+str(d)+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
#        
#%%
#for d, df in B.groupby(['Código e-MEC']):
#    #d=d.replace('/','-')
#    print(d)
#    #df=df[df['Resposta']!='NA']
#    n_colors=df['Questao'].unique().shape[0]
#    #sns.set_palette("Set2", n_colors=n_colors,)   
#
#    pl.figure()
#    ct = pd.crosstab(df['Questao'], df['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\n pelos alunos')
#    g.set_xlabel(u'')
#    #g.set_title(dic_cursos[d])
#    g.set_title(df['Curso Nome'].unique()[0])
#    pl.savefig('resposta_alunos_questoes_curso_'+str(d)+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
#    
#    #disciplinas_curso = df['Disciplina'].unique()
#    #l=[ disciplinas_curso[i] in B['Disciplina'] for i in range(len(disciplinas_curso)) ]
#
#    #professor_curso = df['Professor'].unique()
#    #l=[ professor_curso[i] in B['Professor'] for i in range(len(professor_curso)) ]
#    #print(l)lista_professores_curso = df['Professor'].unique()
#    df1 = pd.DataFrame()
#    for p in lista_professores_curso:
#        df1=df1.append(B[B['Professor'].values==p])
#        
#    print(Z.shape) 
#    n_colors=df1['Questao'].unique().shape[0]
#    pl.figure()
#    ct = pd.crosstab(df1['Questao'], df1['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\n pelos alunos')
#    g.set_xlabel(u'')
#    #g.set_title(dic_cursos[d])
#    g.set_title(df['Curso Nome'].unique()[0])
#    pl.savefig('resposta_alunos_questoes_curso_'+str(d)+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
 
#%%
#for d, df in B.groupby(['Curso']):
#    print(d)
#    #try:
#    #    df=df[df['Resposta']!='NA']
#    #except:
#    #    pass
#
#    n_colors=B['Questao (Docentes)'].unique().shape[0]
#    sns.set_palette("Set1", n_colors=n_colors,)   
#
#    pl.figure()
#    ct = pd.crosstab(df['Questao (Docentes)'], df['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos docentes')
#    g.set_title(dic_cursos[d])
#    pl.savefig('resposta_questoes_curso_'+d+'_docentes.png', dpi=300,bbox_inches='tight')
#    #pl.show()
    
#%%
##
## Análise por departamento
##    
#for d, df in A.groupby(['Departamento']): 
#    print(d)
#    n_participantes =str(len(df['Aluno'].unique()))
#    n_disciplinas =str(len(df['Disciplina'].unique()))
#    periodo=str(df['Ano'].unique()[0])+'-'+str(df['Período'].unique()[0])
#    #df=df[df['Resposta']!='NA']
#    q='Questao'
#    n_colors=df[q].unique().shape[0]
#    sns.set_palette("Set1", n_colors=n_colors,)   

#    pl.figure()
#    ct = pd.crosstab(df[q], df['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    tit=d+' ('+periodo+')\n'+n_participantes+' alunos participantes'+'\n'+n_disciplinas+' disciplinas avaliadas'
#    g.set_title(tit)
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos {\\bf ALUNOS}')
#    #g.set_xlabel(u'Questão')
#    pl.savefig('analise_geral_departamento_'+d+'_alunos.png', dpi=300,bbox_inches='tight')
#    pl.show()
#   
#    for d1, df1 in df.groupby(['Disciplina']):
#        n_participantes_disciplina =str(len(df1['Aluno'].unique()))
#        pl.figure()
#        ct = pd.crosstab(df[q], df1['Resposta'].astype('str'))
#        ct = ct/(ct.sum(axis=1).mean())
#        #
#        for i in colnames:
#            if not i in ct.columns:
#                ct[i]=0
#                
#        ct=ct[np.sort(ct.columns)]
#        #
#        g = ct.plot.bar(stacked=True)
#        g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#        tit='Departamento: '+d+'\n'+'Disciplina '+d1+' -- ('+periodo+')'+'\n'+n_participantes_disciplina+' alunos responderam o questionário'
#        g.set_title(tit)
#        g.set_ylim([0,1])
#        g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#        g.set_ylabel(u'Porcentagem de questões respondidas\npelos {\\bf ALUNOS}')
#        g.set_xlabel(u'Questão (Alunos)')
#        pl.savefig('analise_disciplina_departamento_'+d+'_disciplina_'+d1+'_alunos.png', dpi=300,bbox_inches='tight')
#        pl.show()
#     
        
#%%    
#for d, df in B.groupby(['Departamento']): 
#    print(d)
#    n_participantes =str(len(df['Professor'].unique()))
#    n_disciplinas =str(len(df['Disciplina'].unique()))
#    periodo=str(df['Ano'].unique()[0])+'-'+str(df['Período'].unique()[0])
#    
#    #df=df[df['Resposta']!='NA']
#    q='Questao'
#    n_colors=df[q].unique().shape[0]
#    sns.set_palette("Set1", n_colors=n_colors,)   

#    pl.figure()
#    ct = pd.crosstab(df[q], df['Resposta'].astype('str'))
#    ct = ct/(ct.sum(axis=1).mean())
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', 
#             bbox_to_anchor=(1.0, 0.5),
#             )
#    tit=d+' ('+periodo+')\n'+n_participantes+' docentes participantes'+'\n'+n_disciplinas+' disciplinas avaliadas'
#    g.set_title(tit)
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos {\\bf DOCENTES}')
#    #g.set_xlabel(u'Questão')
#    pl.savefig('analise_geral_departamento_'+d+'_docentes.png', dpi=300,bbox_inches='tight')
#    pl.show()
#    
#    for d1, df1 in df.groupby(['Disciplina']):
#        n_participantes_disciplina =str(len(df1['Professor'].unique()))
#        pl.figure()
#        ct = pd.crosstab(df[q], df1['Resposta'].astype('str'))
#        ct = ct/(ct.sum(axis=1).mean())
#        #
#        for i in colnames:
#            if not i in ct.columns:
#                ct[i]=0
#                
#        ct=ct[np.sort(ct.columns)]
#        #
#        g = ct.plot.bar(stacked=True)
#        g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#        tit='Departamento: '+d+'\n'+'Disciplina '+d1+' -- ('+periodo+')'+'\n'+n_participantes_disciplina+' docentes responderam o questionário'
#        g.set_title(tit)
#        g.set_ylim([0,1])
#        g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#        g.set_ylabel(u'Porcentagem de questões respondidas\npelos {\\bf DOCENTES}')
#        g.set_xlabel(u'Questão (Docentes)')
#        pl.savefig('analise_disciplina_departamento_'+d+'_disciplina_'+d1+'_docentes.png', dpi=300,bbox_inches='tight')
#        pl.show()
    
#%%
#for d, df in A.groupby(['Questao']):
#    print(d)
#    #df=df[df['Resposta']!='NA']
#    n_colors=df['Resposta'].unique().shape[0]
#    sns.set_palette("Set1", n_colors=n_colors,)   
    
#    pl.figure()
#    ct = pd.crosstab(df['Departamento'], df['Resposta'].astype('str'))
#    for i in range(len(ct)):
#        ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()*100
#        
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(d)
#    g.set_ylim([0,100])
#    g.set_aspect(aspect=0.08,)
#    g.set_yticklabels(['{:.0f}%'.format(x*1) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
#    pl.savefig('resposta_departamento_questao_'+d+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
    
#%%
#for d, df in B.groupby(['Questao (Docentes)']):
#    print(d)
#    try:
#        df=df[df['Resposta']!='NA']
#    except:
#        pass
#    
#    pl.figure()
#    ct = pd.crosstab(df['Departamento'], df['Resposta'].astype('str'))
#    for i in range(len(ct)):
#        ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()*100
#        
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(d)
#    g.set_ylim([0,100])
#    g.set_aspect(aspect=0.08,)
#    g.set_yticklabels(['{:.0f}%'.format(x*1) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos docentes')
#    pl.savefig('resposta_departamento_questao_'+d+'_docentes.png', dpi=300,bbox_inches='tight')
#    #pl.show()
    
#%%
#for d, df in A.groupby(['Questao']):
#    print(d)
#    pl.figure()
#    #df=df[df['Resposta']!='NA']
#    ct = pd.crosstab(df['Curso Nome'], df['Resposta'].astype('str'))
#    for i in range(len(ct)):
#        ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()
#        
#    #
#    for i in colnames:
#        if not i in ct.columns:
#            ct[i]=0
#            
#    ct=ct[np.sort(ct.columns)]
#    #
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(d)
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()])
#    #g.set_aspect(aspect=0.2,)
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
#    pl.savefig('resposta_curso_questao_'+d+'.png', dpi=300,bbox_inches='tight')
#    pl.show()

#%%
#sns.set()
#for d, df in A.groupby(['Curso Nome']):
#    print('\n'*3)
#    ii=0
#    df.dropna(subset=['Departamento'], inplace=True)
    #df=df[df['Resposta']!='NA']
    # as loinhas comentadas abaixo geram muitos gráficos -- descomentar se necessário
#    for w, df1 in df.groupby('Disciplina'):
#        n_aluno_disc=df1['Aluno'].unique().shape[0]
#        dep_disc=df1['Departamento'].unique()[0]
#        n_prof_disc = df1['Professor'].unique().shape[0]
#        n_turma_disc=df1['Turma'].unique().shape[0]
#        ii+=1
#        print(ii, d,w, dep_disc, n_aluno_disc, )
#
#        #df1=df1[df1['Resposta']!='NA']
#        df1.dropna(subset=['Resposta'], inplace=True)
#        pl.figure()
#        ct = pd.crosstab(df['Questao'], df['Resposta'].astype('str'))
#        ct = ct/(ct.sum(axis=1).mean())
#        g = ct.plot.bar(stacked=True)
#        g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#        tit='Curso: '+d+', Disciplina: '+w+'\nDepartamento: '+dep_disc+', Total de Professores: '+str(n_prof_disc)+'\nTotal de alunos do curso: '+str(n_aluno_disc)+', No. de turmas: '+str(n_turma_disc)
#        g.set_title(tit)
#        g.set_ylim([0,1])
#        g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#        g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
#        g.set_xlabel(u'Questão (discente)')
#        pl.savefig('resposta_questoes_disciplina__'+str(w)+'__curso'+d+'.png', dpi=300,bbox_inches='tight')
#
#        pl.show()

#%%
#for d, df in A.groupby(['Departamento']):
#    print(d)
#    pl.figure()
#    sns.catplot(x='Questao', y='Resposta', data=df, kind='bar', aspect=2.5, 
#                   orient='v', color='gold', errwidth=1, capsize=0.1)#palette='Reds_d')
#    pl.title(d)
#    pl.savefig('resposta_questoes_departamento_'+d+'.png', dpi=300,bbox_inches='tight')
    
#%%
#for d, df in A.groupby(['Curso Aluno']):
#    print(d)
#    pl.figure()
#    sns.catplot(x='Questao', y='Resposta', data=df, kind='bar', aspect=2.5, 
#                   orient='v', color='magenta', errwidth=1, capsize=0.1)#palette='Reds_d')
#    pl.title(d)
#    pl.savefig('resposta_questoes_curso_'+d+'.png', dpi=300,bbox_inches='tight')
    #pl.show()
#%%
#for d, df in A.groupby(['Questao']):
#    print(d)
#    pl.figure()
#    sns.catplot(x='Departamento', y='Resposta', data=df, kind='bar', 
#                   aspect=4, orient='v', color='orange', errwidth=1, 
#                   capsize=0.1, order=lista_dep)#palette='Reds_d')
#    pl.xticks(rotation=90)
#    pl.savefig('resposta_questoes_Q'+d+'.png', dpi=300,bbox_inches='tight')
#    #pl.show()
#%%
#corr = X[questoes].corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
#
#f, ax = pl.subplots(figsize=(16, 16))
#
## Generate a custom diverging colormap
##cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
## Draw the heatmap with the mask and correct aspect ratio
#pl.figure(figsize=(16, 16))
#sns.heatmap(corr, mask=mask, #cmap=cmap, #vmin=.0, vmax=1., 
#            center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .8})
#
#pl.title(u"Correlação entre as questões (alunos)")
#pl.savefig('matriz_corr.png', dpi=300,bbox_inches='tight')
##pl.show()
#%%
#for d, df in A.groupby(['Curso Nome']):
#    print(d)
    
#    corr = df[questoes].corr()
#    mask = np.zeros_like(corr, dtype=np.bool)
#    mask[np.triu_indices_from(mask)] = True
#    corr=np.round(corr, decimals=2)
#    # Set up the matplotlib figure
#    
#    #f, ax = pl.subplots(figsize=(15, 15))
#    
#    # Generate a custom diverging colormap
#    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
#    
#    # Draw the heatmap with the mask and correct aspect ratio
#    pl.figure(figsize=(16, 16))
#    sns.heatmap(corr, mask=mask, #cmap=cmap, #vmin=.0, vmax=1., 
#                center=0, square=True, linewidths=.5, 
#                annot=True, 
#                cbar_kws={"shrink": .7}
#                )
#    
#    pl.title(u"Correlação entre as questões (alunos)\n"+d)
#    emec=df['Código e-MEC'].unique()[0]
#    pl.savefig('matriz_corr__'+str(emec)+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
#%%
#corr = Y[questoes_docentes].corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#
#f, ax = pl.subplots(figsize=(12, 12))
##cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
## Draw the heatmap with the mask and correct aspect ratio
#pl.figure(figsize=(12, 12))
#sns.heatmap(corr, mask=mask, #cmap=cmap, #vmin=.0, vmax=1., 
#            center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .8})
#
#pl.title(u"Correlação entre as questões (docentes)")
#pl.savefig('matriz_corr_docentes.png', dpi=300,bbox_inches='tight')
#%%
A['Cursos']=A['Curso Nome']

A['id_respondente']=A['Aluno']
B['id_respondente']=A['Professor']

#C = pd.concat([A,B],)
#C.drop(['Questao (Docentes)'], inplace=True,axis=1)

#%%
#for c, df1 in C.groupby(['Cursos',]):
#    Q=pd.DataFrame()
#    for d, df in df1.groupby(['Tipo Avaliacao']):
#        print(c,d)
#        df=df[df['Resposta']!='NA']
#        ct = pd.crosstab(index=df['Questao'], columns=[df['Resposta'].astype('str'),])
#        ct.index=[i+'('+d[0]+')' for i in ct.index]
#        #print(ct)    
#        for i in range(len(ct)):
#            ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()
#    
#        Q=pd.concat([Q,ct])    
#        
#    pl.figure()
#    g = Q.sort_index().plot.bar(stacked=True,figsize=(12,4))
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(c)
#    g.set_title(dic_cursos[c])
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()])
#    #g.set_aspect(aspect=0.2,)
#    g.set_ylabel(u'Porcentagem de questões respondidas')
#    g.set_xlabel('(A): alunos; (D): docentes')
#    pl.savefig('comparacao_vert_resposta_curso_'+c+'.png', dpi=300,bbox_inches='tight')
#    pl.show()
#    
#
#    pl.figure()
#    g = Q.sort_index().plot.barh(stacked=True,figsize=(4,12))
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(c)
#    g.set_title(dic_cursos[c])
#    g.set_xlim([0,1])
#    g.set_xticklabels(['{:.0f}%'.format(x*100./5) for x in g.get_yticks()])
#    #g.set_aspect(aspect=0.2,)
#    g.set_xlabel(u'Porcentagem de questões respondidas')
#    g.set_ylabel('(A): alunos; (D): docentes')
#    pl.savefig('comparacao_hori_resposta_curso_'+c+'.png', dpi=300,bbox_inches='tight')
#    pl.show()

#%%
#for (c,d), df in C.groupby(['Cursos','Questao']):
#    print(d)
#    df=df[df['Resposta']!='NA']
#
#    ct = pd.crosstab(index=df['Tipo Avaliacao'], columns=[df['Resposta'].astype('str'),])
#    print(ct)
#    for i in range(len(ct)):
#        ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()
#        
#    pl.figure()
#    g = ct.plot.bar(stacked=True)
#    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(d)
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()])
##    #g.set_aspect(aspect=0.2,)
##    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
##    pl.savefig('resposta_curso_questao_'+d+'.png', dpi=300,bbox_inches='tight')
##    #pl.show()
#    
#%%
##
## Relatórios por curso
##
#dir_pdf='./relatorios_cursos_pdf'
#os.system('mkdir '+dir_pdf)
#
#for cod_emec, df1 in X.groupby(['Código e-MEC']):
#    nome_curso = df1['Curso Nome'].unique()[0]
#    ano = df1['Ano'].unique()[0] if len(df1['Ano'].unique())==1 else -1
#    df1[u'Período']=[int(i) for i in df1[u'Período']]
#    periodo=df1[u'Período'].unique()
#    n_alunos_respondentes  = df1.shape[0]
#    #n_prof_respondentes   = 
#    if len(periodo)==1:
#        periodo=periodo[0]
#    else:
#        print('Erro na contagem de perídos para o curso '+nome_curso)
#        break
#
#    print(cod_emec, nome_curso, ano, periodo, n_alunos_respondentes)
#
#    head =''
#    head+='\n'+'\documentclass[a4paper,10pt]{article}'
#    head+='\n'+'\\usepackage{ucs}'
#    head+='\n'+'\\usepackage[utf8]{inputenc}'
#    head+='\n'+'\\usepackage[brazil]{babel}'
#    head+='\n'+'\\usepackage{fontenc}'
#    head+='\n'+'\\usepackage{graphicx,tabularx}'
#    head+='\n'+'\\usepackage[]{hyperref}'
#    head+='\n'+'\sloppy'    
#    head+='\n'+'\date{Data de processamento: \\today}'    
#    
#    head+='\\begin{document}'
#    
#    head+='\n'+'\\author{Diretoria de Avaliação Institucional (DIAVI) \\\\ Universidade Federal de Juiz de Fora}'+'\n'
#    head+='\n'+'\\title{RELATÓRIO DE RESULTADOS DA AVALIAÇÃO DO CURSO DE '+nome_curso+'}'
#    head+='\n'+'\maketitle'
#
#    head+='\n'+'\section{INTRODUÇÃO}'    
#    head+='\n'+'Este relatório objetiva apresentar os resultados da avaliação de disciplinas do Curso \
#    de '+nome_curso+' da Universidade Federal de Juiz de Fora, código e-MEC '+str(cod_emec)+', realizada pela \
#    Diretoria de Avaliação Institucional e os encaminhamentos propostos a \
#    partir destes resultados.'    
#    
#    head+='\n'+''    
#    head+='\n'+'\\begin{center}'
#    head+='\n'+'\\begin{tabularx}{\linewidth}{r|X}'
#    head+='\n'+'\nPúblico-alvo:& Curso de '+nome_curso+'\\\\'
#    #head+='\n'+'\nCampus:& '+campus+'\\\\'
#    head+='\n'+'\nPeríodo de coleta de dados:& '+str(ano)+'/'+str(periodo)+'.'+'\\\\'
#    head+='\n'+'\nForma de aplicação:& Online, por meio do SIGA.'+'\\\\'
#    head+='\n'+'\nAlunos respondentes:& '+str(n_alunos_respondentes)+'\\\\'
#    #head+='\n'+'\nProfessores respondentes:& '+str(n_prof_respondentes)+'\\\\'
#    head+='\n'+'\end{tabularx}'
#    head+='\n'+'\end{center}'
#    head+='\n'+''    
#
#
#    head+='\n'+'\section{MÉTODOS}'    
#    head+='\n'+'Este relatório se refere ao período '+str(ano)+'/'+str(periodo)+', com base em dados \
#    coletados através da aplicação de instrumentos de avaliação via SIGA \
#    implementados pela Diretoria de Avaliação Institucional (DIAVI) da UFJF, em atendimento \
#    ao que estabelece a Lei Sinais e a Resolução Consu 13/2015 (UFJF), \
#    com objetivo de contribuir para a avaliação própria do curso de '+nome_curso+' (código e-MEC'\
#    +str(cod_emec)+'). Foram aplicados um instrumento para discentes e outro para docentes, ambos contendo \
#    15 questões versando sobre as disciplinas na modalidade presencial oferecidas pela UFJF no \
#    referido período, visando, especificamente, coletar impressões sobre: atuação docente, atuação discente, \
#    recursos empregados, qualidade da disciplina ministrada. \
#    As respostas foram colhidas entre os dias 19/07/2018 e 12/08/2018, com participação espontânea e garantia de\
#    sigilo de participantes e avaliados.'
#
#    
#    head+='\n'+'\section{FORMULÁRIO}'
#    
#    head+='\n'+'As seguintes questões foram {\\bf objeto de avaliação pelos discentes} através do SIGA.'
#    head+='\n'+''
#
##    P = pd.read_csv('lista_questoes_alunos_formulario.csv', sep=';', 
##                    header=None, error_bad_lines=False, encoding='latin-1')
#    P = pd.read_csv('lista_questoes_alunos.csv', sep=';', 
#                    header=None, error_bad_lines=False, encoding='latin-1')
#    head+='\n'+'\small{'    
#    head+='\n'+'\\begin{center}'
#    head+='\n'+'\\begin{tabularx}{\linewidth}{l|X}'
#    for i in range(len(P)):
#        a,b = P.iloc[i]
#        c='\\\\\\\\' if i!=len(P)-1 else ''
#        #c='\\\\\\hline' if i!=len(P)-1 else ''
#        head+='\n'+a+'&'+b+c
#    head+='\n'+'\end{tabularx}'
#    head+='\n'+'\end{center}'
#    head+='\n'+'}'    
#        
#  
#        
#    head+='\n'+'As questões abaixo foram {\\bf objeto de avaliação pelos docentes} através do SIGA.'
#    head+='\n'+''
#
##    P = pd.read_csv('lista_questoes_docentes_formulario.csv', sep=';', 
##                    header=None, error_bad_lines=False, encoding='latin-1')
#    P = pd.read_csv('lista_questoes_docentes.csv', sep=';', 
#                    header=None, error_bad_lines=False, encoding='latin-1')
#    head+='\n'+'\small{'    
#    head+='\n'+'\\begin{center}'
#    head+='\n'+'\\begin{tabularx}{\linewidth}{c|X}'
#    for i in range(len(P)):
#        a,b = P.iloc[i]
#        c='\\\\\\\\' if i!=len(P)-1 else ''
#        #c='\\\\\\hline' if i!=len(P)-1 else ''
#        head+='\n'+a+'&'+b+c
#    head+='\n'+'\end{tabularx}'
#    head+='\n'+'\end{center}'
#    head+='\n'+'}'
#
#
#    head+='\n'+'\section{RESPOSTAS}'
#    head+='\n'+"As questões podem ser respondidas com um número de 1 a 5 numa escala que vai de {\it Discordo Totalmente} (1) a {\it Concordo Totalmente} (5). Não sendo permitidas múltiplas respostas e sendo possível a alteração antes do envio do formulário. O valor 0 (zero) indica {\it Não se Aplica}."
#    head+='\n'+''    
#    head+='\n'+'\\begin{figure}[h]'
#    head+='\n'+'\centering'
#    head+='\n'+'\includegraphics[width=0.85\linewidth]{resposta_questoes_curso_'+str(cod_emec)+'}'
#    head+='\n'+'\caption{\label{fig:qalunos}Distribuição das respostas dos discentes para as questões apresentadas}'
#    head+='\n'+'\end{figure}'
#
#    head+='\n'+''    
#    head+='\n'+'\\begin{figure}[h]'
#    head+='\n'+'\centering'
#    head+='\n'+'\includegraphics[width=0.999\linewidth]{matriz_corr__'+str(cod_emec)+'}'
#    head+='\n'+'\caption{\label{fig:corr_alunos}Correlação das respostas dos discentes para as questões apresentadas}'
#    head+='\n'+'\end{figure}'
#
##    if n_prof_respondentes>0:
##    
##        head+='\n'+''
##        head+='\n'+'\\begin{figure}[h]'
##        head+='\n'+'\centering'
##        head+='\n'+'\includegraphics[width=0.85\linewidth]{resposta_questoes_curso_'+curso+'_docentes}'
##        head+='\n'+'\end{figure}'
##    
##        head+='\n'+''
##        head+='\n'+'\\begin{figure}[h]'
##        head+='\n'+'\centering'
##        head+='\n'+'\includegraphics[width=0.8\linewidth]{comparacao_hori_resposta_curso_'+curso+'}'
##        head+='\n'+'\caption{\label{fig:comp}Compararação das respostas dos docentes e alunos}'
##        head+='\n'+'\end{figure}'
##        head+='\n'+''
#    
#    if n_alunos_respondentes>0:
#    
#        head+='\n'+'\\begin{figure}[h]'
#        head+='\n'+'\centering'
#        head+='\n'+'\includegraphics[width=0.7\linewidth]{ingresso_discentes_curso_ano_'+str(cod_emec)+'}'
#        head+='\n'+'\caption{\label{fig:ingressoano} Perfil  dos alunos respondentes em função do tipo de ingresso.}'
#        head+='\n'+'\end{figure}'
#        head+='\n'+''
#        head+='\n'+'\\begin{figure}[h]'
#        head+='\n'+'\centering'
#        head+='\n'+'\includegraphics[width=0.99\linewidth]{ingresso_discentes_curso_tipo_'+str(cod_emec)+'}'
#        head+='\n'+'\caption{\label{fig:ingressoano} Perfil  dos alunos respondentes  em função do ano de ingresso.}'
#        head+='\n'+'\end{figure}'
#        head+='\n'+''
#        head+='\n'+'\\begin{figure}[h]'
#        head+='\n'+'\centering'
#        head+='\n'+'\includegraphics[width=0.99\linewidth]{quantitativos_estado_de_origem_'+str(cod_emec)+'}'
#        head+='\n'+'\caption{\label{fig:estadoano} Estado de origem dos alunos respondentes em função do ano de ingresso.}'
#        head+='\n'+'\end{figure}'
#        head+='\n'+''
#        head+='\n'+'\\begin{figure}[h]'
#        head+='\n'+'\centering'
#        head+='\n'+'\includegraphics[width=0.99\linewidth]{quantitativos_bolsa_de_apoio_'+str(cod_emec)+'}'
#        head+='\n'+'\caption{\label{fig:bolsaano} Bolsas de apoio para os alunos respondentes em função do ano de ingresso.}'
#        head+='\n'+'\end{figure}'
#
#
#    head+='\n\n'+'\end{document}'
#
#    fbase='relatorio_'+str(ano)+'_'+str(periodo)+'_codigo_emec_'+str(cod_emec)
#    ftex=fbase+'.tex'
#    fpdf=fbase+'.pdf'
#    with open(ftex, 'w') as f:  
#        f.write(head)  
#
#    f.close()
#
#    os.system('pdflatex '+ftex) 
#    os.system('mv '+fpdf+' '+dir_pdf) 
#    for ext in ['.log', '.aux', '.out',  ]:
#        os.system('mv '+fbase+ext+' /tmp') 
    
    
#%%
###############################################################################
#                                                                             #        
#                                                                             #        
# Relatórios por curso                                                        #
#                                                                             #        
#                                                                             #        
###############################################################################
dir_pdf='./relatorios_cursos_pdf'
os.system('mkdir '+dir_pdf)

for cod_emec, df1 in A.groupby(['Código e-MEC']):
    nome_curso = df1['Curso Nome'].unique()[0]
    ano = df1['Ano'].unique()[0] if len(df1['Ano'].unique())==1 else -1
    aux=[int(i) for i in df1[u'Período']]
    df1[u'Período']=aux
    periodo=df1[u'Período'].unique()
    if len(periodo)==1:
        periodo=periodo[0]
    else:
        print('Erro na contagem de perídos para o curso '+nome_curso)
        break
    #--
    lista_professores_curso     = df1['Professor'].unique()
    df2 = pd.DataFrame()
    for p in lista_professores_curso:
        df2=df2.append(B[B['Professor'].values==p])
    #--        
    n_alunos_respondentes       = df1['Aluno'].unique().shape[0]
    n_professores_avaliados     = df1['Professor'].unique().shape[0]
    n_professores_respondentes  = df2['Professor'].unique().shape[0]#sum([ p in B['Professor'].values for p in df1['Professor'].unique()])
    p_professores_respondentes = n_professores_respondentes/n_professores_avaliados*100
    #--    
    print(('%10s %25s %4d %2d \t\t| %3d %3d %3d %4.2f') %
            (cod_emec, nome_curso, ano, periodo, 
            n_alunos_respondentes, n_professores_avaliados, 
            n_professores_respondentes, p_professores_respondentes)
          )
    #--
    head =''
    head+='\n'+'\documentclass[a4paper,10pt]{article}'
    head+='\n'+'\\usepackage{ucs}'
    head+='\n'+'\\usepackage[utf8]{inputenc}'
    head+='\n'+'\\usepackage[brazil]{babel}'
    head+='\n'+'\\usepackage{fontenc}'
    head+='\n'+'\\usepackage{times}'
    head+='\n'+'\\usepackage{indentfirst}'
    head+='\n'+'\\usepackage{graphicx,tabularx}'
    head+='\n'+'\\usepackage[]{hyperref}'
    head+='\n'+'\sloppy'    
    head+='\n'+'\date{Data de processamento: \\today}'    
    
    head+='\\begin{document}'
    
    head+='\n'+'\\author{Diretoria de Avaliação Institucional (DIAVI) \\\\ Universidade Federal de Juiz de Fora}'+'\n'
    head+='\n'+'\\title{RELATÓRIO DE RESULTADOS DA AVALIAÇÃO DO CURSO DE '+nome_curso+'}'
    head+='\n'+'\maketitle'

    head+='\n'+'\section{INTRODUÇÃO}'    
    head+='\n'+'Este relatório visa apresentar os resultados da avaliação de disciplinas do Curso \
    de '+nome_curso+' da Universidade Federal de Juiz de Fora, código e-MEC '+str(cod_emec)+', realizada pela \
    Diretoria de Avaliação Institucional e os encaminhamentos propostos a \
    partir destes resultados.'    
    
    head+='\n'+''    
    head+='\n'+'\\begin{center}'
    head+='\n'+'\\begin{tabularx}{\linewidth}{r|X}'
    head+='\n'+'\nPúblico-alvo:& Curso de '+nome_curso+'\\\\'
    #head+='\n'+'\nCampus:& '+campus+'\\\\'
    head+='\n'+'\nPeríodo de coleta de dados:& '+str(ano)+'/'+str(periodo)+' '+'\\\\'
    head+='\n'+'\nForma de aplicação:& Online, por meio do SIGA'+'\\\\'
    head+='\n'+'\nAlunos respondentes:& '+str(n_alunos_respondentes)+'\\\\'
    head+='\n'+'\nProfessores avaliados:& '+str(n_professores_avaliados)+'\\\\'
    head+='\n'+'\nProfessores respondentes:& '+str(n_professores_respondentes)+'\\\\'
    #head+='\n'+'\n\\% Professores respondentes:& '+str("%5.2f" % p_professores_respondentes)+'\\\\'
    head+='\n'+'\end{tabularx}'
    head+='\n'+'\end{center}'
    head+='\n'+''    

    if n_alunos_respondentes<=1:
        head+=' '+'Para resguardar o sigilo dos participantes, os resultados relativos aos discentes serão omitidos na ausência de alunos respondentes ou na eventualidade de somente um aluno responder o questionário.'    

    

    head+='\n'+'\section{MÉTODOS}'    
    head+='\n'+'Este relatório se refere ao período '+str(ano)+'/'+str(periodo)+', com base em dados \
    coletados através da aplicação de instrumentos de avaliação via SIGA \
    implementados pela Diretoria de Avaliação Institucional (DIAVI) da UFJF, em atendimento \
    ao que estabelece a Lei Sinais e a Resolução Consu 13/2015 (UFJF), \
    com objetivo de contribuir para a avaliação própria do curso de '+nome_curso+' (código e-MEC'\
    +str(cod_emec)+'). Foram aplicados um instrumento para discentes e outro para docentes, ambos contendo \
    15 questões versando sobre as disciplinas na modalidade presencial oferecidas pela UFJF no \
    referido período, visando, especificamente, coletar impressões sobre: atuação docente, atuação discente, \
    recursos empregados, qualidade da disciplina ministrada. \
    As respostas foram colhidas entre os dias 19/07/2018 e 12/08/2018, com participação espontânea e garantia de\
    sigilo de participantes e avaliados.'

    
    head+='\n'+'\section{QUESTÕES APRESENTADAS NO FORMULÁRIO}'
    
    if n_alunos_respondentes<=1:
        head+='\n'+'{ \it O questionário discente foi omitido pois ocorreu uma das condições listadas a seguir: ausência de alunos respondentes, ou somente um aluno respondeu o questionário.}'
        head+='\n'+''
    else:
        head+='\n'+'As seguintes questões foram {\\bf objeto de avaliação pelos discentes} através do SIGA.'
        head+='\n'+''

        P = pd.read_csv('lista_questoes_alunos.csv', sep=';', 
                        header=None, error_bad_lines=False, encoding='latin-1')
        head+='\n'+'\\begin{center}'
        head+='\n'+'\small{'    
        head+='\n'+'\\begin{tabularx}{\linewidth}{l|X}'
        for i in range(len(P)):
            a,b = P.iloc[i]
            c='\\\\\\\\' if i!=len(P)-1 else ''
            #c='\\\\\\hline' if i!=len(P)-1 else ''
            head+='\n'+a+'&'+b+c
        head+='\n'+'\end{tabularx}'
        head+='\n'+'}'    
        head+='\n'+'\end{center}'
        
        
    head+='\n'+'As questões abaixo foram {\\bf objeto de avaliação pelos docentes} através do SIGA.'
    head+='\n'+''

#    P = pd.read_csv('lista_questoes_docentes_formulario.csv', sep=';', 
#                    header=None, error_bad_lines=False, encoding='latin-1')
    if n_professores_respondentes<=1:
        head+='\n'+'{ \it O questionário docente foi omitido pois ocorreu uma das condições listadas a seguir: ausência de professores respondentes, ou somente um professor respondeu o questionário.}'
        head+='\n'+''
    else:
        P = pd.read_csv('lista_questoes_docentes.csv', sep=';', 
                    header=None, error_bad_lines=False, encoding='latin-1')
        head+='\n'+'\small{'    
        head+='\n'+'\\begin{center}'
        head+='\n'+'\\begin{tabularx}{\linewidth}{c|X}'
        for i in range(len(P)):
            a,b = P.iloc[i]
            c='\\\\\\\\' if i!=len(P)-1 else ''
            #c='\\\\\\hline' if i!=len(P)-1 else ''
            head+='\n'+a+'&'+b+c
        head+='\n'+'\end{tabularx}'
        head+='\n'+'\end{center}'
        head+='\n'+'}'


    head+='\n'+'\section{RESPOSTAS}'
    head+='\n'+"As questões podem ser respondidas com um número de 1 a 5 numa escala que vai de {\it Discordo Totalmente} (1) a {\it Concordo Totalmente} (5). Não sendo permitidas múltiplas respostas e sendo possível a alteração antes do envio do formulário. O valor 0 (zero) indica {\it Não se Aplica}."
    head+='\n'+''          
    #--
    head+='\n'+''    
    head+='\n'+'\\begin{figure}[h]'
    head+='\n'+'\centering'
    cap='O gráfico da avaliação dos alunos não foi mostrado  por um dos motivos:  não houve avaliação por parte dos alunos ou somente um aluno realizou a avaliação. '
    if n_alunos_respondentes>1:
        cap=''
        pl.figure()
        ct = pd.crosstab(df1['Questao'], df1['Resposta'].astype('str'))
        ct = ct/(ct.sum(axis=1).mean())
        for i in colnames:
            if not i in ct.columns:
                ct[i]=0
                
        ct=ct[np.sort(ct.columns)]
        sns.set_palette("Set1", n_colors=6*len(colnames),) 
        g = ct.plot.bar(stacked=True)
        g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
        g.set_ylim([0,1])
        g.set_yticks(ticks=g.get_yticks())
        g.set_yticklabels(['{:.0f}\%'.format(x*100) for x in g.get_yticks()]) 
        g.set_ylabel(u'Porcentagem de questões respondidas\n pelos {\\bf ALUNOS}')
        g.set_xlabel(u'')
        g.set_title(nome_curso+' ('+str(ano)+'-'+str(periodo)+')')
        fn = 'resposta_alunos_questoes_curso_'+str(cod_emec)+'.png'
        pl.savefig(fn, dpi=300,bbox_inches='tight')
        pl.show()
        head+='\n'+'\includegraphics[width=0.999\linewidth]{'+fn +'}'
     
    if n_professores_respondentes>1:
        pl.figure()
        ct = pd.crosstab(df2['Questao'], df2['Resposta'].astype('str'))
        ct = ct/(ct.sum(axis=1).mean())
        for i in colnames:
            if not i in ct.columns:
                ct[i]=0
                
        ct=ct[np.sort(ct.columns)]
        sns.set_palette("Set1", n_colors=6*len(colnames),) 
        g = ct.plot.bar(stacked=True)
        g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
        g.set_ylim([0,1])
        g.set_yticks(ticks=g.get_yticks())
        g.set_yticklabels(['{:.0f}\%'.format(x*100) for x in g.get_yticks()]) 
        g.set_ylabel(u'Porcentagem de questões respondidas\n pelos {\\bf DOCENTES}')
        g.set_xlabel(u'')
        g.set_title(nome_curso+' ('+str(ano)+'-'+str(periodo)+')')
        fn1 = 'resposta_docentes_questoes_curso_'+str(cod_emec)+'.png'
        pl.savefig(fn1, dpi=300,bbox_inches='tight')
        pl.show()
        
    head+='\n'+'\includegraphics[width=0.999\linewidth]{'+fn1+'}'
    head+='\n'+'\caption{\label{fig:resposta_questoes_curso}Distribuição das respostas dos alunos e docentes para as questões apresentadas. '+cap+'}'
    head+='\n'+'\end{figure}'
    #--
    if n_alunos_respondentes>1:
        corr = df1[questoes].corr()#.fillna(-1)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        corr=np.round(corr, decimals=2)
        pl.figure(figsize=(10, 10))
        sns.heatmap(corr, mask=mask, cmap='jet', #cmap=cmap, #vmin=.0, vmax=1., 
                    center=0, square=True, linewidths=.5, 
                    annot=True, cbar=False, 
                    cbar_kws = dict(use_gridspec=False, location="right",  shrink=0.9)
                    )
        
        pl.title(u"Correlação entre as questões respondidas pelos {\\bf ALUNOS}\n"+nome_curso+'. ')
        gn='matriz_corr__alunos_'+str(cod_emec)+'.png'
        pl.savefig(gn, dpi=300,bbox_inches='tight')
        pl.show()
            
        head+='\n'+''    
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.999\linewidth]{'+gn+'}'
        head+='\n'+'\caption{\label{fig:corr_alunos}Correlação das respostas dos alunos do curso '+nome_curso+' para as questões apresentadas. . Entradas com em branco indicam que não havia informação suficiente para o cálculo das correlações.}'
        head+='\n'+'\end{figure}'
    #--
    if n_professores_respondentes>1:
        corr = df2[questoes].corr()#.fillna(-1)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        corr=np.round(corr, decimals=2)
        pl.figure(figsize=(10, 10))
        sns.heatmap(corr, mask=mask, cmap='jet', #cmap=cmap, #vmin=.0, vmax=1., 
                    center=0, square=True, linewidths=.5, 
                    annot=True, cbar=False, 
                    cbar_kws = dict(use_gridspec=False, location="right",  shrink=0.9)
                    )
        
        pl.title(u"Correlação entre as questões respondidas pelos {\\bf DOCENTES}\n"+nome_curso+'.')
        gn1='matriz_corr__docentes_'+str(cod_emec)+'.png'
        pl.savefig(gn1, dpi=300,bbox_inches='tight')
        pl.show()
            
        head+='\n'+''    
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.999\linewidth]{'+gn1+'}'
        head+='\n'+'\caption{\label{fig:corr_docentes}Correlação das respostas dos docentes do curso '+nome_curso+' para as questões apresentadas. . Entradas com em branco indicam que não havia informação suficiente para o cálculo das correlações.}'
        head+='\n'+'\end{figure}'
    #--
    if n_alunos_respondentes>1:
        df_aux=df1[['Aluno','Ano Ingresso','Tipo Ingresso', 'Estado', 'Bolsa Apoio']].drop_duplicates()
        sns.set_palette("Set1", n_colors=24)  
        
        pl.figure()
        ct = pd.crosstab(df_aux['Tipo Ingresso'], df_aux['Ano Ingresso'].astype('int').astype('str'))
        n_colors=df1['Tipo Ingresso'].unique().shape[0]+1
         
        g = ct.T.plot.bar(stacked=True)
        g.set_ylabel(u'Número de alunos')
        if g.get_yticks().max()>=2:
            g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
        tit=re.sub('\n', ' ', nome_curso)+' - Total de alunos:'+str(n_alunos_respondentes)
        g.set_title(tit)
        g.legend(title='Ano', loc='center left', bbox_to_anchor=(1.0, 0.5))
        fn = 'ingresso_discentes_curso_ano_'+cod_emec+'.png'
        pl.savefig(fn, dpi=300,bbox_inches='tight')
         
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.85\linewidth]{'+fn+'}'
        head+='\n'+'\caption{\label{fig:ingresso_ano} Perfil do ano de ingresso dos alunos respondentes em função do ano ingresso.}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
        #--
#        g = ct.plot.bar(stacked=True)
#        g.set_ylabel(u'Número de alunos')
#        if g.get_yticks().max()>=2:
#            g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
#        tit=re.sub('\n', ' ', nome_curso)+' - Total de alunos:'+str(n_alunos_respondentes)
#        g.set_title(tit)
#        g.legend(title='Tipo de ingresso',loc='center left', bbox_to_anchor=(1.0, 0.5))
#        fn = 'ingresso_discentes_curso_tipo_'+cod_emec+'.png'
#        pl.savefig(fn, dpi=300,bbox_inches='tight')
#    
#        head+='\n'+'\\begin{figure}[h]'
#        head+='\n'+'\centering'
#        head+='\n'+'\includegraphics[width=0.85\linewidth]{'+fn+'}'
#        head+='\n'+'\caption{\label{fig:tipo_ingresso_ano}  Perfil  do tipo de ingresso dos alunos respondentes  em função do ano de ingresso.}'
#        head+='\n'+'\end{figure}'
#        head+='\n'+''
        #--
        ct=pd.crosstab(df_aux['Ano Ingresso'].astype('int'), df_aux['Estado']); 
        g = ct.plot.bar(stacked=True, )
        g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
        tit=u'Estado de origem - Total de alunos:'+str(n_alunos_respondentes)
        g.set_title(tit)
        g.set_ylabel(u'Número de discentes respondentes')
        g.legend(title='Estado', loc='center left', bbox_to_anchor=(1.0, 0.5))
        fn='quantitativos_estado_de_origem_'+cod_emec+'.png'
        pl.savefig(fn, dpi=300,bbox_inches='tight')
     
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.85\linewidth]{'+fn+'}'
        head+='\n'+'\caption{\label{fig:estado_ano} Perfil do estado de origem dos alunos respondentes em função do ano de ingresso.}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
        #-- 
        ct=pd.crosstab(df_aux['Ano Ingresso'].astype('int'), df_aux['Bolsa Apoio']); 
        g = ct.plot.bar(stacked=True, )    
        g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
        tit=u'Bolsas de Apoio - Total de alunos:'+str(n_alunos_respondentes)
        g.set_title(tit)
        g.set_ylabel(u'Número de discentes com bolsa de apoio')
        g.legend(title='Bolsa de Apoio', loc='center left', bbox_to_anchor=(1.0, 0.5))
        fn='quantitativos_bolsa_de_apoio_'+cod_emec+'.png'
        pl.savefig(fn, dpi=300,bbox_inches='tight')
     
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.85\linewidth]{'+fn+'}'
        head+='\n'+'\caption{\label{fig:bolsa_ano} Perfil das bolsas de apoio  dos alunos respondentes em função do ano de ingresso.}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
    #--
    head+='\n\n'+'\end{document}'

    fbase='novo_relatorio_'+str(ano)+'_'+str(periodo)+'_codigo_emec_'+str(cod_emec)
    ftex=fbase+'.tex'
    fpdf=fbase+'.pdf'
    with open(ftex, 'w') as f:  
        f.write(head)  

    f.close()

    os.system('pdflatex '+ftex) 
    os.system('mv '+fpdf+' '+dir_pdf) 
    for ext in ['.log', '.aux', '.out',  ]:
        os.system('mv '+fbase+ext+' /tmp') 
    
    
#%%
###############################################################################
#                                                                             #        
#                                                                             #        
# Relatórios por departamento                                                 #
#                                                                             #        
#                                                                             #        
###############################################################################
dir_pdf='./relatorios_departamento_pdf'
os.system('mkdir '+dir_pdf)

fn1 = './aux/'+'departamentos_nomes_siglas.csv'
dataframe_departamentos= pd.read_csv(fn1, encoding = 'latin-1')

codigos_departamentos={}
for i in range(len(dataframe_departamentos)): 
    d, s = dataframe_departamentos.iloc[i]['Nome'], dataframe_departamentos.iloc[i]['Sigla']
    codigos_departamentos[d]=s

#lista_departamentos=['ANA', 'CAD', 'TRN']
#aux=pd.DataFrame()
#for p in lista_departamentos:
#    aux=aux.append( B[B['Departamento']==p])

for cod_dep, df1 in B.groupby(['Departamento']):
    ano = df1['Ano'].unique()[0] if len(df1['Ano'].unique())==1 else -1
    aux=[int(i) for i in df1[u'Período']]
    df1[u'Período']=aux
    periodo=df1[u'Período'].unique()
    #n_prof_respondentes   = 
    if len(periodo)==1:
        periodo=periodo[0]
    else:
        print('Erro na contagem de perídos para departamento '+cod_dep)
        break

    #--
    lista_disciplinas_departamento     = df1['Disciplina'].unique()
    df2 = pd.DataFrame()
    for p in lista_disciplinas_departamento:
        df2=df2.append(A[A['Disciplina'].values==p])
    #--        
    n_alunos_respondentes               = df2['Aluno'].unique().shape[0]
    n_disciplinas_avaliadas_alunos      = df2['Disciplina'].unique().shape[0]
    n_disciplinas_avaliadas_docentes    = df1['Disciplina'].unique().shape[0]
    n_professores_respondentes          = df1['Professor'].unique().shape[0]#sum([ p in B['Professor'].values for p in df1['Professor'].unique()])
    #p_professores_respondentes = n_professores_respondentes/n_professores_avaliados*100
    #--    
    print(('%10s %4d %2d \t\t| %3d %3d %3d %3d ') %
            (cod_dep, ano, periodo, 
            n_alunos_respondentes, n_disciplinas_avaliadas_alunos, 
            n_disciplinas_avaliadas_docentes, n_professores_respondentes,)
          )
    #-- 
    head =''
    head+='\n'+'\documentclass[a4paper,10pt]{article}'
    head+='\n'+'\\usepackage{ucs}'
    head+='\n'+'\\usepackage[utf8]{inputenc}'
    head+='\n'+'\\usepackage[brazil]{babel}'
    head+='\n'+'\\usepackage{a4wide}'
    head+='\n'+'\\usepackage{fontenc}'
    head+='\n'+'\\usepackage{times}'
    head+='\n'+'\\usepackage{indentfirst}'
    head+='\n'+'\\usepackage{graphicx,tabularx}'
    head+='\n'+'\\usepackage[]{hyperref}'
    head+='\n'+'\sloppy'    
    head+='\n'+'\date{Data de processamento: \\today}'    
    
    head+='\\begin{document}'
    
    head+='\n'+'\\author{Diretoria de Avaliação Institucional (DIAVI) \\\\ Universidade Federal de Juiz de Fora}'+'\n'
    head+='\n'+'\\title{RELATÓRIO DE RESULTADOS DA AVALIAÇÃO DE DEPARTAMENTO\\\\ Código do Departamento: '+cod_dep+'}'
    head+='\n'+'\maketitle'

    head+='\n'+'\section{INTRODUÇÃO}'    
    head+='\n'+'Este relatório objetiva apresentar os resultados da avaliação de disciplinas do Departamento \
    de código '+cod_dep+' da Universidade Federal de Juiz de Fora, realizada pela \
    Diretoria de Avaliação Institucional e os encaminhamentos propostos a \
    partir destes resultados.'    
    
    head+='\n'+''    
    head+='\n'+'\\begin{center}'
    head+='\n'+'\\begin{tabularx}{\linewidth}{X|l}'
    head+='\n'+'\nPúblico-alvo:& Departamento  '+cod_dep+'\\\\'
    #head+='\n'+'\nCampus:& '+campus+'\\\\'
    head+='\n'+'\nPeríodo de coleta de dados:& '+str(ano)+'/'+str(periodo)+'.'+'\\\\'
    head+='\n'+'\nForma de aplicação:& Online, por meio do SIGA.'+'\\\\'
    head+='\n'+'\nDocentes respondentes:& '+str(n_professores_respondentes)+'\\\\'
    head+='\n'+'\nNúmero de disciplinas do departamento avaliadas pelos Docentes:& '+str(n_disciplinas_avaliadas_docentes)+'\\\\'
    head+='\n'+'\nAlunos   respondentes:& '+str(n_alunos_respondentes)+'\\\\'
    head+='\n'+'\nNúmero de disciplinas do departamento  avaliadas pelos   Alunos:& '+str(n_disciplinas_avaliadas_alunos)+'\\\\'
    #head+='\n'+'\nProfessores respondentes:& '+str(n_prof_respondentes)+'\\\\'
    head+='\n'+'\end{tabularx}'
    head+='\n'+'\end{center}'
    head+='\n'+''    


    head+='\n'+'Os resultados relativos as respostas dos discentes serão omitidos na ausência de alunos respondentes, ou na eventualidade de somente um aluno responder o questionário discente.'    

    
    head+='\n'+'\section{MÉTODOS}'    
    head+='\n'+'Este relatório se refere ao período '+str(ano)+'/'+str(periodo)+', com base em dados \
    coletados através da aplicação de instrumentos de avaliação via SIGA \
    implementados pela Diretoria de Avaliação Institucional (DIAVI) da UFJF, em atendimento \
    ao que estabelece a Lei Sinais e a Resolução Consu 13/2015 (UFJF), \
    com objetivo de contribuir para a avaliação própria do departamento '+cod_dep+'.\
    Foram aplicados um instrumento para discentes e outro para docentes, ambos contendo \
    '+str(B['Questao'].unique().shape[0])+' questões versando sobre as disciplinas na modalidade presencial oferecidas pela UFJF no \
    referido período, visando, especificamente, coletar impressões sobre: atuação docente, atuação discente, \
    recursos empregados, qualidade da disciplina ministrada. \
    As respostas foram colhidas  \
    com participação espontânea e garantia de\
    sigilo de participantes e avaliados.'
    
    if n_alunos_respondentes<=1:
        head+=' '+'Para resguardar o sigilo dos participantes, os resultados relativos aos discentes serão omitidos na ausência de alunos respondentes ou na eventualidade de somente um aluno respondeu o questionário.'    

    
    head+='\n'+'\section{QUESTÕES APRESENTADAS NOS FORMULÁRIOS}'
    

    if n_alunos_respondentes<=1:
        head+='\n'+'{ \it O questionário discente foi omitido pois ocorreu uma das condições listadas a seguir: ausência de alunos respondentes, ou somente um aluno respondeu o questionário.}'
        head+='\n'+''
    else:
        head+='\n'+'As seguintes questões foram {\\bf objeto de avaliação pelos discentes} através do SIGA.'
        head+='\n'+''

        P = pd.read_csv('lista_questoes_alunos.csv', sep=';', 
                        header=None, error_bad_lines=False, encoding='latin-1')
        head+='\n'+'\\begin{center}'
        head+='\n'+'\small{'    
        head+='\n'+'\\begin{tabularx}{\linewidth}{l|X}'
        for i in range(len(P)):
            a,b = P.iloc[i]
            c='\\\\\\\\' if i!=len(P)-1 else ''
            #c='\\\\\\hline' if i!=len(P)-1 else ''
            head+='\n'+a+'&'+b+c
        head+='\n'+'\end{tabularx}'
        head+='\n'+'}'    
        head+='\n'+'\end{center}'
        
  
        
    head+='\n'+'As questões abaixo foram {\\bf objeto de avaliação pelos docentes} através do SIGA.'
    head+='\n'+''

#    P = pd.read_csv('lista_questoes_docentes_formulario.csv', sep=';', 
#                    header=None, error_bad_lines=False, encoding='latin-1')
    P = pd.read_csv('lista_questoes_docentes.csv', sep=';', 
                    header=None, error_bad_lines=False, encoding='latin-1')
    head+='\n'+'\\begin{center}'
    head+='\n'+'\small{'    
    head+='\n'+'\\begin{tabularx}{\linewidth}{c|X}'
    for i in range(len(P)):
        a,b = P.iloc[i]
        c='\\\\\\\\' if i!=len(P)-1 else ''
        #c='\\\\\\hline' if i!=len(P)-1 else ''
        head+='\n'+a+'&'+b+c
    head+='\n'+'\end{tabularx}'
    head+='\n'+'}'
    head+='\n'+'\end{center}'
    

    head+='\n'+'\section{RESPOSTAS}'
    head+='\n'+"As questões podem ser respondidas com um número de 1 a 5 numa escala desde {\it Discordo Totalmente} (1) a {\it Concordo Totalmente} (5). Não sendo permitidas múltiplas respostas e sendo possível a alteração antes do envio do formulário. O valor 0 (zero) indica {\it Não se Aplica}."
    head+='\n'+''    
    head+='\n'+'Os nomes das disciplinas foram alterados para garantir a segurança da informações de modo que não seja possível a identificação das turmas, dos alunos  e dos professores que responderam a avaliação.'    
    head+='\n'+''  
    
    #head+='\n'+'\subsection{Panorama de todas as disciplinas avaliadas no departamento}'

    head+='\n'+'\\begin{figure}[h]'
    head+='\n'+'\centering'
    
    pl.figure()
    ct = pd.crosstab(df1['Questao'], df1['Resposta'].astype('str'))
    ct = ct/(ct.sum(axis=1).mean())
    for i in colnames:
        if not i in ct.columns:
            ct[i]=0
            
    ct=ct[np.sort(ct.columns)]
    sns.set_palette("Set1", n_colors=6*len(colnames),) 
    g = ct.plot.bar(stacked=True)
    g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_ylim([0,1])
    g.set_yticklabels(['{:.0f}\%'.format(x*100) for x in g.get_yticks()]) 
    g.set_ylabel(u'Porcentagem de questões respondidas\n pelos {\\bf DOCENTES}')
    g.set_xlabel(u'')
    g.set_title(cod_dep+' ('+str(ano)+'-'+str(periodo)+')')
    fn='analise_geral_departamento_'+cod_dep+'_docentes'+'.png'
    pl.savefig(fn, dpi=300,bbox_inches='tight')
    pl.show()
    
    head+='\n'+'\includegraphics[width=0.85\linewidth]{'+fn+'}'
     
    if n_alunos_respondentes>1:
        pl.figure()
        ct = pd.crosstab(df2['Questao'], df2['Resposta'].astype('str'))
        ct = ct/(ct.sum(axis=1).mean())
        for i in colnames:
            if not i in ct.columns:
                ct[i]=0
                
        ct=ct[np.sort(ct.columns)]
        sns.set_palette("Set1", n_colors=6*len(colnames),) 
        g = ct.plot.bar(stacked=True)
        g.legend(title='Escala', loc='center left', bbox_to_anchor=(1.0, 0.5))
        g.set_ylim([0,1])
        g.set_yticklabels(['{:.0f}\%'.format(x*100) for x in g.get_yticks()]) 
        g.set_ylabel(u'Porcentagem de questões respondidas\n pelos {\\bf ALUNOS}')
        g.set_xlabel(u'')
        g.set_title(cod_dep+' ('+str(ano)+'-'+str(periodo)+')')
        fn='analise_geral_departamento_'+cod_dep+'_alunos'+'.png'
        pl.savefig(fn, dpi=300,bbox_inches='tight')
        pl.show()    
        head+='\n'+'\includegraphics[width=0.85\linewidth]{'+fn+'}'

    head+='\n'+'\caption{\label{fig:analise_geral_departamento}\
            Panorama geral das respostas de todas as  disciplinas do departamenbto para as questões apresentadas.}'
    head+='\n'+'\end{figure}'

    #head+='\n'+'\subsection{Correlação entre as respostas coletadas nos questionários}'
    #--
    corr = df1[questoes].corr()#.fillna(-1)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    corr=np.round(corr, decimals=2)
    pl.figure(figsize=(10, 10))
    sns.heatmap(corr, mask=mask, cmap='jet', #cmap=cmap, #vmin=.0, vmax=1., 
                center=0, square=True, linewidths=.5, 
                annot=True, cbar=False, 
                cbar_kws = dict(use_gridspec=False, location="right",  shrink=0.9)
                )
    
    pl.title(u"Correlação entre as questões respondidas pelos {\\bf DOCENTES}\n Departamento "+cod_dep+'. ')
    gn='matriz_corr__'+str(cod_dep)+'_docentes.png'
    pl.savefig(gn, dpi=300,bbox_inches='tight')
    pl.show()
        
    head+='\n'+''    
    head+='\n'+'\\begin{figure}[h]'
    head+='\n'+'\centering'
    head+='\n'+'\includegraphics[width=0.999\linewidth]{'+gn+'}'
    head+='\n'+'\caption{\label{fig:corr_docentes}Correlação das respostas dos professores do departamento '+cod_dep+'. Entradas com em branco indicam que não havia informação suficiente para o cálculo das correlações.}'
    head+='\n'+'\end{figure}'
    #--
    if n_alunos_respondentes>1:
        corr = df2[questoes].corr()#.fillna(-1)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        corr=np.round(corr, decimals=2)
        pl.figure(figsize=(10, 10))
        sns.heatmap(corr, mask=mask, cmap='jet', #cmap=cmap, #vmin=.0, vmax=1., 
                    center=0, square=True, linewidths=.5, 
                    annot=True, cbar=False, 
                    cbar_kws = dict(use_gridspec=False, location="right",  shrink=0.9)
                    )
        
        pl.title(u"Correlação entre as questões respondidas pelos {\\bf ALUNOS}\n Departamento "+cod_dep+'. ')
        gn1='matriz_corr__'+str(cod_dep)+'_alunos.png'
        pl.savefig(gn1, dpi=300,bbox_inches='tight')
        pl.show()
            
        head+='\n'+''    
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.999\linewidth]{'+gn1+'}'
        head+='\n'+'\caption{\label{fig:corr_alunos}Correlação das respostas dos alunos para o departamento '+cod_dep+'. Entradas com em branco indicam que não havia informação suficiente para o cálculo das correlações.}'
        head+='\n'+'\end{figure}'
    #--
   
    
    for disc in lista_disciplinas_departamento:
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        
        df_aux=df1[df1['Disciplina']==disc]
        q='Questao'
        n_colors=df_aux[q].unique().shape[0]
        sns.set_palette("Set1", n_colors=n_colors,)       
        pl.figure()
        ct = pd.crosstab(df_aux[q], df_aux['Resposta'].astype('str'))
        ct = ct/(ct.sum(axis=1).mean())           
        for i in colnames:
            if not i in ct.columns:
                ct[i]=0
            
        ct=ct[np.sort(ct.columns)]
        #
        g = ct.plot.barh(stacked=True)
        g.legend(title='Escala', loc='center left', 
                 bbox_to_anchor=(1.0, 0.5),
                 )
        g.set_xlim([0,1.0])
        g.set_xticks(ticks=g.get_xticks())
        g.set_xticklabels(labels=['{:.0f}\%'.format(x*100) for x in g.get_xticks()]) 
        g.set_aspect(0.08)
        tit=' '+disc+'\n Departamento: '+cod_dep+' ('+str(ano)+'-'+str(periodo)+')'#+'\nProfessores Respondentes: '+str(n_resp)+' '
        g.set_title(tit)
        g.set_xlabel(u'Porcentagem de questões respondidas\npelos {\\bf DOCENTES}')
        g.set_ylabel(u'Questão')
        fn='analise_disciplina_departamento_'+cod_dep+'_'+disc+'_docentes.png'
        pl.savefig(fn, dpi=300,bbox_inches='tight')
        pl.show()
        head+='\n'+'\includegraphics[width=0.485\linewidth]{'+fn+'}'
        cap='O gráfico da avaliação dos alunos não foi mostrado  por um dos motivos:  a disciplina não foi avaliada pelos alunos ou somente um aluno realizou a avaliação. '

        df_aux=df2[df2['Disciplina']==disc]        
        if len(df_aux)>0 and n_alunos_respondentes>1:
            cap=''
            n_alunos_resp_disciplina = df_aux['Aluno'].unique().shape[0]
            q='Questao'
            n_colors=df_aux[q].unique().shape[0]
            sns.set_palette("Set1", n_colors=n_colors,)       
            pl.figure()
            ct = pd.crosstab(df_aux[q], df_aux['Resposta'].astype('str'))
            ct = ct/(ct.sum(axis=1).mean())           
            for i in colnames:
                if not i in ct.columns:
                    ct[i]=0
                
            ct=ct[np.sort(ct.columns)]
            #
            g = ct.plot.barh(stacked=True)
            g.legend(title='Escala', loc='center left', 
                     bbox_to_anchor=(1.0, 0.5),
                     )
            g.set_xlim([0,1.0])
            g.set_xticks(ticks=g.get_xticks())
            g.set_xticklabels(labels=['{:.0f}\%'.format(x*100) for x in g.get_xticks()]) 
            g.set_aspect(0.08)
            tit=' '+disc+'\n Departamento: '+cod_dep+' ('+str(ano)+'-'+str(periodo)+')'#+'\nProfessores Respondentes: '+str(n_resp)+' '
            g.set_title(tit)
            g.set_xlabel(u'Porcentagem de questões respondidas\npelos {\\bf ALUNOS}')
            g.set_ylabel(u'Questão')
            fn='analise_disciplina_departamento_'+cod_dep+'_'+disc+'_alunos.png'
            pl.savefig(fn, dpi=300,bbox_inches='tight')
            pl.show()
            head+='\n'+'\includegraphics[width=0.485\linewidth]{'+fn+'}'


        head+='\n'+'\caption{\label{fig:analise_geral_departamento}\
                Distribuição das respostas para a disciplina '+str(disc)+'. '+cap+'}'
        head+='\n'+'\end{figure}'



    head+='\n\n'+'\end{document}'

    fbase='relatorio_'+str(ano)+'_'+str(periodo)+'_codigo_departamento_'+str(cod_dep)
    ftex=fbase+'.tex'
    fpdf=fbase+'.pdf'
    with open(ftex, 'w') as f:  
        f.write(head)  

    f.close()

    os.system('pdflatex '+ftex) 
    os.system('mv '+fpdf+' '+dir_pdf) 
    for ext in ['.log', '.aux', '.out',  ]:
        os.system('mv '+fbase+ext+' /tmp') 
    
    
#%%

#%%




