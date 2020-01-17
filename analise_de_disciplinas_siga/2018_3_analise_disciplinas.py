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
     'Docentes'     :'./data/RelatorioRespostasDocente_634a37c.csv',
     'Discentes'    :'./data/RelatorioRespostasAluno_8e9447e.csv',
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

#%%


#%%

#lista_cursos=['77A', '65A', '34A', '04A', '23A', '65B', '65D','15A', '64A', 
#                                                     '08GV', '87A']
#lista_cursos=[u'08GV', u'63E', u'88A', '73BL', '63B', '71A']
lista_cursos=[u'08GV', '71A']
#X.dropna(subset=['Curso Aluno'], inplace=True)
#lista_cursos=[i.replace('/','-') for i in X['Curso Aluno'].unique()]
#
lista_cursos=X['Curso Aluno'].dropna().unique()
lista_cursos=['66E']
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
Y['ind_disc']=Y['Disciplina'].values+Y['Turma'].values

X=X[X['Ano']!='REFERENCIA']
Y=Y[Y['Ano']!='REFERENCIA']
X['Ano'] = [int(i) for i in X['Ano'].values ] 
Y['Ano'] = [int(i)  for i in Y['Ano'].values ] 

X['Ano Ingresso'] = [int(i) for i in X['Ano Ingresso'].values ] 
#Y['Ano Ingresso'] = [int(i) for i in Y['Ano Ingresso'].values ] 

#%%

aux=[]
for i in lista_cursos:
    print(i)
    aux.append(X[X['Curso Aluno']==i])

X=pd.concat(aux)    
#%%

idx = Y.columns.isin(cabecalho)

cabecalho_docente=Y.columns[idx]
perguntas_docentes = Y.columns[~idx]
questoes_docentes = [ 'Q'+'{:02d}'.format(i+1) for i in range(len(perguntas_docentes))]


lista_questoes_alunos = pd.DataFrame(np.c_[questoes_docentes, perguntas_docentes])
lista_questoes_alunos.to_csv('lista_questoes_docentes.csv', encoding='latin-1',
                             header=False,index=False, sep=';')


C = Y[Y.columns[idx]]
Q = Y[Y.columns[~idx]]
Q.columns = [ 'Q'+'{:02d}'.format(i+1) for i in range(len(Q.columns))]
Y = pd.concat([C,Q], axis=1)
#%%
Z=[]
for c,df in Y.groupby(['Cursos']):
    aux=df.copy()
    print(c)
    for i in c.split('|'):
        aux['Cursos']=i
        Z.append(aux)

Z=pd.concat(Z)

Z['Curso']=Z['Cursos']

idx = Z['Curso'].isin(lista_cursos)
Z = Z[idx]

#%% 
list(X['Disciplina'].unique())

G = []
for a, df in X.groupby(['Curso Aluno']):
    num_alunos=df['Aluno'].unique().shape[0]
    lista_disciplinas=list(df['Disciplina'].unique())
    lista_disciplinas = np.unique(lista_disciplinas)
    ind_disc = df['ind_disc'].unique()
    num_disciplinas = len(lista_disciplinas)
    G.append({'Curso':a, 'Num. Alunos':num_alunos, 'Disciplinas':lista_disciplinas, 
              'Num. Disciplinas':num_disciplinas,
              'ind_disc':ind_disc})

G1=pd.DataFrame(G); G1.index=G1['Curso'].values
writer = pd.ExcelWriter('quantitativo_info_cursos.xlsx')
G1.drop(['ind_disc','Disciplinas', 'Num. Disciplinas'], axis=1).to_excel(writer,'Alunos por Disciplina', index=False)

H=[]
for c in G1['Curso']:
    for i in G1['ind_disc']:
        for j in i:
            df1=Z[(Z['ind_disc']==j) & (Z['Curso']==c)]
            if len(df1)>0:
                print(c,df1.Curso.unique(), len(df1))
                H.append(df1)
                aux=df1

H=pd.concat(H)

J=[]
for a, df in H.groupby(['Cursos']):
    dic={'Departamento':'Num. Departamentos','Professor':'Num. Professores', 
         'Disciplina':'Num. Disciplinas', 'ind_disc':'Num. Turmas'}
    aux={'Curso':a}
    for i in dic:
        print(a,dic[i],df[i].unique().shape[0])
        aux[dic[i]]=df[i].unique().shape[0]

    J.append(aux)
    
J1=pd.DataFrame(J); J1.index=J1['Curso'].values

J1.to_excel(writer,'Quantitativos por Curso', index=False)
writer.save()
#%%
fn1 = './aux/'+'departamentos_nomes_siglas.csv'
dataframe_departamentos= pd.read_csv(fn1, encoding = 'latin-1')

dic={}
for i in range(len(dataframe_departamentos)): 
    d, s = dataframe_departamentos.iloc[i]['Nome'], dataframe_departamentos.iloc[i]['Sigla']
    dic[d]=s
    
dep_list=[]    
for i in X['Departamento']:
    dep_list.append(dic[i])

X['Departamento'] = dep_list   

#%% 
dep_list=[]    
for i in Z['Departamento']:
    dep_list.append(dic[i])

Z['Departamento'] = dep_list    
#%% 
fn2 = './aux/'+'cursos_nomes_siglas.csv'
dataframe_cursos= pd.read_csv(fn2)

dic={}
for i in range(len(dataframe_cursos)): 
    d, s = dataframe_cursos.iloc[i]['CODIGO'], dataframe_cursos.iloc[i][u'NOME']
    dic[d]=s
    
dicionario_cursos=dic

#%%
aux=[]
for l in lista_cursos:
    df = X[X['Curso Aluno']==l]
    aux.append(df)
        
X = pd.concat(aux)
X = X.reindex()
lista_dep = X['Departamento'].dropna().unique(); lista_dep.sort()

dic_cursos={i:dicionario_cursos[i]+' ('+i+')' for k,i in enumerate(lista_cursos)}
nomes_cursos={}
for c in dic_cursos:
    nomes_cursos[c] = dic_cursos[c].replace('-','').replace('  ',' ').replace(' (',' (').replace(' ','\n').replace('/','-')
    nomes_cursos[c] = nomes_cursos[c].replace('EM\n','EM ').replace('DA\n','DA ')

tab_cursos=pd.DataFrame( [{' Sigla':i, 'Nome do Curso':dicionario_cursos[i]} for k,i in enumerate(lista_cursos)] )
print (tab_cursos)
tab_cursos.to_csv(path_or_buf='cursos_participantes.csv',index=False)

#%% 
tab=[]
info_aluno=[]
for i, df in X.groupby(['Curso Aluno']):
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
             'Curso':nomes_cursos[i],
             'IRA':df1['IRA'].unique()[0]}
        aluno.append(aux)
        info_aluno.append(aux)
        
    dic={'Curso':i, 'NA':n, 'IRA':fstat(df['IRA'])}    
    
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
#info_aluno=info_aluno[info_aluno['Sigla Curso']=='08GV']
#sns.set()    
ct=pd.crosstab(info_aluno['Curso'], info_aluno['Bolsa Apoio']); 
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
g.set_title(u'Bolsas de Apoio')
g.set_ylabel(u'Porcentagem de discentes\ncom bolsa de apoio')
g.legend(title='Bolsa de Apoio', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('quantitativos_bolsa_de_apoio_por_curso.png', dpi=300,bbox_inches='tight')
print(X['Tipos Bolsa'].unique())

#sns.set()
n_colors=info_aluno['Ano Ingresso'].unique().shape[0]
sns.set_palette("RdBu", n_colors=n_colors,)  
ct=pd.crosstab(info_aluno['Curso'], info_aluno['Ano Ingresso']); 
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
g.set_title(u'Distribuição do Ano de Ingresso')
g.set_ylabel(u'Porcentagem de discentes participantes')
g.legend(title='Ano de Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('quantitativos_ano_de_ingresso_por_curso.png', dpi=300,bbox_inches='tight')

#sns.set()
n_colors=info_aluno['Tipo Ingresso'].unique().shape[0]
sns.set_palette("cubehelix", n_colors=n_colors,)   
ct=pd.crosstab(info_aluno['Curso'], info_aluno['Tipo Ingresso']); 
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
g.set_title(u'Forma de ingresso dos discentes')
g.set_ylabel(u'Porcentagem de discentes participantes')
g.legend(title='Ano de Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('quantitativos_tipo_de_ingresso_por_curso.png', dpi=300,bbox_inches='tight')

#sns.set()
n_colors=info_aluno['NDspln'].unique().shape[0]
sns.set_palette("cubehelix", n_colors=n_colors,)   
ct=pd.crosstab(info_aluno['Curso'], info_aluno['NDspln']); 
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
g.set_title(u'Número de disciplinas')
g.set_ylabel(u'Porcentagem de discentes participantes')
g.legend(title='Número de\nDisciplinas', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('quantitativos_no_de_disciplinas_por_curso.png', dpi=300,bbox_inches='tight')

#sns.set()
n_colors=info_aluno['Estado'].unique().shape[0]
sns.set_palette("Set1", n_colors=n_colors,)   
ct=pd.crosstab(info_aluno['Curso'], info_aluno['Estado']); 
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
g.set_title(u'Estado de origem')
g.set_ylabel(u'Porcentagem de discentes participantes')
g.legend(title='Estado', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('quantitativos_estado_de_origem_por_curso.png', dpi=300,bbox_inches='tight')

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

print ("Número de docentes participantes por curso:")
tab_docentes=[]
for i, df in Z.groupby(['Curso']):
    n= df['Professor'].unique().shape[0]
    print (i,'\t',len(df)/n)
    tab_docentes.append({'NA':n, 'NF':len(df),
                'Curso':i})   
    
tab_docentes=pd.DataFrame(tab_docentes)
tab_docentes['Cursos']=[nomes_cursos[i] for i in tab_docentes['Curso']]


pl.figure()
g=sns.catplot(x='Cursos', y='NA', data=tab_docentes, kind='bar',palette='Greens_d',
               aspect=2, order=tab_docentes['Cursos'])
g.set_xticklabels(rotation=90)
g.set_xlabels('Curso (sigla)')
g.set_ylabels(u'Número de docentes participantes')
for p in g.ax.patches:
        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars

g.savefig('numero_participantes_curso_docentes.png', dpi=300,bbox_inches='tight')


#%%

print ("Número de participantes por curso:")
tab=[]
for i, df in X.groupby(['Curso Aluno']):
    n= df['Aluno'].unique().shape[0]
    print (i,'\t',len(df)/n)
    tab.append({'NA':n, 'NF':len(df),
                'MRA':len(df)/(1.0*n),
                'Curso':i})   
    
tab=pd.DataFrame(tab)
tab.to_csv('tabela_informacoes_2'+'.csv')
tab['Cursos']=[nomes_cursos[i] for i in tab['Curso']]

pl.figure()
g=sns.catplot(x='Cursos', y='NA', data=tab, kind='bar',palette='Blues_d',
               aspect=2, order=tab['Cursos'])
g.set_xticklabels(rotation=90)
g.set_xlabels('Curso (sigla)')
g.set_ylabels(u'Número de participantes')
for p in g.ax.patches:
        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars

g.savefig('numero_participantes_curso.png', dpi=300,bbox_inches='tight')




tab['Curso*'] = [dic_cursos[i]+' ('+i+')'  for i in tab['Curso']]
pl.figure()
g=sns.catplot(x='Cursos', y='NA', data=tab, kind='bar',palette='Blues_d',
               aspect=2, errwidth=0,)
g.set_xticklabels(rotation=90)
g.set_xlabels('Curso (sigla)')
g.set_ylabels(u'Número de participantes')
for p in g.ax.patches:
        g.ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
             ha='right', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')  #vertical bars

g.savefig('numero_participantes_curso_nome.png', dpi=300,bbox_inches='tight')


pl.figure()
ct = pd.crosstab(X['Aluno'], X['Disciplina'])
g=sns.distplot(ct.sum(), rug=True, kde=False, bins=30)
g.set_xlabel(u'Número de participantes por disciplina')
g.set_ylabel(u'Ocorrência')
g.set_xlim([1,ct.sum().max()])
pl.savefig('numero_participantes_disciplina_histograma.png', dpi=300,bbox_inches='tight')


#pl.figure()
#ct = pd.crosstab(X['Disciplina'], X['Aluno'])
#g=sns.distplot(ct.sum(), rug=True, kde=False, bins=30)
#g.set_xlabel(u'Número de participantes por disciplina')
#g.set_ylabel(u'Ocorrência')
#g.set_xlim([1,ct.sum().max()])
#pl.savefig('numero_participantes_disciplina_histograma.png', dpi=300,bbox_inches='tight')

aux_2=[]
for i in X['Curso Aluno']:
    aux_2.append(nomes_cursos[i])

X['Curso Discente'] = aux_2

pl.figure()
ct = pd.crosstab(X['Disciplina'], [X['Aluno'], X['Curso Discente']])
tab = ct.sum().unstack().mean()
print(tab)

pl.figure()
sns.set_palette("Blues_d", len(lista_cursos))
print(tab)
g=tab.plot(kind='bar')
g.set_ylabel(u'Número médio de formulários\nrespondidos por discente')  
g.set_xlabel('Curso')
pl.legend('')
for p in g.patches:
        g.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
             ha='right', va='center', rotation=90, xytext=(0, 15), textcoords='offset points')  #vertical bars

pl.savefig('numero_formularios_respondidos_discente_curso.png', dpi=300,bbox_inches='tight')

pl.figure()
sns.set_palette("Blues_d", len(lista_cursos))

tab_nome = tab.copy()
#tab_nome.index = [dic_cursos[i]+' ('+i+')'  for i in tab.index.values]
print(tab_nome)
g=tab_nome.plot(kind='bar')
g.set_ylabel(u'Número médio de formulários\nrespondidos por discente')  
g.set_xlabel('Curso')
pl.legend('')
for p in g.patches:
        g.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
             ha='right', va='center', rotation=90, xytext=(0, 15), textcoords='offset points')  #vertical bars

pl.savefig('numero_formularios_respondidos_discente_curso_nome.png', dpi=300,bbox_inches='tight')


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
A = []    
for i in range(len(X)):
    df = X.iloc[i]
    for q in questoes:
        dic = dict(df[cabecalho])
        dic['Questao']  = q#.decode('latin1').encode('utf8')
        dic['Resposta'] = df[q]
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else str(df[q])
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else int(df[q])
        dic['Resposta'] = 0 if np.isnan(df[q]) else int(df[q])
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else int(df[q])
        #dic['Resposta'] = 'NA' if df[q]==0 else int(df[q])
        A.append(dic)
    
A = pd.DataFrame(A)
#A.dropna(inplace=True)
A['Cursos Atendidos'] = [len(i.split('|')) for i in A['Cursos']]

aux_2=[]
for i in A['Curso Aluno']:
    aux_2.append(nomes_cursos[i])
    
A['Curso Discente'] = aux_2

A['Curso Aluno']=[i.replace('/','-') for i in A['Curso Aluno']]

#A['Resposta'].replace(['NA',0], inplace=True)

lista_dep = A['Departamento'].dropna().unique(); lista_dep.sort()

#%%    

B = []    
for i in range(len(Z)):
    df = Z.iloc[i]
    for q in questoes_docentes:
        dic = dict(df[cabecalho_docente])
        dic['Questao']  = q#.decode('latin1').encode('utf8')
        #dic['Resposta'] = df[q]
        #dic['Resposta'] = 'NA' if np.isnan(df[q]) else str(df[q])
        dic['Resposta'] = 0 if np.isnan(df[q]) else int(df[q])
        #dic['Resposta'] =  'NA' if np.isnan(df[q]) else int(df[q])
        B.append(dic)
    
B = pd.DataFrame(B)
B['Curso']=B['Cursos']
B['Questao (Docentes)']=B['Questao']
#A.dropna(inplace=True)
#A['Cursos Atendidos'] = [len(i.split('|')) for i in A['Cursos']]

#B['Resposta'].replace(['NA',0], inplace=True)

#aux_2=[]
#for i in A['Curso Aluno']:
#    aux_2.append(nomes_cursos[i])
#A['Curso Discente'] = aux_2
lista_dep = B['Departamento'].dropna().unique(); lista_dep.sort()
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

#sns.catplot(x='Questao', y='Resposta', hue='Departamento', data=A, kind='bar', aspect=2.5, orient='v')

#A=A[A['Curso Aluno']=='08GV']

#sns.set(palette='Blues_d',)
sns.set_palette("Set1", 15, .99)
pl.figure()
ct = pd.crosstab(A['Curso Discente'], A['Ano Ingresso'].astype('str'))
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Total de Questões respondidas')
g.set_ylabel(u'Porcentagem de discentes participantes')
g.legend(title='Ano de\nIngresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('ano_de_ingresso_geral.png', dpi=300,bbox_inches='tight')
#pl.show()
#%%
#sns.catplot(x='Questao', y='Resposta', hue='Departamento', data=A, kind='bar', aspect=2.5, orient='v')
#sns.set(palette='Blues_d',)
sns.set_palette("Set1", 15, .99)
pl.figure()
ct = pd.crosstab(A['Curso Discente'], A['Tipo Ingresso'].astype('str'))
ct = ct.T/ct.sum(axis=1).values
g = ct.T.plot.bar(stacked=True, )
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Total de Questões respondidas')
g.set_ylabel(u'Porcentagem de discentes participantes')
g.legend(title='Ingresso', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('tipo_de_ingresso_geral.png', dpi=300,bbox_inches='tight')
#pl.show()
#%%

for d, df in A.groupby(['Curso Discente']):
    print(d)
    id_curso=d.split('(')[1].split(')')[0]
    id_curso=id_curso.replace('/','-')
    pl.figure()
    df1=df.drop_duplicates(subset=['Aluno', 'Tipo Ingresso', 'Ano Ingresso'])
    n_alunos=len(df1)
    ct = pd.crosstab(df1['Tipo Ingresso'], df1['Ano Ingresso'].astype('str'))
    n_colors=df1['Tipo Ingresso'].unique().shape[0]
    sns.set_palette("Set1", n_colors=n_colors,)   
    g = ct.plot.bar(stacked=True)
    g.set_ylabel(u'Número de alunos')
    if g.get_yticks().max()>=2:
        g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
    tit=re.sub('\n', ' ', d)+' - Total de alunos:'+str(n_alunos)
    g.set_title(tit)
    g.legend(title='Ano',loc='center left', bbox_to_anchor=(1.0, 0.5))
    pl.savefig('ingresso_discentes_curso_ano_'+id_curso+'.png', dpi=300,bbox_inches='tight')
    #ct = ct.T/ct.sum(axis=1).values
    #g = ct.plot.bar(stacked=True)
    #g.set_ylim([0,1])
    ct = pd.crosstab(df1['Tipo Ingresso'], df1['Ano Ingresso'].astype('str'))
    g = ct.T.plot.bar(stacked=True)
    g.set_ylabel(u'Número de alunos')
    if g.get_yticks().max()>=2:
        g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
    tit=re.sub('\n', ' ', d)+' - Total de alunos:'+str(n_alunos)
    g.set_title(tit)
    g.legend(title='Tipo de ingresso',loc='center left', bbox_to_anchor=(1.0, 0.5))
    pl.savefig('ingresso_discentes_curso_tipo_'+id_curso+'.png', dpi=300,bbox_inches='tight')
    #
    n_colors=df1['Estado'].unique().shape[0]
    #sns.set_palette("Set1", n_colors=n_colors,)   
    ct=pd.crosstab(df1['Ano Ingresso'], df1['Estado']); 
    g = ct.plot.bar(stacked=True, )
    g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
    tit=u'Estado de origem - Total de alunos:'+str(n_alunos)
    g.set_title(tit)
    g.set_ylabel(u'Número de discentes respondentes')
    g.legend(title='Estado', loc='center left', bbox_to_anchor=(1.0, 0.5))
    pl.savefig('quantitativos_estado_de_origem_'+id_curso+'.png', dpi=300,bbox_inches='tight')

    ct=pd.crosstab(df1['Ano Ingresso'], df1['Bolsa Apoio']); 
    g = ct.plot.bar(stacked=True, )    
    g.set_yticklabels(['{:.0f}'.format(x*1) for x in g.get_yticks()]) 
    tit=u'Bolsas de Apoio - Total de alunos:'+str(n_alunos)
    g.set_title(tit)
    g.set_ylabel(u'Número de discentes com bolsa de apoio')
    g.legend(title='Bolsa de Apoio', loc='center left', bbox_to_anchor=(1.0, 0.5))
    pl.savefig('quantitativos_bolsa_de_apoio_'+id_curso+'.png', dpi=300,bbox_inches='tight')

    pl.show()
#%%
#sns.set()

n_colors=A['Questao'].unique().shape[0]
sns.set_palette("Set1", n_colors=n_colors,)   

pl.figure()
ct = pd.crosstab(A['Questao'], A['Resposta'].astype('str'))
#ct.drop(['NA'], axis=1, inplace=True)
ct = ct/(ct.sum(axis=1).mean())
g = ct.plot.bar(stacked=True)
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Total de Questões respondidas')
g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('resposta_questoes_geral.png', dpi=300,bbox_inches='tight')
#pl.show()
    
#%%
#sns.set()
n_colors=B['Questao (Docentes)'].unique().shape[0]
sns.set_palette("Set1", n_colors=n_colors,)   

pl.figure()
ct = pd.crosstab(B['Questao (Docentes)'], B['Resposta'].astype('str'))
#try:
#    ct.drop(['NA'], axis=1, inplace=True)
#except:
#    pass
ct = ct/(ct.sum(axis=1).mean())
g = ct.plot.bar(stacked=True)
g.set_ylim([0,1])
g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
#g.set_title(u'Total de Questões respondidas')
g.set_ylabel(u'Porcentagem de questões respondidas\npelos docentes')
g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.savefig('resposta_questoes_geral_docentes.png', dpi=300,bbox_inches='tight')
#pl.show()
    
#%%
for d, df in A.groupby(['Curso Aluno']):
    #d=d.replace('/','-')
    print(d)
    #df=df[df['Resposta']!='NA']
    n_colors=df['Questao'].unique().shape[0]
    sns.set_palette("Set1", n_colors=n_colors,)   

    pl.figure()
    ct = pd.crosstab(df['Questao'], df['Resposta'].astype('str'))
    ct = ct/(ct.sum(axis=1).mean())
    g = ct.plot.bar(stacked=True)
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_ylim([0,1])
    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
    g.set_ylabel(u'Porcentagem de questões respondidas')
    g.set_title(dic_cursos[d])
    pl.savefig('resposta_questoes_curso_'+d+'.png', dpi=300,bbox_inches='tight')
    #pl.show()
    
#%%
for d, df in B.groupby(['Curso']):
    print(d)
    #try:
    #    df=df[df['Resposta']!='NA']
    #except:
    #    pass

    n_colors=B['Questao (Docentes)'].unique().shape[0]
    sns.set_palette("Set1", n_colors=n_colors,)   

    pl.figure()
    ct = pd.crosstab(df['Questao (Docentes)'], df['Resposta'].astype('str'))
    ct = ct/(ct.sum(axis=1).mean())
    g = ct.plot.bar(stacked=True)
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_ylim([0,1])
    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
    g.set_ylabel(u'Porcentagem de questões respondidas\npelos docentes')
    g.set_title(dic_cursos[d])
    pl.savefig('resposta_questoes_curso_'+d+'_docentes.png', dpi=300,bbox_inches='tight')
    #pl.show()
    
#%%
for d, df in A.groupby(['Departamento']):
    print(d)
    #df=df[df['Resposta']!='NA']
    n_colors=df['Questao'].unique().shape[0]
    sns.set_palette("Set1", n_colors=n_colors,)   

    pl.figure()
    ct = pd.crosstab(df['Questao'], df['Resposta'].astype('str'))
    ct = ct/(ct.sum(axis=1).mean())
    g = ct.plot.bar(stacked=True)
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_title(d)
    g.set_ylim([0,1])
    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
    pl.savefig('resposta_questoes_departamento_'+d+'.png', dpi=300,bbox_inches='tight')
    #pl.show()
    
#%%
for d, df in A.groupby(['Questao']):
    print(d)
    #df=df[df['Resposta']!='NA']
    n_colors=df['Resposta'].unique().shape[0]
    sns.set_palette("Set1", n_colors=n_colors,)   
    
    pl.figure()
    ct = pd.crosstab(df['Departamento'], df['Resposta'].astype('str'))
    for i in range(len(ct)):
        ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()*100
        
    g = ct.plot.bar(stacked=True)
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_title(d)
    g.set_ylim([0,100])
    g.set_aspect(aspect=0.08,)
    g.set_yticklabels(['{:.0f}%'.format(x*1) for x in g.get_yticks()]) 
    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
    pl.savefig('resposta_departamento_questao_'+d+'.png', dpi=300,bbox_inches='tight')
    #pl.show()
    
##%%
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
#    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(d)
#    g.set_ylim([0,100])
#    g.set_aspect(aspect=0.08,)
#    g.set_yticklabels(['{:.0f}%'.format(x*1) for x in g.get_yticks()]) 
#    g.set_ylabel(u'Porcentagem de questões respondidas\npelos docentes')
#    pl.savefig('resposta_departamento_questao_'+d+'_docentes.png', dpi=300,bbox_inches='tight')
#    #pl.show()
    
#%%

for d, df in A.groupby(['Questao']):
    print(d)
    pl.figure()
    #df=df[df['Resposta']!='NA']

    ct = pd.crosstab(df['Curso Discente'], df['Resposta'].astype('str'))
    for i in range(len(ct)):
        ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()
        
    g = ct.plot.bar(stacked=True)
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_title(d)
    g.set_ylim([0,1])
    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()])
    #g.set_aspect(aspect=0.2,)
    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
    pl.savefig('resposta_curso_questao_'+d+'.png', dpi=300,bbox_inches='tight')
    #pl.show()

#%%
#sns.set()
for d, df in A.groupby(['Curso Aluno']):
    print('\n'*3)
    ii=0
    df.dropna(subset=['Departamento'], inplace=True)
    #df=df[df['Resposta']!='NA']
    for w, df1 in df.groupby('Disciplina'):
        n_aluno_disc=df1['Aluno'].unique().shape[0]
        dep_disc=df1['Departamento'].unique()[0]
        n_prof_disc = df1['Professor'].unique().shape[0]
        n_turma_disc=df1['Turma'].unique().shape[0]
        ii+=1
        print(ii, d,w, dep_disc, n_aluno_disc, )

        #df1=df1[df1['Resposta']!='NA']
        df1.dropna(subset=['Resposta'], inplace=True)
        pl.figure()
        ct = pd.crosstab(df['Questao'], df['Resposta'].astype('str'))
        ct = ct/(ct.sum(axis=1).mean())
        g = ct.plot.bar(stacked=True)
        g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
        tit='Curso: '+d+', Disciplina: '+w+'\nDepartamento: '+dep_disc+', Total de Professores:'+str(n_prof_disc)+'\nTotal de alunos do curso:'+str(n_aluno_disc)+', No. de turmas:'+str(n_turma_disc)
        g.set_title(tit)
        g.set_ylim([0,1])
        g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()]) 
        g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
        g.set_xlabel(u'Questão (discente)')
        pl.savefig('resposta_questoes_disciplina__'+str(w)+'__curso'+d+'.png', dpi=300,bbox_inches='tight')

        pl.show()

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
corr = X[questoes].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = pl.subplots(figsize=(12, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
pl.figure(figsize=(12, 12))
sns.heatmap(corr, mask=mask, cmap=cmap, #vmin=.0, vmax=1., 
            center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .8})

pl.title(u"Correlação entre as questões (alunos)")
pl.savefig('matriz_corr.png', dpi=300,bbox_inches='tight')
#pl.show()
#%%
corr = Z[questoes_docentes].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = pl.subplots(figsize=(12, 12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
pl.figure(figsize=(12, 12))
sns.heatmap(corr, mask=mask, cmap=cmap, #vmin=.0, vmax=1., 
            center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .8})

pl.title(u"Correlação entre as questões (docentes)")
pl.savefig('matriz_corr_docentes.png', dpi=300,bbox_inches='tight')
#%%
A['Cursos']=A['Curso Aluno']
C = pd.concat([A,B],)
C.drop(['Questao (Docentes)'], inplace=True,axis=1)

#%%
for c, df1 in C.groupby(['Cursos',]):
    Q=pd.DataFrame()
    for d, df in df1.groupby(['Tipo Avaliacao']):
        print(c,d)
        df=df[df['Resposta']!='NA']
        ct = pd.crosstab(index=df['Questao'], columns=[df['Resposta'].astype('str'),])
        ct.index=[i+'('+d[0]+')' for i in ct.index]
        #print(ct)    
        for i in range(len(ct)):
            ct.iloc[i] = ct.iloc[i]/ct.iloc[i].sum()
    
        Q=pd.concat([Q,ct])    
        
    pl.figure()
    g = Q.sort_index().plot.bar(stacked=True,figsize=(12,4))
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_title(c)
    g.set_title(dic_cursos[c])
    g.set_ylim([0,1])
    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()])
    #g.set_aspect(aspect=0.2,)
    g.set_ylabel(u'Porcentagem de questões respondidas')
    g.set_xlabel('(A): alunos; (D): docentes')
    pl.savefig('comparacao_vert_resposta_curso_'+c+'.png', dpi=300,bbox_inches='tight')
    pl.show()
    

    pl.figure()
    g = Q.sort_index().plot.barh(stacked=True,figsize=(4,12))
    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
    g.set_title(c)
    g.set_title(dic_cursos[c])
    g.set_xlim([0,1])
    g.set_xticklabels(['{:.0f}%'.format(x*100./5) for x in g.get_yticks()])
    #g.set_aspect(aspect=0.2,)
    g.set_xlabel(u'Porcentagem de questões respondidas')
    g.set_ylabel('(A): alunos; (D): docentes')
    pl.savefig('comparacao_hori_resposta_curso_'+c+'.png', dpi=300,bbox_inches='tight')
    pl.show()

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
#    g.legend(title='Escala\nLikert', loc='center left', bbox_to_anchor=(1.0, 0.5))
#    g.set_title(d)
#    g.set_ylim([0,1])
#    g.set_yticklabels(['{:.0f}%'.format(x*100) for x in g.get_yticks()])
##    #g.set_aspect(aspect=0.2,)
##    g.set_ylabel(u'Porcentagem de questões respondidas\npelos alunos')
##    pl.savefig('resposta_curso_questao_'+d+'.png', dpi=300,bbox_inches='tight')
##    #pl.show()
#    
#%%

    

    
#%%
dir_pdf='./relatorios_pdf'
os.system('mkdir '+dir_pdf)    
for curso in dic_cursos:
    print(curso)
    
    campus='Juiz de Fora' if len(curso.split('GV'))==1 else 'Governador Valadares'
    
    nome_curso, _ = dic_cursos[curso].split(' (')
    ano=int(A['Ano'].unique()[0])
    df1[u'Período']=[int(i) for i in df1[u'Período']]
    periodo=df1[u'Período'].unique()
    if len(periodo)==1:
        periodo=periodo[0]
    else:
        print('Erro na contagem de perídos para o curso '+nome_curso)
        break
        
    df1=C.loc[(C['Cursos']==curso)];
    
    n_alunos_respondentes  = df1.loc[df1['Tipo Avaliacao']=='ALUNO_TURMA']['Aluno'].unique().shape[0]
    n_prof_respondentes   = df1.loc[df1['Tipo Avaliacao']=='DOCENTE_TURMA']['Professor'].unique().shape[0]

    
    head =''
    head+='\n'+'\documentclass[a4paper,10pt]{article}'
    head+='\n'+'\\usepackage{ucs}'
    head+='\n'+'\\usepackage[utf8]{inputenc}'
    head+='\n'+'\\usepackage[brazil]{babel}'
    head+='\n'+'\\usepackage{fontenc}'
    head+='\n'+'\\usepackage{graphicx,tabularx}'
    head+='\n'+'\\usepackage[]{hyperref}'
    head+='\n'+'\sloppy'    
    head+='\n'+'\date{Data de processamento: \\today}'    
    
    head+='\\begin{document}'
    
    head+='\n'+'\\author{Diretoria de Avaliação Institucional (DIAVI) \\\\ Universidade Federal de Juiz de Fora}'+'\n'
    head+='\n'+'\\title{RELATÓRIO DE RESULTADOS DA AVALIAÇÃO DO CURSO DE '+nome_curso+'}'
    head+='\n'+'\maketitle'

    head+='\n'+'\section{INTRODUÇÃO}'    
    head+='\n'+'Este relatório objetiva apresentar os resultados da avaliação de disciplinas do Curso \
    de '+nome_curso+' da Universidade Federal de Juiz de Fora, campus '+campus+', realizada pela \
    Diretoria de Avaliação Institucional e os encaminhamentos propostos a \
    partir destes resultados.'    
    
    head+='\n'+''    
    head+='\n'+'\\begin{center}'
    head+='\n'+'\\begin{tabularx}{\linewidth}{r|X}'
    head+='\n'+'\nPúblico-alvo:& Curso de '+nome_curso+'\\\\'
    head+='\n'+'\nCampus:& '+campus+'\\\\'
    head+='\n'+'\nPeríodo de coleta de dados:& '+str(ano)+'/'+str(periodo)+'.'+'\\\\'
    head+='\n'+'\nForma de aplicação:& Online, por meio do SIGA.'+'\\\\'
    head+='\n'+'\nAlunos respondentes:& '+str(n_alunos_respondentes)+'\\\\'
    head+='\n'+'\nProfessores respondentes:& '+str(n_prof_respondentes)+'\\\\'
    head+='\n'+'\end{tabularx}'
    head+='\n'+'\end{center}'
    head+='\n'+''    


    head+='\n'+'\section{MÉTODOS}'    
    head+='\n'+'Este relatório se refere ao período '+str(ano)+'/'+str(periodo)+', com base em dados \
    coletados através da aplicação de instrumentos de avaliação via SIGA \
    implementados pela Diretoria de Avaliação Institucional (DIAVI) da UFJF, em atendimento \
    ao que estabelece a Lei Sinais e a Resolução Consu 13/2015 (UFJF), \
    com objetivo de contribuir para a avaliação própria do curso de '+nome_curso+' do Campus '\
    +campus+'. Foram aplicados um instrumento para discentes e outro para docentes, ambos contendo \
    15 questões versando sobre as disciplinas na modalidade presencial oferecidas pela UFJF no \
    referido período, visando, especificamente, coletar impressões sobre: atuação docente, atuação discente, \
    recursos empregados, qualidade da disciplina ministrada. \
    As respostas foram colhidas entre os dias 19/07/2018 e 12/08/2018, com participação espontânea e garantia de\
    sigilo de participantes e avaliados.'

    
    head+='\n'+'\section{FORMULÁRIO}'
    
    head+='\n'+'As seguintes questões foram {\\bf objeto de avaliação pelos discentes} através do SIGA.'
    head+='\n'+''

#    P = pd.read_csv('lista_questoes_alunos_formulario.csv', sep=';', 
#                    header=None, error_bad_lines=False, encoding='latin-1')
    P = pd.read_csv('lista_questoes_alunos.csv', sep=';', 
                    header=None, error_bad_lines=False, encoding='latin-1')
    head+='\n'+'\small{'    
    head+='\n'+'\\begin{center}'
    head+='\n'+'\\begin{tabularx}{\linewidth}{l|X}'
    for i in range(len(P)):
        a,b = P.iloc[i]
        c='\\\\\\\\' if i!=len(P)-1 else ''
        #c='\\\\\\hline' if i!=len(P)-1 else ''
        head+='\n'+a+'&'+b+c
    head+='\n'+'\end{tabularx}'
    head+='\n'+'\end{center}'
    head+='\n'+'}'    
        
  
        
    head+='\n'+'As questões abaixo foram {\\bf objeto de avaliação pelos docentes} através do SIGA.'
    head+='\n'+''

#    P = pd.read_csv('lista_questoes_docentes_formulario.csv', sep=';', 
#                    header=None, error_bad_lines=False, encoding='latin-1')
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
    head+='\n'+"As questões podem ser respondidas com um número de 1 a 5 numa escala que vai de {\it Discordo Totalmente} (1) a {\it Concordo Totalmente} (5). Não sendo permitidas múltiplas respostas e sendo possível a alteração antes do envio do formulário. O valor 0 (zero) indicam {\it Não se Aplica}."
    head+='\n'+''    
    head+='\n'+'\\begin{figure}[h]'
    head+='\n'+'\centering'
    head+='\n'+'\includegraphics[width=0.85\linewidth]{resposta_questoes_curso_'+curso+'}'
    head+='\n'+'\caption{\label{fig:qalunos}Distribuição das resposta dos discentes para as questões apresentadas}'
    head+='\n'+'\end{figure}'

    if n_prof_respondentes>0:
    
        head+='\n'+''
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.85\linewidth]{resposta_questoes_curso_'+curso+'_docentes}'
        head+='\n'+'\caption{\label{fig:qdocentes}Distribuição das respostas dos docentes}'
        head+='\n'+'\end{figure}'
    
        head+='\n'+''
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.8\linewidth]{comparacao_hori_resposta_curso_'+curso+'}'
        head+='\n'+'\caption{\label{fig:comp}Compararação das respostas dos docentes e alunos}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
    
    if n_alunos_respondentes>0:
    
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.7\linewidth]{ingresso_discentes_curso_ano_'+curso+'}'
        head+='\n'+'\caption{\label{fig:ingressoano} Perfil  dos alunos respondentes em função do tipo de ingresso.}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.99\linewidth]{ingresso_discentes_curso_tipo_'+curso+'}'
        head+='\n'+'\caption{\label{fig:ingressoano} Perfil  dos alunos respondentes  em função do ano de ingresso.}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.99\linewidth]{quantitativos_estado_de_origem_'+curso+'}'
        head+='\n'+'\caption{\label{fig:estadoano} Estado de origem dos alunos respondentes em função do ano de ingresso.}'
        head+='\n'+'\end{figure}'
        head+='\n'+''
        head+='\n'+'\\begin{figure}[h]'
        head+='\n'+'\centering'
        head+='\n'+'\includegraphics[width=0.99\linewidth]{quantitativos_bolsa_de_apoio_'+curso+'}'
        head+='\n'+'\caption{\label{fig:bolsaano} Bolsas de apoio para os alunos respondentes em função do ano de ingresso.}'
        head+='\n'+'\end{figure}'


    head+='\n\n'+'\end{document}'

    fbase='relatorio_'+str(ano)+'_'+str(periodo)+'_'+curso
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


