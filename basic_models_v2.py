import numpy as np
import pandas as pd
import os, re, kss
from collections import OrderedDict
from konlpy.tag import Mecab
import gensim
from gensim.models import Phrases
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from operator import itemgetter
from collections import Counter
from gensim.matutils import corpus2dense
import networkx as nx
from plotly.subplots import make_subplots
from datetime import date, timedelta
from pororo import Pororo


mecab = Mecab(dicpath='/home/work/mecab-ko-dic-2.1.1-20180720/')
parser = Pororo(task='dep_parse', lang='ko')
p_ner  = Pororo(task='ner', lang='ko')

import plotly
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px
import random
import pickle
from kss import kss 


with open('./data/ner_dic_july_0825.pkl', 'rb') as fp:
    ner_dic = pickle.load(fp)
    
    
with open('./data/ref_word.txt','rb') as f:
    ref_word=pickle.load(f)        

    
color_pallette = ['#f5da42', '#e03498', '#ff9500', '#ff9500', '#2937ff', '#4df0ff', '#008062', '#a608d1', '#bf0f0f', '#120040']
es_ner = ['제품', '기업', '기술', '전지', '소재', '서비스', '사건/사고', '이벤트', '트렌드', '스펙']

ner_color_dic = dict(zip(es_ner, color_pallette))



es_ner = ['제품', '기업', '기술', '전지', '소재', '서비스', '사건/사고', '이벤트', '트렌드', '스펙']

# target_data = tmp_article.iloc[:,4] 형태로 들어옴 


def extract_parenthese(str):
    items_lst = re.findall('\(([^)]+)', str) #extracts string in () 
    newList = [x for x in items_lst if len(x)>=2] # more than 2
    return newList

def extract_quotes(str):
    items_lst = re.findall('"([^"]*)"', str)
    return items_lst

def parentheses_(tmp_input_sent):    
    tmp_input_sent = re.sub(pattern='\(+', repl=' ', string=tmp_input_sent)#tmp_input_sent = re.sub(pattern='\(\(', repl='\(', string=tmp_input_sent)
    tmp_input_sent = re.sub(pattern='\)+', repl=' ', string=tmp_input_sent)#tmp_input_sent = re.sub(pattern='\)\)', repl='\)', string=tmp_input_sent)
    tmp_input_sent = re.sub(pattern=' +', repl=' ', string=tmp_input_sent)
    input_sent = re.sub(pattern='\\\\',   repl='', string=tmp_input_sent)
    return input_sent


def clean_data(text: str):
    
        
    step0_ptn= '[\'\‘\’]'
    step1_ptn= '[\u00a0\u3000①②③④⑤⑥⑦⑧⑨⑩』◦※→®↑↓‣★▶■△◇◆▲○●\{\}\[\]\/?,+;:‧·ᆞ…》ⓒ|*~`\""“”!^_<>@\#&\\\=\'\n]'     
    step2_ptn= '[\.]' 

    text = text.upper()
    text = re.sub(pattern=step0_ptn, repl='\"', string=text)#.replace('\'‘’', '\"')
    text = re.sub(pattern=step1_ptn, repl='', string=text)
    tmp_sentence = kss.split_sentences(text)
    tmp_sentence = [re.sub(pattern=step2_ptn, repl='', string=s) for s in tmp_sentence]
    tmp_sentence = [parentheses_(s) for s in tmp_sentence] 
        
    return tmp_sentence

def make_corpus(target_data: str):
    #주식 기사 삭제
    sent = clean_data(target_data)
    sent = sent[0].split(' ')
    
    new_sent = []
    idx = 0

    for token in sent:
        if token[-2:] == '기자' :
            idx = sent.index(token) +1
            print(idx)
        else:
            pass
        
    new_sent.append(' '.join(sent[idx:]))

    return [sent for sent in new_sent if len(sent)>1 ][0]



def test_preprocessing(tmp_sent):
#     if len(tmp_sent)>120:
    tmp_lst = tmp_sent.strip().split(' ')
    chunk_lst = [' '.join(tmp_lst[i:i+20]) for i in range(0, len(tmp_lst), 20)]

    test1 = []
    test2 = []
    final_token = []

    for sent in chunk_lst: 


        tmp_step1 = parser(sent) #dependency parsing 

        tmp_step1_rslt  = []
        for x in tmp_step1:    

            if x[2] - x[0] == 1 and x[3] == 'NP': 
                tmp_step1_rslt.append(x[1])
            else:
                tmp_step1_rslt.append(x[1]+' ')

        tmp_sent1, tmp_rslt1 = '', [] # tmp_sent1은 dependency parsing 후 pororo ner 결과 하나의 토큰으로 묶이는 것을 기준으로 만든 문장 

        for x in p_ner(''.join(tmp_step1_rslt)):
            
    #             tmp_sent1.append(x[0].replace(' ', '')+' ') #뽀로로 ner에서 나온 결과 뭉치 단위로 붙여써주기
            if x[0] != " ":
                
                tmp_sent1 += x[0].replace(' ', '')
                tmp_sent1 += ' '
                tmp_rslt1.append(x[1])

            else:
                pass

        test1.append(tmp_rslt1)
        final_token.append(tmp_sent1.split(' ')[:-1]) # 끝에 있는 ''는 append 해주지 않음 

    return final_token


# def final_processed_sent(target_data):

#     p_final = []
#     for sent in target_data:
#         p = test_preprocessing(sent)
#         p_final.append([v for vs in p for v in vs])
#     return p_final


def extract_noun(processed_sent):
    noun = ['NNG', 'NNP', 'NNC', 'SL', 'SN', 'NR','NNBC', 'NNB', 'SH', 'XPN']
    stopwords = ['의', '후', '것', '인', '등', '데', '은', '과', '외', '사', '']
    pos_tag= [[mecab.pos(token) for token in sent] for sent in processed_sent]
    
    new_set = []
    for sent in pos_tag:
        new_sent_tokens = []
        for tokens in sent:
            new_token = ''
            for token in tokens: 
                if token[1] in noun:
                    new_token+=token[0]
            if new_token not in stopwords:
                new_sent_tokens.append(new_token)
        new_set.append(new_sent_tokens)
    return new_set


def processing_final(target_data):
    
    split_sents = [kss.split_sentences(article) for article in target_data]
    processed_article = []
    for article in split_sents:
        by_article = []
        for sent in article:
            processed = make_corpus(sent)
            processed = test_preprocessing(processed)
            processed = extract_noun(processed)
            by_article.append(processed)

        processed_article.append([w for sent in by_article for w in sent])
    processed_article_final = [[word for sent in article for word in sent] for article in processed_article]
    processed_sent_final = [sent for article in processed_article for sent in article]
    return processed_article_final, processed_sent_final


def draw_wordcloud(processed_sent, ner_selected):
    
    if ner_selected == '전체':
        ner_selected = es_ner
    else:
        pass 
    
    temp_ner = []

    for sent in processed_sent:
        t_n = []

        for word in sent:
            if word in ner_dic.keys():
                if ner_dic[word] in ner_selected:
                    t_n.append(word)

            else:
                pass

        temp_ner.append(t_n)
    
    
    
    word_freq = Counter([word for sent in temp_ner for word in sent])
    
    words =list(dict(word_freq.most_common(100)).keys()) 
    ner_tup = dict((w,tag)for w,tag in ner_dic.items() if w in words)
    
    colors = [ner_color_dic[v] for k,v in ner_tup.items()]

    real_freq =  np.array(list(dict(word_freq.most_common(100)).values()))
    reg_value = real_freq[0]/80
    weights = np.array(list(dict(word_freq.most_common(100)).values()))/reg_value
    
    if any(x < 1 for x in weights) ==True:
        weights= weights + 10

    


    data = go.Scatter(x=random.choices(range(len(words)), k=len(words)), #weight 기준으로 클때 가운데로 모이게 해야하나?????
                      y=random.choices(range(len(words)), k=len(words)),
                      mode='text',
                      text=list(ner_tup.keys()),
                      customdata = list(zip(ner_tup.values(),[dict(word_freq)[w] for w in ner_tup.keys()])),
                      hovertemplate = 
                    "<b>%{text}</b><br><br>" +
                    "entity:%{customdata[0]}<br>" +
                    "frequency:%{customdata[1]}<br>"+
                    "<extra></extra>",
                      marker={'opacity': 0.3},
                      textfont={'size': weights,
                               'color': colors})



    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})



    fig = go.Figure(data=[data], layout=layout)

    fig.update_layout(
        autosize=False,
        width=500,
        height=800,
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=4
        ),
        paper_bgcolor="white",
    )
        

    return fig








def show_article(tmp_article, processed_article_final,keyword):

    idx_lst = []
    for i, article in enumerate(processed_article_final):
        for token in article:
            if keyword in token:
                idx_lst.append(i)
                
                
    return dict(zip([title for i,title in enumerate(tmp_article[5].tolist()) if i in idx_lst],[title for i,title in enumerate(tmp_article[3].tolist()) if i in idx_lst]))

def show_spec(tmp_article):
    spec_search = ['km', 'kmh', 'nm', 'kw','gwh', 'whkg','acm2', 'lkm', 'kg', 'gkm',  'kmkwh', 'mah','kgfm','마력', 'whl','kml']

    target_data = tmp_article.iloc[:,6].tolist()
    ss = [kss.split_sentences(article) for article in target_data]
    final_sent = []
    for i, sents in enumerate(ss):
        temp_sent = []
        for sent in sents: 
            if any(ext in sent for ext in spec_search):
                if len(sent)<=200:
                    temp_sent.append(sent)

        final_sent.append(temp_sent)
    
    return list(zip([tmp_article.iloc[:,3][i] for i, v in enumerate(final_sent) if len(v)>=1], [v for i, v in enumerate(final_sent) if len(v)>=1]))


def cal_tfidf(processed_sent, ner_selected):
    
    if ner_selected == '전체':
        ner_selected = es_ner
    else:
        pass 

    temp_ner = []

    for sent in processed_sent:
        for word in sent:
            if word in ner_dic.keys():
                temp_ner.append((word, ner_dic[word]))

            else:

                temp_ner.append((word,'O')) 


    temp_ner = dict(temp_ner)



    dct = Dictionary(processed_sent)
    dct_dic = dict(dct)
    corpus = [dct.doc2bow(doc) for doc in processed_sent]
    model = TfidfModel(corpus)
    tfidf_dic = {}
    for c in corpus:
        vector = model[c]
        tfidf=[(dct_dic[v[0]],v[1]) for v in sorted(vector, key=itemgetter(1), reverse=True)]
        for tup in tfidf:
            if tup[0] in tfidf_dic.keys():
                tfidf_dic[tup[0]].append(tup[1])
            else:
                tfidf_dic[tup[0]] = []
                tfidf_dic[tup[0]].append(tup[1])

    max_tfidf=dict([(k,max(v)) for k, v in tfidf_dic.items() if temp_ner[k] in ner_selected])
    avg_tfidf = dict([(k,(sum(v)/len(v))) for k, v in tfidf_dic.items() if temp_ner[k] in ner_selected])
    max_tfidf_df = pd.DataFrame(dict(sorted(max_tfidf.items(), key=lambda item: item[1], reverse=True)), index=['max_tfidf']).T
    avg_tfidf_df = pd.DataFrame(dict(sorted(dict([(k,float(sum(v)/len(v))) for k, v in tfidf_dic.items()]).items(), key=lambda x: x[1], reverse=True)), index=['avg_tfidf']).T
    


    return max_tfidf, avg_tfidf, max_tfidf_df, avg_tfidf_df



def draw_tfidf(processed_sent, s_date, e_date, tmp_article, ner_selected):
    
#     bigram_mod,_ = bigram_corpus(target_data)
    
    max_tfidf, avg_tfidf, _, _ = cal_tfidf(processed_sent, ner_selected)


    y_avg = [v[1] for v in sorted(avg_tfidf.items(), key=lambda item: item[1], reverse=False)][-20:]
    y_max =[v[1] for v in sorted(max_tfidf.items(), key=lambda item: item[1], reverse=False)][-20:]
    x =[v[0] for v in sorted(max_tfidf.items(), key=lambda item: item[1], reverse=False)][-20:]


    # Creating two subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                        shared_yaxes=False, vertical_spacing=0.001)

    fig.append_trace(go.Bar(
        x=y_avg,
        y=x,
        marker=dict(
            color='rgba(50, 171, 96, 0.6)',
            line=dict(
                color='rgba(50, 171, 96, 1.0)',
                width=1),
        ),
        name='기간 평균 TF-IDF',
        orientation='h',
    ), 1, 1)

    fig.append_trace(go.Scatter(
        x=y_max, y=x,
        mode='lines+markers',
        line_color='rgb(128, 0, 128)',
        name='기간 최대 TF-IDF',
    ), 1, 2)

    fig.update_layout(
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            domain=[0, 0.85],
        ),
        yaxis2=dict(
            showgrid=False,
            showline=True,
            showticklabels=False,
            linecolor='rgba(102, 102, 102, 0.8)',
            linewidth=2,
            domain=[0, 0.85],
        ),
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
            domain=[0, 0.42],
        ),
        xaxis2=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
            domain=[0.47, 1],
            side='top',
            dtick=25000,
        ),
        legend=dict(x=0.0, y=1, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
    )

    annotations = []

    y_avg2 = np.round(y_avg, decimals=2)
    y_max2 = np.round(y_max, decimals = 2)

    # Adding labels
    for ydn, yd, xd in zip(y_max2, y_avg2, x):
        # labeling the scatter savings
        annotations.append(dict(xref='x2', yref='y2',
                                y=xd, x=ydn - 0.018,
                                text='{:,}'.format(ydn),
                                font=dict(family='Arial', size=12,
                                          color='rgb(128, 0, 128)'),
                                showarrow=False))
        # labeling the bar net worth
        annotations.append(dict(xref='x1', yref='y1',
                                y=xd, x=yd+0.05,
                                text=str(yd),
                                font=dict(family='Arial', size=12,
                                          color='rgb(50, 171, 96)'),
                                showarrow=False))
    # Source
    annotations.append(dict(xref='paper', yref='paper',
                            x=-0.05, y=-0.1,
                            text='TFIDF(Term Freqeuncy - Inverse Document Frequency): 기사 내에서 키워드가 갖는 특이성을 수치화 해주는 지표',
                            font=dict(family='Arial', size=10, color='rgb(150,150,150)'),
                            showarrow=False))

    fig.update_layout(annotations=annotations,     
                      autosize=False,
                      width=800,
                      height=800,)

    
    s_date_d =date(int(s_date[:4]),int(s_date[4:6]),int(s_date[6:8]))
    e_date_d =date(int(e_date[:4]),int(e_date[4:6]),int(e_date[6:8]))
    delta = e_date_d-s_date_d
    date_lst = []
    for i in range(delta.days +1):
        date_lst.append((s_date_d+timedelta(days=i)).strftime('%Y%m%d'))
    time_graph = []
    for d in date_lst:
        top20 = list(reversed(x)) #[v[0] for v in sorted(max_tfidf.items(), key=lambda item: item[1], reverse=True)][:20]
        top20_dic = {}
        for w in top20:
            top20_dic[w] = 0
        d_target = tmp_article[tmp_article.iloc[:,1]==d].iloc[:,6]       
        _, d_processed_sent = processing_final(d_target)    


        
#         d_corpus = make_corpus(d_target)
    
# #     processed_sent = final_processed_sent(corpus)
#         d_processed_sent = []
#         for sent in d_corpus:
#             d_p = test_preprocessing(sent)
#             d_processed_sent.append([v for vs in d_p for v in vs])


#         d_processed_sent = extract_noun(d_processed_sent)

        _,d_tf_dic,_,_ = cal_tfidf(d_processed_sent,ner_selected)
        for k,v in d_tf_dic.items():
            if k in top20_dic.keys():
                top20_dic[k] = v
        time_graph.append(top20_dic)
    date_tfidf_df = pd.DataFrame.from_dict(dict(zip(date_lst, time_graph)))
    date_tfidf_df['word'] = date_tfidf_df.index
    date_tfidf_df = date_tfidf_df.loc[:, (date_tfidf_df != 0).any(axis=0)].reset_index(drop=True)
    final_df = pd.melt(date_tfidf_df, id_vars=['word'], var_name='date', value_name='tfidf')
    fig2 = px.line(final_df, x="date", y="tfidf", color="word",
              line_group="word", hover_name="word")
    fig.update_layout(autosize=False,
                      width=1000,
                      height=800,)

    return fig, fig2, top20
    


def draw_network(processed_sent, n ,keyword:'str', ner_selected):
    
    if ner_selected == '전체':
        ner_selected = es_ner
    else:
        pass 
    
    
    temp_ner = []

    for sent in processed_sent:
        t_n = []

        for word in sent:
            if word in ner_dic.keys():
                if ner_dic[word] in ner_selected:
                    t_n.append(word)

            else:
                pass 

        temp_ner.append(t_n)
    

    
#     _, ngram_sent = bigram_corpus(target_data)
    unique_noun = list(set([word for sent in temp_ner for word in sent]))
    
    noun_index = {noun: i for i, noun in enumerate(unique_noun)}
    occurs = np.zeros([len(temp_ner), len(unique_noun)])
    for i, sent in enumerate(temp_ner):
        for word in sent:
            index = noun_index[word]  
            occurs[i][index] = 1
            
    co_occurs = occurs.T.dot(occurs)

    
    graph = nx.Graph()

    for i in range(len(unique_noun)):
        for j in range(i + 1, len(unique_noun)):
            if co_occurs[i][j] >= n:
                graph.add_edge(unique_noun[i], unique_noun[j], weight = co_occurs[i][j])
                graph.add_node(unique_noun[i], bipartite=0)
                graph.add_node(unique_noun[j], bipartite=1)

    keyword_list = []
    keyword_list.append(keyword)

    
    deg_l = {i: graph.degree(i) for i in keyword_list}

    highest_centrality_node = max(deg_l.items(), key=lambda x: x[1])[0]
    ego_G= nx.ego_graph(graph, highest_centrality_node)
    
    
    delete_node = []
    for node in ego_G.nodes():
        if node in ner_dic.keys():
            if ner_dic[node] not in ner_selected:
                delete_node.append(node)
        else:
            if node.split('_')[0] in ner_dic.keys():
                if ner_dic[node.split('_')[0]] not in ner_selected:
                    delete_node.append(node)

    
    for d_node in delete_node:
        if d_node == keyword:
            pass
        else:
            ego_G.remove_node(d_node)
        
                


#     print(list(ego_G.neighbors(keyword)))
#     print(ego_G.edges)
#     print(nx.get_edge_attributes(ego_G,'weight').values())
    
#     print(statistics.median(list(nx.get_edge_attributes(ego_G,'weight').values())))
    
    edge_weights = nx.get_edge_attributes(ego_G,'weight')
    
    no_of_occur = [ego_G.degree(v) for v in list(ego_G.nodes())]
    print(no_of_occur)
    
    
#     ego_G.remove_edges_from((e for e, w in edge_weights.items() if w < 2))
    print(len(ego_G.nodes))
    
    occur_dic = dict(zip(ego_G.nodes, no_of_occur))
    occur_dic = dict(sorted(occur_dic.items(), key=lambda item: item[1], reverse=True)[:30])
    
    delete_node2 = [k for k in ego_G.nodes if k not in occur_dic.keys()]
    
#     total_node = ego_G.nodes
    
    for d_node2 in delete_node2:
        if d_node2 == keyword:
            pass
        else:
            ego_G.remove_node(d_node2)
    
#     [k for k,v in occur_dic.items() if ]
#     labels = []
#     nx.set_node_attributes(ego_G, no_of_occur, "weight")
#     print(nx.get_node_attributes(ego_G, 'weight'))
#     print(ego_G.edges())
    
    pos = nx.spring_layout(ego_G)
#     print(pos)
    
    Xn=[pos[list(pos.keys())[k]][0] for k in range(len(pos))]
    Yn=[pos[list(pos.keys())[k]][1] for k in range(len(pos))]
    Xe=[]
    Ye=[]
    for e in ego_G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
    
    
#     no_of_occur = [ego_G.degree(v) for v in list(ego_G.nodes())]
#     no_of_occur = np.array(no_of_occur) + n
    

    pr = nx.pagerank(ego_G)
    
    node_sizes = list(pr.values())
    node_reg = 30/max(node_sizes)
    node_sizes = list(np.array(node_sizes) * node_reg)

    
    
    node_labels = list(pos.keys())

    node_colors= []
    ner_tag_lst = []
    print(ego_G)
    for word in list(ego_G.nodes()):
        ner_tag = ner_dic[word]
        ner_tag_lst.append(ner_tag)
        if word != keyword:
            node_colors.append(ner_color_dic[ner_tag])
        else:
            node_colors.append('#e73b3b')

    
    trace_nodes=dict(type='scatter',
                 x=Xn,
                 y=Yn,
                 mode='markers + text',
                 marker=dict(size=node_sizes, color=node_colors),
                 text=node_labels,
                 customdata = ner_tag_lst,
                 hovertemplate=
                "<b>%{text}</b><br><br>" +
                "entity:%{customdata}<br>" +
                "<extra></extra>",
                 hoverinfo='all')

    trace_edges=dict(type='scatter',
                 mode='lines',
                 x=Xe,
                 y=Ye,
                 line=dict(width=0.5, color='lightgrey'),
                 hoverinfo='skip') 
    
    layout=dict(title= None,
    font= dict(family='Nanum Gothic', size=10, color='black'),
    width=800,
    height=800,
    autosize=True,
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    hovermode='closest',
    plot_bgcolor='white',        
    )

    fig = go.Figure(data=[trace_edges, trace_nodes], layout=layout)
    
    

    return fig

