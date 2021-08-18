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


mecab = Mecab()

import plotly
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px
import random
import pickle
from kss import kss 
# ner_tagged_df = pd.read_csv('need_ner_v2_rslt.csv', encoding='utf-8')
# ner_dic = dict(zip(ner_tagged_df['bigram'], ner_tagged_df['ver2']))
with open('ner_dic_v3.pickle', 'rb') as fp:
    ner_dic = pickle.load(fp)
    
color_pallette = [ '#FAF0E6', '#9ACD32', '#006400', '#ff9eb8', '#7b76ab', '#f47a45',  '#6e6975' ,'#00b9cf',
                  '#00FA9A', '#7FFFD4', '#F0E68C', '#6B8E23', '#D3D3D3', '#696969',
 '#2F4F4F','#32977e', '#0055ff', '#ff9eb8', '#7b76ab', '#f47a45',  '#6e6975' ,'#00b9cf', '#DB7093',
 '#4169E1', '#696969', '#8A2BE2', '#F0FFF0', '#FF0000', '#90EE90', '#F5DEB3', '#8B0000', '#FFDAB9',
 '#E0FFFF', '#8FBC8F', '#2F4F4F', '#FF4500', '#483D8B''black', 'grey']
# unique_ner_labels = ner_tagged_df['ver2'].unique()
unique_ner_labels = list(set(ner_dic.values()))
ner_color_dic = dict(zip(unique_ner_labels, color_pallette))
es_ner = ['제품-B', '기업-B', '기술-B', '전지-B', '소재-B', 'ORGANIZATION']



es_ner = ['제품', '기업', '기술', '전지', '소재', '서비스']

def bigram_corpus(target_data):
    split_sentence =[s for article in [kss.split_sentences(sent) for sent in target_data.tolist()] for s in article] # 추가
    pos_tag = [[word for word, pos in mecab.pos(text.lower()) if pos in ['NNP','NNG', 'SL'] and len(word)>=2] for text in split_sentence]
    bigram_mod = Phrases.load("bigram_model.pkl")
    bigram_sent = [bigram_mod[sent] for sent in pos_tag]
    return bigram_mod, bigram_sent

def draw_wordcloud(bigram_sent, ner_selected):
    
    if ner_selected == '전체':
        ner_selected = es_ner
    else:
        pass 
    
    temp_ner = []

    for sent in bigram_sent:
        t_n = []

        for word in sent:
            if word in ner_dic.keys():
                if ner_dic[word] in ner_selected:
                    t_n.append(word)

            else:
                if word.split('_')[0] in ner_dic.keys():           
                    if ner_dic[word.split('_')[0]] in ner_selected:
                        t_n.append(word)
                else:
                    pass 

        temp_ner.append(t_n)
    

    

    word_freq = Counter([word for sent in temp_ner for word in sent])
    
    words =list(dict(word_freq.most_common(100)).keys()) 
    ner_tup = dict((w,tag)for w,tag in ner_dic.items() if w in words)
    
    colors = [ner_color_dic[v] for k,v in ner_tup.items()]

    real_freq =  np.array(list(dict(word_freq.most_common(100)).values()))
    reg_value = real_freq[0]/50
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

# def show_article(tmp_article, bigram_sent, word_list):
#     idx_lst = []
#     for i, sent in enumerate(bigram_sent):
#         if any(w for w in sent if w in word_list):
# #         if any(w for w in sent if w ==word):
#             idx_lst.append(i)
# #             article.append(tmp_article.iloc[i,4])
#     return tmp_article.iloc[idx_lst,:]


def show_article(tmp_abs, bigram_sent, top20, keyword):
    if keyword == '전체':
        word_list = top20
    else:
        word_list = list(keyword)
        
    idx_lst = []
    for i, sent in enumerate(bigram_sent):
        if any(w for w in sent if w in word_list):
#         if any(w for w in sent if w ==word):
            idx_lst.append(i)
#             article.append(tmp_article.iloc[i,4])
    return dict(zip([title for i,title in enumerate(tmp_abs[5].tolist()) if i in idx_lst],[title for i,title in enumerate(tmp_abs[4].tolist()) if i in idx_lst]))



def cal_tfidf(bigram_sent, ner_selected):
    
    if ner_selected == '전체':
        ner_selected = es_ner
    else:
        pass 

    temp_ner = []

    for sent in bigram_sent:
        for word in sent:
            if word in ner_dic.keys():
                temp_ner.append((word, ner_dic[word]))

            else:
                if word.split('_')[0] in ner_dic.keys():           
                    temp_ner.append((word,ner_dic[word.split('_')[0]]))
                else:
                    temp_ner.append((word,'O')) 


    temp_ner = dict(temp_ner)
    print(temp_ner)

#     _,bigram_sent = bigram_corpus(target_data)
    dct = Dictionary(bigram_sent)
    dct_dic = dict(dct)
    corpus = [dct.doc2bow(doc) for doc in bigram_sent]
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



def draw_tfidf(bigram_sent, bigram_mod, s_date, e_date, tmp_article, ner_selected):
    
#     bigram_mod,_ = bigram_corpus(target_data)
    
    max_tfidf, avg_tfidf, _, _ = cal_tfidf(bigram_sent, ner_selected)


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
        d_target = tmp_article[tmp_article.iloc[:,1]==d].iloc[:,4]
        d_pos_tag = [[word for word, pos in mecab.pos(text.lower()) if pos in ['NNP','NNG', 'SL'] and len(word)>=2] for text in d_target]
        d_bigram_sent = [bigram_mod[sent] for sent in d_pos_tag]
        _,d_tf_dic,_,_ = cal_tfidf(d_bigram_sent,ner_selected)
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
    
#     article = show_article(tmp_article, bigram_sent, top20)
    
    return fig, fig2, top20
    


def draw_network(bigram_sent, n ,keyword:'str', ner_selected):
    
    if ner_selected == '전체':
        ner_selected = es_ner
    else:
        pass 
    
    
    temp_ner = []

    for sent in bigram_sent:
        t_n = []

        for word in sent:
            if word in ner_dic.keys():
                if ner_dic[word] in ner_selected:
                    t_n.append(word)

            else:
                if word.split('_')[0] in ner_dic.keys():           
                    if ner_dic[word.split('_')[0]] in ner_selected:
                        t_n.append(word)
                else:
                    pass 

        temp_ner.append(t_n)
    
    bigram_sent = temp_ner
    
#     _, ngram_sent = bigram_corpus(target_data)
    unique_noun = list(set([word for sent in bigram_sent for word in sent]))
    
    noun_index = {noun: i for i, noun in enumerate(unique_noun)}
    occurs = np.zeros([len(bigram_sent), len(unique_noun)])
    for i, sent in enumerate(bigram_sent):
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

