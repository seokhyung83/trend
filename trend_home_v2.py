import streamlit as st
from SessionState import get
import pandas as pd
import numpy as np
import datetime, pymysql
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from basic_models_v2 import *
from basic_models_v2 import clean_data, make_corpus, test_preprocessing, extract_noun, processing_final
from basic_models_v2 import draw_wordcloud
from basic_models_v2 import draw_network
from basic_models_v2 import draw_tfidf
from basic_models_v2 import show_article
from basic_models_v2 import show_spec
import itertools
import re, time
from collections import Counter
import networkx as nx
import itertools
import SessionState
import pandas as pd

with open('./data/ner_dic_july_0825.pkl', 'rb') as fp:
    ner_dic = pickle.load(fp)
with open('./data/ref_word.txt','rb') as f:
    ref_word=pickle.load(f)        

    
unique_ner_labels = list(set(ner_dic.values()))

es_ner = ['제품', '기업', '기술', '전지', '소재', '서비스']

def _call_db_info():
    return pymysql.connect(
        host = 'trend.cb7jqghocrtb.ap-northeast-2.rds.amazonaws.com',
        port= 3306,
        user = 'root',
        password='ensol2020!',
        db = 'trend',
        charset = 'utf8')

st.set_page_config(page_title='Trend Sensing', page_icon=":smile:", layout='wide', initial_sidebar_state='auto')


c0, c1, c2, c3, c4 = st.beta_columns((3,4, 1, 1, 1))

c1.markdown("<h1 style='text-align: center; color: #A50135;'>Trend Sensing</h1>", unsafe_allow_html=True)


s_date = st.sidebar.date_input('Start Date').strftime('%Y%m%d')
e_date = st.sidebar.date_input('End Date').strftime('%Y%m%d')
button1 = st.sidebar.empty()
text1 = st.sidebar.empty()
st.sidebar.text("")


st.warning('기간 설정 후 serach 버튼을 누르세요.')


ss_test = SessionState.get(button1 = False)




if button1.button('Search'):
    ss_test.button1 = True

if ss_test.button1:
    

    conn = _call_db_info()
    curs = conn.cursor()
    tmp_insert_sql = "select * from content where date>='%s' and date <='%s'"%(s_date, e_date)
    tmp_abs_sql = "select * from abstract where date>='%s' and date <='%s'"%(s_date, e_date)
    curs.execute(tmp_insert_sql)
    tmp_article = pd.DataFrame(curs.fetchall())
    curs.execute(tmp_abs_sql)
    tmp_abs = pd.DataFrame(curs.fetchall())
    conn.commit()
    conn.close()
    

    target_data = tmp_article.iloc[:,6]
    
    stime = time.time()
    processed_article, processed_sent = processing_final(target_data)
    st.write(time.time()-stime)
    
    st.subheader('**| 분석 기사: {}건 ({} ~ {})**'.format(len(target_data), s_date, e_date))

    r1c1,r1c2 = st.beta_columns((1,1))
    
    with r1c1.beta_container():
        stime = time.time()
        st.text("")
        st.subheader('**| 주요 키워드**')
        ner_selected = st.selectbox('entity 선택',['전체']+es_ner, key = '1')
        fig_colud = draw_wordcloud(processed_sent, ner_selected)
        st.plotly_chart(fig_colud, use_container_width=True)
        st.write(time.time()-stime)
    
    with r1c2.beta_container():
        
        st.text("")
        st.subheader('**| 키워드 네트워크**')
        keyword_selected = st.text_input("분석할 단어 입력")
        ner_selected2 = st.selectbox('entity 선택',['전체']+es_ner, key = '2')
        


        button2 = st.empty()
        text2 = st.empty()
        
        if button2.button('확인'):
            stime = time.time()
            try:
                fig_network =  draw_network(processed_sent, 1,  keyword_selected, ner_selected2)
                st.plotly_chart(fig_network, use_container_width=True)
                st.write(time.time()-stime)
            except:
                st.warning('해당 키워드와 연결된 {} 없습니다'.format(ner_selected))
        
        
        
        
    r2c1,r2c2 = st.beta_columns((1,1))
 
    
    with r2c1.beta_container():
        st.text("")
        st.subheader('**| 키워드 분석 그래프**')
        ner_selected = st.selectbox('entity 선택',['전체']+es_ner)
        
        stime = time.time()
        fig_tfidf, fig_date_tfidf, top20 = draw_tfidf(processed_sent, s_date, e_date, tmp_article, ner_selected)
        st.plotly_chart(fig_tfidf, use_container_width=True)
        st.write(time.time()-stime)
        
        
    with r2c2.beta_container():
        st.text("")
        st.subheader('**| 관련 기사**')
        keyword_selected_article = st.text_input("분석할 단어 입력", key='textinput_2')
        stime = time.time()
        if len(keyword_selected_article)==0 : 
            article_dic = show_article(tmp_article, processed_article, top20[0])
        else: 
            article_dic = show_article(tmp_article, processed_article, keyword_selected_article)
        

        
        for i, (k,v) in enumerate(article_dic.items()):
            if i<6:
                r2c2.markdown("**{}**".format(k))
                r2c2.markdown('-자세히 보기: {}'.format(v), unsafe_allow_html=True)
                r2c2.markdown("**********")
        st.write(time.time()-stime)
    r3c1,r3c2 = st.beta_columns((4,1))
    
    
    with r3c1.beta_container():


        st.text(" ")
        st.subheader('**| 키워드 변화 동향**')

        st.plotly_chart(fig_date_tfidf, use_container_width = True)
        
    with r3c2.beta_container():
        st.text(" ")
        st.subheader('**| 추천 키워드**')
        stime = time.time()
        recommand_word = [word for word in top20 if word not in ref_word]
        for reco_w in recommand_word:
            st.markdown("{}".format(reco_w))
#             st.markdown("**********")
        st.write(time.time()-stime)

    st.text(" ")
    st.subheader('**| Spec Search**')
    stime = time.time()
    spec_lst = show_spec(tmp_article)
    
    for tup in spec_lst:
        st.markdown("{}".format(tup[1]))
        st.markdown('-자세히 보기: {}'.format(tup[0]), unsafe_allow_html=True)
        st.markdown("**********")
    
    st.write(time.time()-stime)

