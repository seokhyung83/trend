import streamlit as st
from SessionState import get
import pandas as pd
import numpy as np
import datetime, pymysql
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from basic_models import *
from basic_models import bigram_corpus
from basic_models import draw_wordcloud
from basic_models import draw_network
from basic_models import draw_tfidf
from basic_models import show_article
from basic_models import show_spec
import itertools
import re
from collections import Counter
import networkx as nx
import itertools
import SessionState
import pandas as pd

with open('ner_dic_v3.pickle', 'rb') as fp:
    ner_dic = pickle.load(fp)
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
ner_selected = st.sidebar.selectbox('entity 선택',['전체']+es_ner)

# s_date = c2.date_input('Start Date').strftime('%Y%m%d')
# e_date = c3.date_input('End Date').strftime('%Y%m%d')

# # submit = c4.button("Search")

# button1 = c4.empty()
# text1 = c4.empty()


st.warning('기간 설정 후 serach 버튼을 누르세요.')


ss_test = SessionState.get(button1 = False)



# if not button1.button('Search'):
#     st.markdown("<h1 style='text-align: center; color: grey;'>Search 버튼을 눌러주세요.</h1>", unsafe_allow_html=True)


if button1.button('Search'):
    ss_test.button1 = True

# if not ss_test.button1:
#     st.markdown('test')
    
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
    
#     st.write(tmp_article.iloc[[1,3,5],:])

#     try:  
    target_data = tmp_article.iloc[:,4]

    bigram_mod, bigram_sent = bigram_corpus(target_data)
    
    st.subheader('**| 분석 기사: {}건 ({} ~ {})**'.format(len(target_data), s_date, e_date))

    r1c1,r1c2 = st.beta_columns((1,1))
    
    with r1c1.beta_container():
        
        st.text("")
        st.subheader('**| 주요 키워드**')
#         ner_selected = st.selectbox('entity 선택',['전체']+es_ner)
        fig_colud = draw_wordcloud(bigram_sent, ner_selected)
        st.plotly_chart(fig_colud, use_container_width=True)
    
    with r1c2.beta_container():
        st.text("")
        st.subheader('**| 키워드 네트워크**')


        keyword_selected = st.text_input("분석할 단어 입력")

    #         ner_selected2 = st.multiselect('entity 선택', ['제품-B', '기업-B', '기술-B', '소재-B', '동향-B'])


    #         if not keyword_selected:
    #             st.warning(" ")

#         noccur_selected = st.slider('최소 동시 출현 횟수', min_value=1, max_value=20)

        button2 = st.empty()
        text2 = st.empty()

        if button2.button('확인'):
            try:
                fig_network =  draw_network(bigram_sent, 1,  keyword_selected, ner_selected)
                st.plotly_chart(fig_network, use_container_width=True)
            except:
                st.warning('해당 키워드와 연결된 {} 없습니다'.format(ner_selected))
        
        
    r2c1,r2c2 = st.beta_columns((1,1))
    #     row1_1, row1_2  = st.beta_columns((4,6))
    
    with r2c1.beta_container():
        st.text("")
        st.subheader('**| 키워드 분석 그래프**')

        fig_tfidf, fig_date_tfidf, top20 = draw_tfidf(bigram_sent, bigram_mod, s_date, e_date, tmp_article, ner_selected)
        st.plotly_chart(fig_tfidf, use_container_width=True)
        
    with r2c2.beta_container():
        st.text("")
        st.subheader('**| 관련 기사**')
        article_selected = st.selectbox('선택', ['전체']+ top20)
        article_dic = show_article(tmp_abs, bigram_sent, top20, article_selected)
        

        
        for i, (k,v) in enumerate(article_dic.items()):
            if i<6:
                r2c2.markdown("**{}**".format(k))
                r2c2.markdown('-자세히 보기: {}'.format(v), unsafe_allow_html=True)
                r2c2.markdown("**********")


#         article = show_article(tmp_article, bigram_sent, top20)
#         st.write(article)

    st.text(" ")
    st.subheader('**| 키워드 변화 동향**')
    st.plotly_chart(fig_date_tfidf, use_container_width = True)


    st.text(" ")
    st.subheader('**| Spec Search**')
    spec_lst = show_spec(tmp_abs, tmp_article)
    
    for tup in spec_lst:
        st.markdown("{}".format(tup[1]))
        st.markdown('-자세히 보기: {}'.format(tup[0]), unsafe_allow_html=True)
        st.markdown("**********")
   



#         except:
#             st.markdown('존재하지 않는 키워드입니다. 키워드를 수정해주세요.')

#     except IndexError:
#         st.warning('검색 기간을 수정해 주세요 .')

    
#     fig_network = draw_network(bigram_sent, 10, '배터리')
#     st.plotly_chart(fig_network, use_container_width=False)
    
