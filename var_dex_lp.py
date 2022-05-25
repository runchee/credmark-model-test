# pylint:disable=locally-disabled,missing-function-docstring,missing-class-docstring,missing-module-docstring

from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import altair as alt

from cmf import CmfRun

st.set_page_config(layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_onll6pxb.json')
st_lottie(lottie_book, speed=1, height=200, key="initial")

matplotlib.use("agg")

_lock = RendererAgg.lock

CREDMARK_GATEWAY = 'https://gateway.credmark.com/v1/model/run'
CREDMARK_GATEWAY_LOCAL = 'http://192.168.68.122:8700/v1/model/run'

gateway = st.selectbox('Gateway', [CREDMARK_GATEWAY_LOCAL, CREDMARK_GATEWAY])

cmf = CmfRun(gateway = CREDMARK_GATEWAY_LOCAL, block_number=14836288)

sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Test')

with row0_2:
    st.subheader('Token info')

    token_info,_,_ = cmf.token_info()
    price_info,_,_ = cmf.token_price()
    st.write('Price',
             price_info['price'],
             token_info['meta']['name'],
             token_info['meta']['decimals'],
             token_info['meta']['symbol'],
             token_info['meta']['total_supply'])

st.write('')
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.1, 1, .1, 1, .1))

curve_pools = \
    ["0xDC24316b9AE028F1497c275EB9192a3Ea0f67022",
    "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",
    "0xd632f22692FaC7611d2AA1C0D552930D43CAEd3B",
    "0xCEAF7747579696A2F0bb206a14210e3c9e6fB269",
    "0xD51a44d3FaE010294C616388b506AcdA1bfAAE46",
    "0xCEAF7747579696A2F0bb206a14210e3c9e6fB269",
    "0x5a6A4D54456819380173272A5E8E9B9904BdF41B",
    "0x93054188d876f558f4a66B2EF1d97d16eDf0895B",
    "0x2dded6Da1BF5DBdF597C45fcFaa3194e53EcfeAF",
    "0x9D0464996170c6B9e75eED71c68B99dDEDf279e8",
    "0xd658A338613198204DCa1143Ac3F01A722b5d94A"]

with row3_1, _lock:
    st.subheader('TVL - Curve')
    pool_address = st.selectbox('Which pool to show TVL?', curve_pools)

    curve_pool_info,_,_ = cmf.curve_pool_info(pool_address = pool_address)
    tvl_history,_,_ = cmf.run_historical(
                        model_slug='curve-fi.pool-info-tvl',
                        model_input = {'address':pool_address},
                        window = '90 days',
                        interval='1 day')

    df_tvl = CmfRun.to_dataframe(
                tvl_history['result']['series'],
                fields=[('tvl', lambda p: p['tvl']),
                        ('name', lambda p: p['name']),
                        ('tokens_symbol', lambda p: p['tokens_symbol']),])

    if not df_tvl.empty:
        # pylint:disable=locally-disabled,no-member
        fig = Figure()
        ax = fig.subplots()
        p = sns.lineplot(x=df_tvl['blockTime'],y=df_tvl['tvl'], color='goldenrod', ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('TVL')
        ax.set_title(df_tvl.loc[:, 'name'].unique()[0])
        ax.tick_params(axis='x', rotation=30)
        # ax.text(x=0.5, y=1.05, s=pool_address, fontsize=0.1, alpha=0.75)
        # breakpoint()
        st.pyplot(fig)
    else:
        st.markdown(
            "We do not have information for this pool")

with row3_2, _lock:
    st.subheader('VaR - BTC / ETH')
    amount_btc = st.slider('BTC', 0.0, 1.0, 0.1)
    amount_eth = st.slider('ETH', 0.0, 1.0, 0.3)

    var_model_input = {
        'window': '20 days',
        'interval':1,
        'confidences':[0,0.01,0.05,1],
        'portfolio':{"positions":
                        [{"amount": amount_btc, "asset": {"symbol": "WBTC"}},
                         {"amount": amount_eth, "asset": {"symbol": "WETH"}}]
                    },
        'price_model': 'chainlink.price-usd'}

    var,_,_ = cmf.finance_var_portfolio_historical(**var_model_input)

    'Value:', var['total_value']

    var_hp,_,_ = cmf.run_historical(
                    model_slug='finance.var-portfolio-historical',
                    model_input = var_model_input,
                    window = '90 days',
                    interval='1 day')

    df_var = CmfRun.to_dataframe(var_hp['result']['series'], fields=[('0.01', lambda x: x['0.01'])])

    if not df_var.empty:
        # pylint:disable=locally-disabled,no-member
        fig = Figure()
        ax = fig.subplots()
        p = sns.lineplot(x=df_var['blockTime'],y=df_var['0.01'], color='goldenrod', ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('VaR')
        ax.set_title('VaR for (ETH,BTC) portfolio')
        ax.tick_params(axis='x', rotation=30)
        # ax.text(x=0.5, y=1.05, s=pool_address, fontsize=0.1, alpha=0.75)
        # breakpoint()
        st.pyplot(fig)
    else:
        st.markdown(
            "We do not have information for this portfolio")

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (.1, 1, .1, 1, .1))

sushi_pools=['0x6a091a3406E0073C3CD6340122143009aDac0EDa',
            '0x397ff1542f962076d0bfe58ea045ffa2d347aca0',
            '0xceff51756c56ceffca006cd410b03ffc46dd3a58',
            '0xe12af1218b4e9272e9628d7c7dc6354d137d024e',
            '0xd4e7a6e2d03e4e48dfc27dd3f46df1c176647e38',
            '0x06da0fd433c1a5d7a4faa01111c044910a184553',
            '0x055475920a8c93cffb64d039a8205f7acc7722d3',
            '0xc3d03e4f041fd4cd388c549ee2a29a9e5075882f',
            '0xdB06a76733528761Eda47d356647297bC35a98BD',
            '0x795065dcc9f64b5614c407a6efdc400da6221fb0']

univ2_pools=['0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc',
             '0x21b8065d10f73ee2e260e5b47d3344d3ced7596e',
             '0x9928e4046d7c6513326ccea028cd3e7a91c7590a',
             '0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852',
             '0xe1573b9d29e2183b1af0e743dc2754979a40d237',
             '0xae461ca67b15dc8dc81ce7615e0320da1a9ab8d5',
             '0xccb63225a7b19dcf66717e4d40c9a72b39331d61',
             '0x3041cbd36888becc7bbcbc0045e3b1f144466f5f',
             '0x9fae36a18ef8ac2b43186ade5e2b07403dc742b1',
             '0x61b62c5d56ccd158a38367ef2f539668a06356ab',
             '0xCEfF51756c56CeFFCA006cD410B03FFC46dd3a58',]

univ3_pools=['0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
             '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed',
             '0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168',
             '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
             '0xc63b0708e2f7e69cb8a1df0e1389a98c35a76d52',
             '0x3416cf6c708da44db2624d63ea0aaef7113527c6',
             '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
             '0x97e7d56a0408570ba1a7852de36350f7713906ec',
             '0x7379e81228514a1d2a6cf7559203998e20598346',
             '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35',
             '0x4674abc5796e1334B5075326b39B748bee9EaA34']

with row3_1, _lock:
    st.subheader('VaR - DEX LP')
    pool_address = st.selectbox('Which pool to show DEX VaR?', univ2_pools)

    var_dex = cmf.finance_var_dex_lp(
        window = '280 days',
        interval = 10,
        pool_address = '0xe12af1218b4e9272e9628d7c7dc6354d137d024e',
        confidences = [0.01],
        lower_range = 0.01,
        upper_range = 0.01,
        price_model = 'chainlink.price-usd')

    var_dex

    if not df_tvl.empty:
        # pylint:disable=locally-disabled,no-member
        fig = Figure()
        ax = fig.subplots()
        p = sns.lineplot(x=df_tvl['blockTime'],y=df_tvl['tvl'], color='goldenrod', ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('TVL')
        ax.set_title(df_tvl.loc[:, 'name'].unique()[0])
        ax.tick_params(axis='x', rotation=30)
        # ax.text(x=0.5, y=1.05, s=pool_address, fontsize=0.1, alpha=0.75)
        # breakpoint()
        st.pyplot(fig)
    else:
        st.markdown(
            "We do not have information for this pool")
