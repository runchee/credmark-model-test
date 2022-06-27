# pylint:disable=locally-disabled,missing-function-docstring,missing-class-docstring,missing-module-docstring

from datetime import datetime

import altair as alt
import matplotlib
import numpy as np
import numpy.random as npr
import pandas as pd
import seaborn as sns
import streamlit as st
# from matplotlib.backends.backend_agg import RendererAgg
# from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from cmf import CmfRun
from datetime import datetime, timedelta, timezone
# matplotlib.use("agg")
# _lock = RendererAgg.lock

sns.set_style('darkgrid')


CREDMARK_GATEWAY = 'https://gateway.credmark.com/v1/model/run'
CREDMARK_GATEWAY_LOCAL = 'http://192.168.68.122:8700/v1/model/run'

st.set_page_config(layout="wide")

gateway = st.selectbox('Gateway', [CREDMARK_GATEWAY, CREDMARK_GATEWAY_LOCAL])
cmf = CmfRun(gateway = gateway, block_number=14836288)

d = st.date_input("CoB", datetime.utcnow().date() - timedelta(days=1))
d_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
block_number = cmf.run_model('rpc.get-blocknumber', {'timestamp': d_utc})
block_number = block_number[0]['blockNumber']

curve_pools = ['0x961226b64ad373275130234145b96d100dc0b655',
                '0x8301AE4fc9c624d1D396cbDAa1ed877821D7C511',
                '0x43b4FdFD4Ff969587185cDB6f0BD875c5Fc83f8c',
                '0xd658A338613198204DCa1143Ac3F01A722b5d94A',
                '0xDC24316b9AE028F1497c275EB9192a3Ea0f67022',
                '0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7',
                '0xd632f22692FaC7611d2AA1C0D552930D43CAEd3B',
                '0xCEAF7747579696A2F0bb206a14210e3c9e6fB269',
                '0xD51a44d3FaE010294C616388b506AcdA1bfAAE46',
                '0x5a6A4D54456819380173272A5E8E9B9904BdF41B',
                '0x93054188d876f558f4a66B2EF1d97d16eDf0895B',
                '0x2dded6Da1BF5DBdF597C45fcFaa3194e53EcfeAF',
                '0x9D0464996170c6B9e75eED71c68B99dDEDf279e8',
                '0x828b154032950C8ff7CF8085D841723Db2696056']

curves_info = cmf.run_model('compose.map-inputs',
                            {'modelSlug': 'curve-fi.pool-info',
                            'modelInputs': [{'address':addr} for addr in curve_pools]},
                            block_number=block_number)[0]['results']

two_pools = [
    {
        'name': '/'.join(pif['output']['tokens_symbol']),
        'ratio':pif['output']['ratio'],
        'a/b': (pif['output']['balances'][0] * pif['output']['token_prices'][0]['price'] /
                (pif['output']['balances'][0] * pif['output']['token_prices'][0]['price'] +
                pif['output']['balances'][1] * pif['output']['token_prices'][1]['price']))
    }
    for pif in curves_info
    if len(pif['output']['tokens_symbol']) == 2
]

three_pools = [
    {
        'name': '/'.join(pif['output']['tokens_symbol']),
        'ratio':pif['output']['ratio'],
        'a/b': (pif['output']['balances'][0] * pif['output']['token_prices'][0]['price'],
                pif['output']['balances'][1] * pif['output']['token_prices'][1]['price'],
                pif['output']['balances'][2] * pif['output']['token_prices'][2]['price'])
    }
    for pif in curves_info
    if len(pif['output']['tokens_symbol']) == 3
]


st.title('Curve - Pool Balance Ratio')

col1, col2 = st.columns(2)

def bal_2pool(xs):
    ys = 1 - xs
    return xs * ys / np.power(1/2, 2)

with col1:
    col1.title('2-pool')
    xs = np.linspace(0, 1, 100)
    balance_ratio = bal_2pool(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, balance_ratio)

    for pif in two_pools:
        # print(pif['ratio'], bal_2pool(pif['a/b']), pif['name'], pif['a/b'])
        # print([pif['a/b']], [bal_2pool(pif['a/b'])], [pif['name']])
        ax.scatter(pif['a/b'], bal_2pool(pif['a/b']))
        ax.text(pif['a/b'], bal_2pool(pif['a/b']), pif['name'])

    st.pyplot(fig)

rng = npr.default_rng(12345)

with col2:
    col2.title('3-pool')
    def abc_to_rgb(A=0.0,B=0.0,C=0.0):
        ''' Map values A, B, C (all in domain [0,1]) to
        suitable red, green, blue values.'''
        return (min(B+C,1.0),min(A+C,1.0),min(A+B,1.0))

    def plot_legend():
        ''' Plots a legend for the colour scheme
        given by abc_to_rgb. Includes some code adapted
        from http://stackoverflow.com/a/6076050/637562'''

        # Basis vectors for triangle
        basis = np.array([[0.0, 1.0], [-1.5/np.sqrt(3), -0.5],[1.5/np.sqrt(3), -0.5]])

        fig, ax = plt.subplots(1, 1)
        # ax = fig.add_subplot(111,aspect='equal')

        # Plot points
        n_point = 50j
        a, b, c = np.mgrid[0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point]
        a, b, c = a.flatten(), b.flatten(), c.flatten()

        abc = np.dstack((a,b,c))[0][1:]
        abc = abc / abc.sum(axis=1)[:, np.newaxis]
        # abc = filter(lambda x: x[0]+x[1]+x[2]==1, abc) # remove points outside triangle
        # abc = map(lambda x: x/sum(x), abc) # or just make sure points lie inside triangle ...

        data = np.dot(abc, basis)
        # colours = [abc_to_rgb(A=point[0],B=point[1],C=point[2]) for point in abc]
        # ax.scatter(data[:,0], data[:,1],marker=',',edgecolors='none',facecolors=colours)
        # breakpoint()
        balance_ratio = abc[:,0] * abc[:,1] * abc[:,2] / np.power(1/3, 3)
        ax.scatter(data[:,0], data[:,1],
                   marker=',',
                   edgecolors='none',
                   c=balance_ratio,
                   cmap=plt.get_cmap('Blues'))

        # Plot triangle
        ax.plot([[basis[_,0] for _ in range(3)] + [0,]],[[basis[_,1] for _ in range(3)] + [0,]],**{'color':'black','linewidth':3})

        # ax.text(0, 0, 'Origin')

        # Plot labels at vertices
        offset = 0.25
        fontsize = 32
        ax.text(basis[0,0]*(1+offset), basis[0,1]*(1+offset), 'A', horizontalalignment='center',
                verticalalignment='center', fontsize=fontsize)
        ax.text(basis[1,0]*(1+offset), basis[1,1]*(1+offset), 'B', horizontalalignment='center',
                verticalalignment='center', fontsize=fontsize)
        ax.text(basis[2,0]*(1+offset), basis[2,1]*(1+offset), 'C', horizontalalignment='center',
                verticalalignment='center', fontsize=fontsize)

        for pif in three_pools:
            three_value = np.array(pif['a/b'])
            three_value = three_value / three_value.sum()
            two_value = np.dot(three_value, basis)
            ax.scatter(two_value[0], two_value[1])
            ax.text(two_value[0], two_value[1], pif['name'])


        ax.set_frame_on(False)
        ax.set_xticks(())
        ax.set_yticks(())

        st.pyplot(fig)

    plot_legend()