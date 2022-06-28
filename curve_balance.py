# pylint:disable=locally-disabled,missing-function-docstring,missing-class-docstring,missing-module-docstring

from datetime import datetime, timedelta, timezone

import altair as alt
# from matplotlib.backends.backend_agg import RendererAgg
# from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from streamlit_lottie import st_lottie
import plotly.graph_objects as go

from cmf import CmfRun

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
                '0x828b154032950C8ff7CF8085D841723Db2696056',
                '0x4e0915C88bC70750D68C481540F081fEFaF22273',
                '0xA5407eAE9Ba41422680e2e00537571bcC53efBfD',]

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

four_pools = [
    {
        'name': '/'.join(pif['output']['tokens_symbol']),
        'ratio':pif['output']['ratio'],
        'a/b': (pif['output']['balances'][0] * pif['output']['token_prices'][0]['price'],
                pif['output']['balances'][1] * pif['output']['token_prices'][1]['price'],
                pif['output']['balances'][2] * pif['output']['token_prices'][2]['price'],
                pif['output']['balances'][3] * pif['output']['token_prices'][3]['price'])
    }
    for pif in curves_info
    if len(pif['output']['tokens_symbol']) == 4
]

def plot_pool_n(basis, n_point, pool_n, bal_ratio_func):
    ''' Plots a legend for the colour scheme
    given by abc_to_rgb. Includes some code adapted
    from http://stackoverflow.com/a/6076050/637562'''

    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_subplot(111,aspect='equal')

    # Plot points
    if pool_n == 3:
        grids = np.mgrid[0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point]
    elif pool_n == 4:
        grids = np.mgrid[0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point]
    else:
        raise ValueError(f'Unsupported pool size {pool_n}')

    abc = np.dstack(tuple(g.flatten() for g in grids))[0][1:]
    abc = abc / abc.sum(axis=1)[:, np.newaxis]

    data = np.dot(abc, basis)
    # colours = [abc_to_rgb(A=point[0],B=point[1],C=point[2]) for point in abc]
    # ax.scatter(data[:,0], data[:,1],marker=',',edgecolors='none',facecolors=colours)
    bal_ratio = bal_ratio_func(abc)
    ax.scatter(data[:,0], data[:,1],
                marker=',',
                edgecolors='none',
                c=bal_ratio, # np.ones(data[:,0].shape) * 256
                cmap=plt.get_cmap('Greens'))

    # Plot triangle
    ax.plot([[basis[_,0] for _ in range(pool_n)] + [0,]],
            [[basis[_,1] for _ in range(pool_n)] + [0,]],
            **{'color':'black','linewidth':3})

    # Plot labels at vertices
    offset = 0.25
    fontsize = 8
    for nn in range(pool_n):
        ax.text(basis[nn,0]*(1+offset),
                basis[nn,1]*(1+offset),
                f'Token {nn+1}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=fontsize)

    ax.set_frame_on(False)
    ax.set_xticks(())
    ax.set_yticks(())

    return fig, ax

def plot_pool_n_data(basis, n_point, pool_n, bal_ratio_func):
    ''' Plots a legend for the colour scheme
    given by abc_to_rgb. Includes some code adapted
    from http://stackoverflow.com/a/6076050/637562'''

    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_subplot(111,aspect='equal')

    # Plot points
    if pool_n == 3:
        grids = np.mgrid[0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point]
    elif pool_n == 4:
        grids = np.mgrid[0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point, 0.0:1.0:n_point]
    else:
        raise ValueError(f'Unsupported pool size {pool_n}')

    abc = np.dstack(tuple(g.flatten() for g in grids))[0][1:]
    abc = abc / abc.sum(axis=1)[:, np.newaxis]

    data = np.dot(abc, basis)
    # colours = [abc_to_rgb(A=point[0],B=point[1],C=point[2]) for point in abc]
    # ax.scatter(data[:,0], data[:,1],marker=',',edgecolors='none',facecolors=colours)
    bal_ratio = bal_ratio_func(abc)

    return data, bal_ratio

st.title('Curve - Pool Balance Ratio')

col1, col2, col3 = st.columns(3)

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
        ax.scatter(pif['a/b'], bal_2pool(pif['a/b']))
        ax.text(pif['a/b'], bal_2pool(pif['a/b']), pif['name'])

    ax.set_frame_on(False)
    ax.set_xlabel('token ratio')
    ax.set_ylabel('balance ratio')
    ax.grid(linestyle = '--', linewidth = 0.5)

    st.pyplot(fig)

rng = npr.default_rng(12345)

with col2:
    col2.title('3-pool')
    def abc_to_rgb(A=0.0,B=0.0,C=0.0):
        ''' Map values A, B, C (all in domain [0,1]) to
        suitable red, green, blue values.'''
        return (min(B+C,1.0),min(A+C,1.0),min(A+B,1.0))

    # Basis vectors for shape
    basis = np.array([[0.0, 1.0], [-1.5/np.sqrt(3), -0.5],[1.5/np.sqrt(3), -0.5]])
    def bal_ratio_3pool(abc):
        return abc[:,0] * abc[:,1] * abc[:,2] / np.power(1/3, 3)

    fig_3pool, ax = plot_pool_n(basis, n_point = 30j, pool_n = 3, bal_ratio_func= bal_ratio_3pool)

    for pif in three_pools:
        three_value = np.array(pif['a/b'])
        three_value = three_value / three_value.sum()
        two_value = np.dot(three_value, basis)
        ax.scatter(two_value[0], two_value[1])
        ax.text(two_value[0], two_value[1], pif['name'])

    st.pyplot(fig_3pool)

with col3:
    col3.title('4-pool')

    basis = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    def bal_ratio_4pool(abcd):
        return abcd[:,0] * abcd[:,1] * abcd[:,2] * abcd[:,3] / np.power(1/4, 4)

    data, bal_ratio = plot_pool_n_data(basis, n_point = 12j, pool_n = 4, bal_ratio_func=bal_ratio_4pool)

    df = pd.DataFrame(data)
    df.columns = pd.Index(['x','y'])
    df.loc[:, 'Ratio'] = bal_ratio

    df.x = round(df.x / 0.02) / 50
    df.y = round(df.y / 0.02) / 50
    df = df.groupby(['x', 'y'], as_index=False)['Ratio'].max()
    # df = df.groupby(['Ratio'], as_index=False).agg({'x': ['mean'], 'y': ['mean']})

    fig = go.Figure(data=[go.Scatter3d(
        x=df.x,
        y=df.y,
        z=df.Ratio,
        mode='markers',
        marker=dict(
            size=12,
            color=df.Ratio,                # set color to an array/list of desired values
            colorscale='Greens',   # choose a colorscale
            opacity=0.8
        )
    )])

    annoctation_dict = [
        dict(showarrow=False,
                x=xy[0],
                y=xy[1],
                z=0,
                text=f'Token {alpha}',
                xanchor="left",
                xshift=10,
                opacity=0.7,
                font=dict(
                color="black",
                size=20))
                for xy,alpha in zip(basis, range(1, 5))
                ]
    for pif in four_pools:
        four_value = np.array(pif['a/b'])
        four_value = four_value / four_value.sum()
        two_value = np.dot(four_value, basis)
        # fig.scatter(two_value[0], two_value[1])
        annoctation_dict.append(
            dict(
                showarrow=False,
                x=two_value[0],
                y=two_value[1],
                z=pif['ratio'],
                text=pif['name'],
                xanchor="left",
                xshift=10,
                opacity=0.7,
                font=dict(
                color="black",
                size=20))
        )

    fig.update_layout(
        uniformtext_minsize=12,
        scene=dict(
            xaxis=dict(type="linear"),
            yaxis=dict(type="linear"),
            zaxis=dict(type="linear"),
            annotations=annoctation_dict))

        # fig.add_annotation(x=two_value[0], y=two_value[1], z=pif['ratio'], text=pif['name'])
        # ax.text(two_value[0], two_value[1], pif['name'])

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True, height=800)
    st.plotly_chart(fig)
    # st.pyplot(fig_4pool)
