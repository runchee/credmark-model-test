# pylint:disable=locally-disabled,missing-function-docstring,missing-class-docstring,missing-module-docstring,dangerous-default-value

from typing import List, Optional, Callable, Tuple
from datetime import datetime
import json
import requests
import streamlit as st
import pandas as pd

class CmfRun:
    def __init__(self, gateway, block_number='latest', chain_id = 1):
        self.gateway = gateway
        self.block_number = block_number
        self.chain_id = chain_id

    def model_input(self,model_slug,model_input, block_number: Optional[int]= None):
        model_input = {
            "slug": model_slug,
            "chainId": self.chain_id,
            "blockNumber": self.block_number if block_number is None else block_number,
            "input": model_input
        }
        return model_input

    def request(self, model_input):
        headers = { 'Content-Type': 'application/json' }
        response = requests.post(self.gateway, data=json.dumps(model_input), headers=headers)
        if response.status_code == 201:
            res = response.json()
            try:
                output = res['output']
            except Exception as err:
                print(err)
                raise
            return output, res['chainId'], res['blockNumber']

    def run_model(self, model_slug:str, model_input:dict, block_number: Optional[int]= None):
        model_input = self.model_input(model_slug, model_input, block_number)
        return self.request(model_input)

    @staticmethod
    def to_list(series_in, fields: Optional[List[Callable]] = None) -> List[List]:
        """
        Parameters:
            fields (List[Callable] | None): List of lambda to extract certain field from output.
                Leave empty to extract the entire output.
        Extract tuples from series data
        """
        if fields is None:
            return [[p['blockNumber'],
                    datetime.utcfromtimestamp(p['blockTimestamp']),
                    p['output']]
                    for p in series_in]
        else:
            return [([p['blockNumber'],
                    datetime.utcfromtimestamp(p['blockTimestamp'])] +
                    [f(p['output']) for f in fields])
                    for p in series_in]

    @staticmethod
    def to_dataframe(series_in, fields: Optional[List[Tuple[str, Callable]]] = None) -> pd.DataFrame:
        """
        Parameters:
            fields (List[Tuple[str, Callable]] | None): List of field name and lambda to extract
                certain field from output. Leave empty to extract the entire output.
        Extract tuples from series data

        """
        series_in_list = CmfRun.to_list(series_in, fields=None if fields is None else [f for (_, f) in fields])
        if fields is None:
            return pd.DataFrame(series_in_list, columns=['blockNumber', 'blockTime', 'output'])
        else:
            return pd.DataFrame(series_in_list,
                                columns=['blockNumber', 'blockTime'] + [c for c, _ in fields])

    def run_historical(self, model_slug:str, model_input:dict, window:str, interval:str):
        model_input = self.model_input(
                        "historical.run-model",
                        {
                            "model_slug": model_slug,
                            "model_input": model_input,
                            "window": window,
                            "interval": interval
                        })
        return self.request(model_input)

    def token_price(self,token_address='0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9'):
        return self.run_model("token.price", {"address": token_address })

    def token_info(self,token_address='0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9'):
        return self.run_model("token.info", {"address": token_address })

    def curve_pool_info(self,pool_address='0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7'):
        return self.run_model("curve-fi.pool-info", {"address": pool_address, })

    def finance_var_portfolio_historical(self,
                                        window:str = '20 days',
                                        interval:int = 1,
                                        confidences: List[int] = [0,0.01,0.05,1],
                                        portfolio:dict = {"positions":[{"amount": "0.5", "asset": {"symbol": "WBTC"}},
                                                                       {"amount": "0.5", "asset": {"symbol": "WETH"}}]},
                                        price_model:str = 'chainlink.price-usd'
                                        ):
        return self.run_model("finance.var-portfolio-historical",
                    {"window": window,
                    "interval": interval,
                    "confidences": confidences,
                    "portfolio": portfolio,
                    "price_model": price_model})

    def finance_var_dex_lp(self,
                          window:str = '280 days',
                          interval:int = 10,
                          pool_address: str = '0xe12af1218b4e9272e9628d7c7dc6354d137d024e',
                          confidences: List[float] = [0.01],
                          lower_range: float = 0.01,
                          upper_range: float = 0.01,
                          price_model:str = 'chainlink.price-usd'
                          ):
        return self.run_model(
                    "finance.var-dex-lp",
                    {
                        "pool": {"address":pool_address},
                        "window":window,
                        "interval":interval,
                        "confidences": confidences,
                        "lower_range": lower_range,
                        "upper_range": upper_range,
                        "price_model": price_model
                    })
