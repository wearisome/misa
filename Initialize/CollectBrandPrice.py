# -*- coding: utf-8 -*-

import jsm
import datetime

class CollectBrandPrices:
    '''
    '''

    def __init__(self, codes):
        ''' constructor	'''
        self.brand_codes = codes

    def collect(self, start_date, end_date):
        ''' jsm を用いて株価の収集を行います。
        テストモジュールのため./Initialize/brand_prices.csvを出力します。
        TODO: MonboDBに格納し、利用しやすい形に成型する
        '''
        q = jsm.QuotesCsv()
        for code in self.brand_codes:
            q.save_historical_prices(code + '.csv', code, jsm.DAILY, start_date, end_date)


if __name__ == '__main__':
    f = open('Initialize/brand_codes.csv')
    codes = f.readlines()
    cbp = CollectBrandPrices(codes)
    cbp.collect(datetime.date(2015, 5, 18), datetime.date(2016, 5, 18))
