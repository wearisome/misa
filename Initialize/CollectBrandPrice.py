# -*- coding: utf-8 -*-

import csv
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
        q = jsm.Quotes()
        for code in self.brand_codes:
            print(code[0])
            try:
                data = q.get_historical_prices(code[0], jsm.DAILY, start_date, end_date)
                f = open('data'+str(code[0])+'.csv', 'w')
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(data)
                #print(data)
            except:
                pass
            
            
          

if __name__ == '__main__':
    f = open('brand_codes.csv', 'r')
    codes = csv.reader(f)
    cbp = CollectBrandPrices(codes)
    
    try:
        cbp.collect(datetime.date(2016, 5, 17), datetime.date(2016, 5, 19))
    except:
        pass
