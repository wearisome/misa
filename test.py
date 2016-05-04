# -*- coding: utf-8 -*-

import GeneticAlgorithm as ga
import numpy as np

# テストモジュール
if __name__ == "__main__":
    size = 5 # 遺伝子の個数
    np.random.seed(0)
    data = np.random.rand(size)*5 # 入力データの生成
    teach = 5 # 教師データ
    popSize = 20 # 遺伝アルゴリズムの個体数
    rga = ga.GeneticAlgorithm(popSize, data, teach)
    rga.setName("RealCodeGeneticAlgorithm")
    print(rga.getName())
    # 実行結果を20回確認する
    count = 0
    while count < 20:
        # 1000回の世代交代と親からは6個体の子を作成する
        # 計算から得られた関数の表現行列(ベクトル形式)を取得する
        v_func = rga.execute(500, 6)
        # 取得した表現行列から結果を計算
        result = rga.funcResult(v_func, data)
        count += 1
        print("teach Data = "+str(teach)+", result = "+str(result)+
              ", error = "+str(abs(teach-result)))
    np.random.seed(9)
    ndata = np.random.rand(size)*5
    result = rga.funcResult(v_func, ndata)
    print("teach Data = "+str(teach)+", result = "+str(result))

"""
    rga.createPopulation()
    parent = rga.selectTwoIndividual()
    rga.printIndividual(parent)

    print(rga.getName())
    count = 0
    children = []
    while count < 6:
        children.append( rga.crossover(parent) )
        count += 1
    rga.printIndividual(children)

    print(rga.getName())
    select = rga.selection(children, parent)
"""
"""
    v_func = rga.execute(1000, 6)
    ndata = np.random.rand(size)*5
    result = rga.funcResult(v_func, ndata)
    print("teach Data = "+str(teach)+", result = "+str(result))
    result = rga.funcResult(v_func, data)
    print("teach Data = "+str(teach)+", result = "+str(result))
"""
        
