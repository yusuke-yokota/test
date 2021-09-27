#!/usr/bin/python3
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

sns.set()

def make_data(step, duration, freq, std, plot = False):
# ステップ関数の設定 step: ステップ量、duration: イベント期間
    def f(x):
        # y = 2*x + (step/2.)*np.tanh(4./duration*x) 
                
        # 折れ線の設定
        a = 2 # 定常期間の傾き
        if x <= - duration/2.:
            y = a*x - step/2.
        elif x > - duration/2. and x < duration/2.:
            aa = (a*duration + step)/duration # イベント期間の傾き
            y = aa*x
        elif x >= duration/2.:
            y = a*x + step/2.
                                                                                                               
        return y

# 真値
    x = np.arange(-3, 3+duration, 0.01)
    ytrue = np.array([f(i) for i in x])
# 疑似観測量
# freq: 年間観測回数, std: ばらつきの標準偏差（正規分布）
    xx = np.arange(-3, 3+duration, 1./freq)
    yo0 = np.array([f(i) for i in xx]) #
    yo = np.array([i + np.random.normal(0, std) for i in yo0]) # 乱数を足す
    df_obs = pd.DataFrame({"X":xx,"Y":yo})
    if plot:# グラフを描画する
      plt.ylim([-5,10])
      plt.plot(df_obs.X, df_obs.Y, "o") 
      plt.plot(x, ytrue, "-", label='true')
      plt.ylim([-10, + 10])
      plt.xlabel("Year")
      plt.ylabel("Displacement")
      #plt.show()
      plt.savefig('test0.pdf')
    return (df_obs, (x, ytrue))  # 作成したデータと真値をreturn

step = setSTp # ステップの変位量
duration = setDUr #イベント期間
freq = setFRq # 年間観測回数
std = setPRe1 # 観測値のばらつき（標準偏差）
df, true = make_data(step, duration, freq, std, plot = True)

def regression(df, start_y, width, plot = False):
    
# イベント期間中のデータを抜く(ここではイベント期間中は回帰に使用しないため) -----------------------------------------
    df1 = df[df.X < start_y] # イベント前
    df3 = df[df.X > start_y + width] # イベント後
    df_obs = pd.concat([df1,df3])

# セグメント分割を特徴量としたデータセットを作成 ---------------------------------------------------------------------
    bins = np.array([start_y, start_y + width]) # セグメントを設定
    w_bin = np.digitize(df_obs.X, bins) # 元のデータがどちらのセグメントに属するか、をきめるリスト
    which_bin = [[i] for i in w_bin] # 回帰用の形式にする
                                   
# transform using the OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
    encoder.fit(which_bin)
# transform creates the one-hot encoding
    X_binned = encoder.transform(which_bin)
    #print(X_binned[:5])

    X = np.array([[i] for i in df_obs.X]) # 回帰用の形式にする
    X_combined = np.hstack([X, X_binned]) 
# Xにセグメント分割を示す3つの特徴量を加えた4つの特徴量で線形回帰する
# 実際には、イベント期間は抜いているので特徴量は3つ

# 線形回帰 -----------------------------------------------------------------------------------------------------------
    reg = LinearRegression().fit(X_combined, df_obs.Y) # 折れ線
    resi = reg.predict(X_combined) - df_obs.Y
    var = np.var(resi)
    n = len(df.Y) # データ数
    k = 3 # モデルパラメータ
    cAIC = n*np.log(2*np.pi) + n*np.log(var) + 2*n*k/(n-k-1)
               
    a = (reg.predict([[0, 1, 0]])) # 最初のセグメントの直線の0での値
    b = (reg.predict([[0, 0, 1]])) # 最後のセグメントの直線の0での値
    disp = (b - a)[0]
                                
    reg_lin = LinearRegression().fit(X, df_obs.Y)  # 直線
    resi = reg_lin.predict(X) - df_obs.Y
    var = np.var(resi)
    n = len(df.Y) # データ数
    k = 2 # モデルパラメータ
    cAIC_lin = n*np.log(2*np.pi) + n*np.log(var) + 2*n*k/(n-k-1)
                         
    delta_cAIC = cAIC_lin - cAIC
                                   
    if plot: # グラフ表示
        lt = np.linspace(-3, 3+duration, 1000, endpoint=False)
        lt1 = lt[lt < start_y]
        lt2 = lt[lt > start_y + width]
        lt_new = np.hstack((lt1,lt2)) # イベント期間中を抜く
        line = lt_new.reshape(-1, 1)
        line_binned = encoder.transform(np.digitize(line, bins=bins))
        line_combined = np.hstack([line, line_binned])

        sns.set()

        plt.plot(line, reg.predict(line_combined), label='linear regression') # 回帰直線
        plt.plot(true[0], true[1], "-", label='true') # 真値のtanh関数

        for bin in bins:
            plt.plot([bin, bin], [-15, 15], ':', c='k', linewidth=1) # セグメントの区切り線

        plt.legend(loc="best")
        plt.xlabel("Year")
        plt.ylabel("Displacement")
        plt.text(-3, 9, "delta c-AIC = {:.2f}".format(delta_cAIC))
        plt.text(-3, 6, "step = {:.2f}".format(disp))
        plt.plot(df.X, df.Y, 'o', c='k') # 元データのプロット（イベント期間中は回帰に使用していないことに注意）
                                                                           
    return delta_cAIC, disp

# 例示------------------------------------------------------------------------------
# データ作成
step = setSTp # ステップの変位量
sse_width = setDUr #イベント期間
freq = setFRq # 年間観測回数
std = setPRe # 観測値のばらつき（標準偏差）
df_ex, true = make_data(step, sse_width, freq, std, plot = False)

# 回帰
start_y = -0.5 # スタート年
f_width = 1.0 # fit関数のイベント期間
#delta_cAIC, disp = regression(df_ex, start_y, f_width, plot = True)

# cAICでベストのスタート年の結果を返す関数（関数のSSE期間固定）
def search_start(df, width):
    res = []
    for i in np.arange(-2.8, 3.0 - width - 0.2, 0.1): # SSEのスタート年のサーチ
        i = np.round(i, decimals=1)
        delta_cAIC, disp = regression(df, i, width) 
        res.append([delta_cAIC, disp, i, width])
        #print delta_cAIC, disp, i

    res = np.array(res)
    best = res[np.argmax(res[:,0])] 
                                                     
    return best

# cAICでベストのスタート年の結果を返す関数（関数のSSE期間も推定）
def search_start_end(df):
    res = []
    for j in np.arange(0.1, 3.0, 0.1): # 期間のサーチ
        for i in np.arange(-2.8, 3.0 - j - 0.2, 0.1): # スタート年のサーチ
            delta_cAIC, disp = regression(df, i, j) 
            res.append([delta_cAIC, disp, i, j])
        #print delta_cAIC, disp, i, j

    res = np.array(res)
    best = res[np.argmax(res[:,0])] 
                                                                
    return best

# テストデータのSSE期間を変化させて検定実行（関数のSSE期間固定）
def test(N, step, freq, std, width):
    print("width: {}".format(width))
    result = []
    #for dur in (1.0, 2.0): # SSE期間の設定
    dur = 1.0 # SSE期間の設定
    print("SSE duration: {}".format(dur))
    for i in range(N): # ループ回数
        df, true = make_data(step, dur, freq, std) # データをランダムに生成
        best = search_start(df, width) # 回帰分析
        #print("GGGGGGGGG")
        if best[0] > 0: # delta c-AICが正のもののみ（負のときは直線の方が良い）
            best0 = np.hstack([best, dur])
            result.append(best0)
                                                                                               
    df_res = pd.DataFrame(result)
    df_res.columns = (["delta c-AIC", "step", "start year", "duration", "transient duration"])
                                                                                                         
    return df_res
                                                                                                                  
# テストデータのSSE期間を変化させて検定実行（関数のSSE期間も推定）
def test_dur(N, step, freq, std):
    result = []
    dur=1.0
    #for dur in (1.0): # SSE期間の設定
    #for dur in (0.0001, 1.0, 2.0): # SSE期間の設定
    print("SSE duration: {}".format(dur))
    for i in range(N): # ループ回数
      df, true = make_data(step, dur, freq, std) # データをランダムに生成
      best = search_start_end(df) # 回帰分析
      if best[0] > 0: # delta c-AICが正のもののみ（負のときは直線の方が良い）
        best0 = np.hstack([best, dur])
        result.append(best0)

    df_res = pd.DataFrame(result)
    df_res.columns = (["delta c-AIC", "step", "start year", "duration", "transient duration"])

    return df_res

# テストデータのステップ量を変化させて検定実行（データと関数のSSE期間は固定）
def test_step(N, freq, std, width, sse_dur):
    print("width: {}".format(width))
    result = []
    for step in range(1, 7, 1): # ステップ量
        print("SSE step: {}".format(step))
        for i in range(N): # ループ回数
            df, true = make_data(step, sse_dur, freq, std) # データをランダムに生成
            best = search_start(df, width) # 回帰分析
            if best[0] > 0: # delta c-AICが正のもののみ（負のときは直線の方が良い）
                best0 = np.hstack([best, step])
                result.append(best0)

    df_res = pd.DataFrame(result)
    df_res.columns = (["delta c-AIC", "step", "start year", "duration", "true step"])

    return df_res
#%%time

# 検定実行(期間固定)
N = 1000 # 試行回数
step = setSTp # ステップの変位量
freq = setFRq # 年間観測回数
std = setSTd # 観測値のばらつき（標準偏差）

# fit関数のイベント期間：1年
res_1y = test(N, step, freq, std, 1.)
res_1y["width"] = "1-y fix"

# fit関数のイベント期間：2年
#res_2y = test(N, step, freq, std,  2.)
#res_2y["width"] = "2-y fix"

# fit関数のイベント期間：3年
#res_3y = test(N, step, freq, std, 3.)
#res_3y["width"] = "3-y fix"

# fit関数のイベント期間：0年
#res_0y = test(N, step, freq, std, 0.)
#res_0y["width"] = "0-y fix"

# -----------------------------------------------------
# 検定実行(期間推定)
#res_ey = test_dur(N, step, freq, std)
#res_ey["width"] = "estimate"

## 全データを結合
df_dur = pd.concat([res_1y])

## 推定期間と開始年から終了年の列を作成
df_dur["end year"] = df_dur["start year"] + df_dur["duration"]

## O-Cの計算
df_dur["step (O-C)"] = df_dur["step"] - step
df_dur["start year (O-C)"] = df_dur["start year"] + df_dur["duration"]/2.0 + df_dur["transient duration"]/2.0
df_dur["end year (O-C)"] = df_dur["end year"] - df_dur["transient duration"]/2.0
df_dur["duration (O-C)"] = df_dur["duration"] - df_dur["transient duration"]

print("duration printing")
# データの保存
df_dur.to_csv("duration_test.dat")
