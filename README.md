# DSAI_HW2   Stock Forcasting

本作業所使用的 ``time series`` 模型為 **LSTM** ，並以前五天的資料來當作每次預測的依據，去預測接下來後一天的資料。

# Environment
  - **Python 3.8.3**
  - **Ubuntu 20.04.2 LTS**

# Requirement

  - **pandas == 1.2.3**
  - **keras == 2.4.3**
  - **matplotlib == 3.2.2**
  - **numpy == 1.19.5**
  - **sklearn == 0.24.1**
  - **pydot == 1.4.2**
  - **graphviz == 0.16**

# Build
Install requirement.txt
```
pip3 install -r requirements.txt
```

執行 trader.py。 Input 和 Output path 已經有一份default定義在trader.py里了.
```
python3 trader.py dataset/training.csv dataset/testing.csv output.csv
```
建議直接執行ipynb檔案來直接看我們在各個區塊的輸出結果。

## Input data
Input的data為來自NASDAQ:IBM。在這份資料中有共有**1476**個 ``entries`` 和 **4** 個 ``features``。而本模型只使用的features為 ``close``，而**testing data**則為**20**個``entries``。
  
![image](https://user-images.githubusercontent.com/41318666/114421341-cc2a3180-9be7-11eb-8751-4a2b555898fc.png)
![image](https://user-images.githubusercontent.com/41318666/114421405-d9dfb700-9be7-11eb-85f3-29caae92e537.png)


## Scaling
為了加快模型收斂找到最佳參數組合，這裡使用``MinMaxScaler``把資料重新scaling成 **-1** 至 **1** 之間。

## Model Structure

![image](https://user-images.githubusercontent.com/41318666/114420770-31c9ee00-9be7-11eb-9a19-2624cfc02f3d.png)


## Training
``epochs``設定為**100**，最後``loss``約位於**0.0014**左右：

![image](https://user-images.githubusercontent.com/41318666/114421875-56729580-9be8-11eb-8e69-d1651f86534a.png)

![image](https://user-images.githubusercontent.com/41318666/114421995-75712780-9be8-11eb-8de1-be08e47d1227.png)

 

## Prediction Result

以此模型進行往後20天的股票之預測結果。

![image](https://user-images.githubusercontent.com/41318666/114422284-b5380f00-9be8-11eb-9476-c6e4a226c5ef.png)

## Note

由於題目規定允許買空與賣空之動作，基於準確的預測結果，在決定買賣動作時，我們會觀察兩天後與三天後的曲線變化的預測，並分為以下情形：
當目前持有股票時：
  若曲線向下，則賣，反之則不做動作
當目前無股票時：
  若曲線向下，則賣空，反之則買
當目前賣空時：
  若曲線向上，則買，反之則不做動作
  
觀察兩天後而不是一天後的目的在於當股票在持續漲跌時，可以利用買空與賣空動作獲得更大的利潤，由於一天只能買賣一張股票，故提早兩天預測。

預測獲利經由test_calculator計算後可得約為

![image](https://user-images.githubusercontent.com/41318666/114424521-c97d0b80-9bea-11eb-8e26-fccbde3c11ad.png)

 

## Keywords
  - **Stock**
  - **Forecasting**
  - **LSTM**
  - **RNN**
  - **Multivariables**
