# DSAI_HW2   Stock Forcasting

本作業所使用的 ``time series`` 模型為 **LSTM** ，並以前五天的資料來當作每次預測的依據，去預測接下來後一天的資料。

# Environment
  - **Python 3.8.3**
  - **Ubuntu 20.04.2 LTS**

# Requirement
requirements.txt目前還是手刻，若有python版本和lib版本相衝或不相容，還請自行解決。

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

執行 app.py。 Input 和 Output path 已經定義在app.py里了.
```
python3 app.py
```
強烈建議直接執行ipynb檔案來直接看我們在各個區塊的輸出結果。

## Input data
Input的data為來自NASDAQ:IBM。在這份資料中有共有**1476**個 ``entries`` 和 **4** 個 ``features``。而本模型只使用的features為 ``close``，而``testing``則為**20**個``entries``。
  
![image](https://user-images.githubusercontent.com/41318666/114421341-cc2a3180-9be7-11eb-8751-4a2b555898fc.png)
![image](https://user-images.githubusercontent.com/41318666/114421405-d9dfb700-9be7-11eb-85f3-29caae92e537.png)


## Scaling
為了加快模型收斂找到最佳參數組合，這裡使用``MinMaxScaler``把資料重新scaling成 **-1** 至 **1** 之間。

## Model Structure

![image](https://user-images.githubusercontent.com/41318666/114420770-31c9ee00-9be7-11eb-9a19-2624cfc02f3d.png)


## Training
``epochs``設定為**50**，最後``loss``約位於**0.04**左右：

![image](https://user-images.githubusercontent.com/41318666/114421875-56729580-9be8-11eb-8e69-d1651f86534a.png)

![image](https://user-images.githubusercontent.com/41318666/114421995-75712780-9be8-11eb-8de1-be08e47d1227.png)

 

## Prediction Result

以此模型進行往後20天的股票之預測結果。

![image](https://user-images.githubusercontent.com/41318666/114422284-b5380f00-9be8-11eb-9476-c6e4a226c5ef.png)


## Keywords
  - **Stock**
  - **Forecasting**
  - **LSTM**
  - **RNN**
  - **Multivariables**
