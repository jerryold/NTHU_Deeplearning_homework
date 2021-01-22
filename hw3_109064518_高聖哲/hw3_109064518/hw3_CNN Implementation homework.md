---
tags: Deeplearning homework
---
# hw3_CNN Implementation homework
* 步驟1:透過colab執行此作業
    * Train Dateset dir路徑為"/content/drive/My Drive/Colab Notebooks/Fruit/Data_train"
    * Test Dateset dir路徑為"/content/drive/My Drive/Colab Notebooks/Fruit/Data_test"
    * 並對label統一做onehot 的動作
        *    def to_categorical
* 步驟2:處理Data train和test三個檔案,手作標籤並將其合併
    * train image shape:(1470, 4,32 , 32)
    * train  label shape:(1470,3) 
    * test image shape:(1470, 4, 32, 32)
    * test label shape:(498,3)
* 步驟3:透過shuffle_split_data將training data和validation data分割image和label 7:3
    *    def shuffle_split_data
* 步驟4:將其他function寫成其他py檔放在colab底下的資料夾,然後再import進來
    * 所有py檔資料路徑放置:/content/drive/My Drive/Colab Notebooks/Fruit/src裡面
1. layer.py介紹-主要撰寫本次model所使用的function和layer
        *  class Conv
        *  class AvgPool
        *  class Fc
        *  class ReLU
        *  class AdamGD
        *  class Softmax
        *  class CrossEntropy
        
2.model.py介紹
        *  LeNet5-將layer的function丟進forward & backward計算
        * get_params & set_params-取得並設置params
        
3.utils.py介紹-其他額外的function引入
* save_params_to_file-將params儲存至colab資料夾中
```
        需要建立一個資料夾儲存params,路徑為:
        /content/drive/My Drive/Colab Notebooks/Fruit/src/fast/save_weights/
        
        資料檔名為-final_weights.pkl
```
*   load_params_from_file-讀取剛剛儲存params file的資料夾

* def im2col(X, HF, WF, stride, pad):  def col2im(dX_col, X_shape, HF, WF, stride, pad):>>im2col進行優化卷積層運算,col2im是還原過程
        
* 步驟5:撰寫train的function訓練training 以及 validation Accuracy
    * def train()
* 步驟6:引用AdamGD並設置參數計算 MSE loss
```
def AdamGD

optimizer = AdamGD(lr = lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params()) 
```

* 步驟7:撰寫test的function,將剛剛的params進行計算test accuracy
  * def test(isNotebook=False)