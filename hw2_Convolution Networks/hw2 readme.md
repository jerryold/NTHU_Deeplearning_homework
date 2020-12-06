---
tags: Deeplearning homework
---
## hw2_Convolutional Autoencoder homework
* 步驟1:透過colab執行此作業
    * dir路徑為"/content/drive/My Drive/Colab Notebooks/wafer"
    * 將data.npy和label.npy檔存該資料夾後進行前處理動作
* 步驟2:透過pytorch裡的dataloader套件分別儲存data和label的資料
    * original data shape:(1281,26,26,3)
    * original label shape:(1281,1) 
    * sample data shape:(6405,26,26,3)
    * smaple label shape:(6405,1)
* 步驟3:將class定義好之後show出簡單的圖檔和label確認兩者有對到
 ```classes = ['Center(0)', 'Dount(1)', 'Edge-Loc(2)', 'Edge-Ring(3)', 'Loc(4)','Near-full(5)', 'Random(6)', 'Scratch(7)', 'None(8)']```
* 步驟4:簡單的對wafer圖片做三個channel測試,分別秀出boundary,normal,defect狀況
* 步驟5:Convolutional Autoencoder-
    * Conv2D:(inputsize-kernel+2*padding)/stride+1
    * ConvTranspose2d:(inputsize-1)*stride+kernel-2*padding+outpadding


```
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
#Conv2D:(inputsize-kernel+2*padding)/stride+1
#ConvTranspose2d:(inputsize-1)*stride+kernel-2*padding+outpadding
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        # conv layer (depth from 16 -->32), 3x3 kernels
        self.conv2 =  nn.Conv2d(16, 32, 3, stride=2, padding=1)
        # conv layer (depth from 32 -->64), 5x5 kernels
        self.conv3 =  nn.Conv2d(32, 64, 5)
        
        ## decoder layes ##
        # tconv layer (depth from 64 --> 32), 5x5 kernels
        self.t_conv1 =  nn.ConvTranspose2d(64, 32, 5)
        # tconv layer (depth from 32 --> 16), 3x3 kernels
        self.t_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        # tconv layer (depth from 16 --> 3), 3x3 kernels
        self.t_conv3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=2, output_padding=1)


    def forward(self, x):
        
       ## encode ##
        # add hidden layers with relu activation function
        # add first hidden layer
        x = F.relu(self.conv1(x))  
        
        # add second hidden layer
        x = F.relu(self.conv2(x))
        
        x = self.conv3(x)
        
        
         
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x)) 
        x = F.relu(self.t_conv2(x)) 
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv3(x)) 
        return x

# initialize the NN
model = ConvAutoencoder()
print(model)
```
 * 步驟6:將Gaussian noise 將入latent code
```
def add_noise(inputs,i):
     noise = torch.randn_like(inputs)*(i/10)
     return inputs + noise
```
 * 步驟7:使用Pytorch內建的函式庫計算 MSE loss
 ```
# specify loss function
criterion = nn.MSELoss()
# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
 ```
 * 步驟8:將原圖1280張圖和6405張生成圖呈現出來,同時使用空字串去儲存所產生的label 和image
 
 * 步驟9:將剛剛儲存的image和label的生成圖字串分別儲存在gen_data.npy和gen_label.npy
  ```
from numpy import save
save('/content/drive/My Drive/Colab Notebooks/Wafer_Out_data/gen_data.npy',image_sample_log)
save('/content/drive/My Drive/Colab Notebooks/Wafer_Out_data/gen_label.npy',label_sample_log)
 ```
 * 步驟10:分別對剛剛的隨機8個class圖作autoencoder動作,判別原圖和5張生成圖的差別
 ```
 for idx in range(1280):
      if label_log[idx]=='Center(0)':
          
        for y in range(6):
          ax = fig.add_subplot(1, 10, 1, xticks=[], yticks=[])#subplot行數,subplot列數,每張圖顯示
          imshow(image_display2[idx])#生成原圖  
          ax.set_title(classes[labels1[idx]]) 
          ax = fig.add_subplot(1, 10, y+1, xticks=[], yticks=[])#subplot行數,subplot列數,每張圖顯示
          imshow(image_display[idx])#生成五張重塑後的圖
          ax.set_title(classes[labels1[idx]]
 ```
     