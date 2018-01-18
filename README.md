# captcha-ocr
基于支持向量机的验证码训练、识别
## 用法
下载代码后，
```
from captcha import captcha
```
## 参数说明
```
class captchas:
	def __init__(self,id = '',svm_path = 'svm.xml',temp_path = 'temp',log_path = 'log.txt',max_length = 19,alpha = 0.15,parts = 6,size = 16)
```
初始化：id为图片标识，svm_path为训练样本地址，temp_path为缓存的位置，log_path为log写入的文件，max_length为验证码中每个符号最大像素宽度，alpha为二值化参数，parts为验证码中符号个数，size为映射大小

```
class captchas:
	def train(self,save_name = "svm.xml",path = 'train')
```	
训练：save_name为xml文件保存地址，path为训练集位置，训练集中应以识别结果为文件夹名，里面存放标准化的图片文件。成功返回1.

```
class captchas:
	def ocr(self,pic_path)
```	
识别：pic_path为待识别图片位置。成功返回识别结果。
	
## 图片处理算法
![Original](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/captcha_5a5e2e2d11489.jpg)
### 利用平均灰度判断是否需要反色
半数图片底色为深色，二值化后投影的向量与正常图片相反，所以需要检测进行反色处理。
### 自动选阈值进行二值化
二分查找合适的阈值，使得高于此灰度的像素点和低于此灰度的像素点等于某个参数。
![Bi](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/captcha_5a5e2e2d11489_1.jpg)
### 去噪点、去除或切断干扰线
实验后发现较为优秀的方法，首先重复去除周围至少有7个白点的黑色像素点，直到没有可以去除的，消除噪点；然后去除上下或左右均为白的的黑色像素点，消除横线或竖线；最后重复去除周围至少有7个白点的黑色像素点，直到没有可以去除的，消除线段消除后的剩余噪点。
![Clean](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/captcha_5a5e2e2d11489_2.jpg
### 分割图片
将灰度投影到y坐标轴上，寻找最大的点，两边搜索连续较大点，直至等于最大长度，保存左右坐标内的图片；然后继续寻找，直至达到字符数目。
实验中这个办法远远好于寻找某个下降比例作为左右节点。
![Cut](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/captcha_5a5e2e2d11489_3.jpg

### 遍历每个分割，寻找最大连通
利用dfs染色，寻找最大连通子图，并删除其他子图，这样可以清除大块的干扰线。
### 旋转图像
旋转[-15,15]，寻找宽度最小的位置进行旋转图像。实验中由于测试集验证码基本没有旋转，去除了这一部分。
### 切除空白
分别往x轴、y轴投影，切除值两边为零的连续部分。
### 标准化图像
将图像缩放到某个比例的正方形，便于训练和识别。
![Normalized 1](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/1_captcha_5a5e2e2d11489.jpg
![Normalized 2](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/0_captcha_5a5e2e2d11489.jpg
![Normalized 3](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/2_captcha_5a5e2e2d11489.jpg
![Normalized 4](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/5_captcha_5a5e2e2d11489.jpg
![Normalized 5](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/4_captcha_5a5e2e2d11489.jpg
![Normalized 6](https://github.com/FlouriteJ/captcha-ocr/raw/master/md/3_captcha_5a5e2e2d11489.jpg
## 训练
手动标注了约1000张分割、标准化的图片，肉眼可识别的约为900张，对svm进行训练。
## 测试
在测试中，6位字符验证码全部成功识别比例为35%，根据手动标注样本得到的成功分割率90%来计算，约有52%的6位字符验证码被完全分割成功，这说明对于每个成功分割的字符，训练后的svm识别率大致为94%。