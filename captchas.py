import re
import cv2
import os
import numpy as np
import sys
import time
import random
class Log:
	def __init__(self,path):
		self.path = path
	def write(self,line):
		content = "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "] "+ line
		if self.path !=  '':
			f = open(self.path,'a')
			f.write(content)
			f.write('\n')
			f.close()
		print(content)
	def __exit__(sef):
		pass
		
class captchas:
	def __init__(self,id = '',svm_path = 'svm.xml',temp_path = 'temp',log_path = 'log.txt',max_length = 19,alpha = 0.15,parts = 6,size = 16):
		self.log = Log(log_path)
		self.temp_path = temp_path
		if not os.path.exists(self.temp_path):
			os.mkdir(self.temp_path)
			self.log.write('__init__: Makdir ' + temp_path)
		if svm_path != '':
			self.svm = cv2.ml.SVM_load(svm_path)
			self.log.write('__init__: SVM loaded ' + svm_path)
		if id != '':
			self.id = id
			self.log.write('__init__: Id loaded ' + self.id)
		else:
			self.id = str(int(random.random()*1e16))
			self.log.write('__init__: id Random ' + self.id)
		self.length = max_length
		self.log.write('__init__: Length loaded ' + str(self.length))
		self.alpha = alpha
		self.log.write('__init__: Alpha loaded ' + str(self.alpha))
		self.parts = parts
		self.log.write('__init__: Parts loaded ' + str(self.parts))
		self.size = size
		self.log.write('__init__: Size loaded ' + str(self.size))
		
	def train(self,save_name = "svm.xml",path = 'train'):
		svm = cv2.ml.SVM_create()
		svm.setType(cv2.ml.SVM_C_SVC)
		svm.setKernel(cv2.ml.SVM_LINEAR)
		# svm.setDegree(0.0)
		# svm.setGamma(0.0)
		# svm.setCoef0(0.0)
		# svm.setC(0)
		# svm.setNu(0.0)
		# svm.setP(0.0)
		# svm.setClassWeights(None)
		svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
		set = []
		lables = []

		ws = os.walk(path)
		for root,dirs,files in ws:
			for f in files:
				self.log.write('Train Loaded:'+ os.path.join(root,f))
				img = cv2.imread(os.path.join(root,f))
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				set.append(np.array(gray).reshape(self.size*self.size,1))
				lables.append(ord(root[-1]))
		
		traindata = np.array(set).astype('float32')
		print(lables)
		print(traindata)
		self.log.write('Training')
		svm.train(traindata, cv2.ml.ROW_SAMPLE,np.array(lables).astype('int32'))
		svm.save(save_name)
		self.log.write('Finished')
		return 1
		
	def _dfs_(self,i,j,cnt,gray,x,y):
		gray[i][j] = cnt
		if j+1<y and gray[i][j+1] == 0:
			self._dfs_(i,j+1,cnt,gray,x,y)
		if j-1>=0 and gray[i][j-1] == 0:
			self._dfs_(i,j-1,cnt,gray,x,y)
		if i+1<x and gray[i+1][j] == 0:
			self._dfs_(i+1,j,cnt,gray,x,y)
		if i-1>=0 and gray[i-1][j] == 0:
			self._dfs_(i-1,j,cnt,gray,x,y)	
			
	def ocr(self,pic_path):
		self.log.write('ocr: pic_path')
		basename = os.path.basename(pic_path[:-4])
		# 读取图片
		img = cv2.imread(pic_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		x,y = gray.shape[:2]
		
		# 利用平均灰度判断是否需要反色
		ave = 0
		for i in range(x):
			for j in range(y):
				ave += gray[i][j]
		ave/=(x*y)
		if ave<=100:
			for i in range(x):
				for j in range(y):
					gray[i][j] = 255 - gray[i][j]
					
		# 自动选阈值进行二值化
		left = 0
		right = 255
		black = 0
		alpha = self.alpha
		while left <= right:
			threshold = int((right + left)/2)
			black = 0
			for i in range(x):
				for j in range(y):
					if gray[i][j] <= threshold:
						black+=1
			if black < int(x*y*alpha):
				left = threshold + 1
			elif black > int(x*y*alpha):
				right = threshold - 1
			else:
				break
			
		for i in range(x):
			for j in range(y):
				if gray[i][j] >= threshold:
					gray[i][j] = 255
				else:
					gray[i][j] = 0
					
		# 去噪点、去除或切断干扰线
		beta = 7
		token = True
		while token:
			token = False
			for i in range(1,x-1):
				for j in range(1,y-1):
					if int(gray[i][j]) == 0 and int(int(gray[i-1][j]) + int(gray[i][j-1]) + int(gray[i+1][j]) + int(gray[i][j+1]) +int(gray[i-1][j-1]) + int(gray[i+1][j+1]) + int(gray[i-1][j+1]) + int(gray[i+1][j-1]))/255 >= beta:
						token = True
						gray[i][j] = 255
					
		for i in range(1,x-1):
			for j in range(1,y-1):
				if int(gray[i][j]) == 0 and ( int(int(gray[i-1][j]) + int(gray[i+1][j])) == 255*2 or (int(gray[i][j+1]) + int(gray[i][j-1])) == 255*2 ):
					gray[i][j] = 255
					
		token = True
		while token:
			token = False
			for i in range(1,x-1):
				for j in range(1,y-1):
					if int(gray[i][j]) == 0 and int(int(gray[i-1][j]) + int(gray[i][j-1]) + int(gray[i+1][j]) + int(gray[i][j+1]) +int(gray[i-1][j-1]) + int(gray[i+1][j+1]) + int(gray[i-1][j+1]) + int(gray[i+1][j-1]))/255 >= beta:
						token = True
						gray[i][j] = 255
						
		# 分割图片
		length = self.length
		parts = self.parts
		result = {}
		v = [0]*y
		for i in range(x):
			for j in range(y):
				if gray[i][j] == 0:
					v[j]+=1
		for num in range(parts):
			max = 0
			pos = 0
			for t in range(y):
				if v[t]>max:
					max = v[t]
					pos = t
			left = pos-1
			right = pos+1
			cur = length
			while cur:
				cur-=1
				if left>=0 and right<y:
					if v[left]>v[right]:
						left-=1
					else:
						right+=1
				elif left<0 and right<y:
					right+=1
				elif left>0 and right>=y:
					left-=1
				else:
					break
					
			for t in range(left,right):
				v[t] = 0
			cv2.imwrite(os.path.join(self.temp_path,basename)+ '_' +str(left) +'.tif',gray[:,left:right])
			result[left] = 0
		
		# 遍历每个分割
		for key in result:
			key_path = os.path.join(self.temp_path,basename)+ '_' +str(key) +'.tif'
			img = cv2.imread(key_path)
			os.remove(os.path.join(os.curdir,key_path))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			x,y = gray.shape[:2]
			# 寻找最大连通
			cnt = 0
			flag = True
			while flag:
				flag = False
				cnt+=1
				for i in range(x):
					if flag == False:
						for j in range(y):
							if gray[i][j] == 0:
								flag = True
								self._dfs_(i,j,cnt,gray,x,y)
								break
			v = [0]*(cnt+1)
			for i in range(x):
				for j in range(y):
					if gray[i][j] != 255:
						v[gray[i][j]] +=1
			max = 0
			cat = 0
			for t in range(1,cnt+1):
				if v[t]>max:
					max = v[t]
					cat = t
					
			# 只保留最大连通
			for i in range(x):
				for j in range(y):
					if gray[i][j] == cat:
						gray[i][j] = 0
					else:
						gray[i][j] = 255
			v = [0]*x
			for i in range(x):
				for j in range(y):
					if gray[i][j] == 0:
						v[i]+=1
						
			# 旋转图像
			# for i in range(x):
				# for j in range(y):
					# gray[i][j] = 255 - gray[i][j]

			# rotated = gray
			# # select = 0
			# # min = y
			# scale = 1.0
			# center = ((x+1) / 2, (y+1) / 2)
			# for angle in [0,1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12,13,-13,14,-14,15,-15]:
				# M = cv2.getRotationMatrix2D(center,angle,scale)
				# rotated = cv2.warpAffine(gray, M, (y, x))
				# left = 0
				# right = y - 1
				# flag = False
				# for j in range(y):
					# if flag:
						# for i in range(x):
							# if rotated[i][j] != 0:
								# left = j
				# for j in range(y-1,0,-1):
					# if flag:
						# for i in range(x):
							# if rotated[i][j] != 0:
								# right = j
				# if right - left < min:
					# min = right - left
					# select = angle
			# print(select)
			# M = cv2.getRotationMatrix2D(center,select,scale)
			# rotated = cv2.warpAffine(gray, M, (y, x))
			# for i in range(x):
				# for j in range(y):
					# rotated[i][j] = 255 - rotated[i][j]						
			
			# 切除空白			
			max = 0
			pos = 0
			for t in range(x):
				if v[t]>max:
					max = v[t]
					pos = t
			left = pos
			right = pos
			while left-1>=0 and v[left-1] != 0:
				left-=1
			while right+1<x and v[right+1] != 0:
				right+=1
			gray = gray[left:right+1,:]

			x,y = gray.shape[:2]
			v = [0]*y
			for i in range(x):
				for j in range(y):
					if gray[i][j] == 0:
						v[j]+=1	
			max = 0
			pos = 0
			for t in range(y):
				if v[t]>max:
					max = v[t]
					pos = t
			left = pos
			right = pos
			while left-1>=0 and v[left-1] != 0:
				left-=1
			while right+1<y and v[right+1] != 0:
				right+=1
			gray = gray[:,left:right+1]
			x,y = gray.shape[:2]
			
			# 标准化图像
			resized = cv2.resize(gray, (self.size,self.size), interpolation=cv2.INTER_AREA)

			
			# 利用训练模型进行识别
			pred = []
			pred.append(np.array(resized).reshape(self.size*self.size,1))
			response = self.svm.predict(np.array(pred).astype('float32'))
			result[key] = chr(response[1][0][0])
			
		# 按顺序整理结果
		final = ''
		keys = list(result)
		keys = sorted(keys)
		for key in keys:
			final += result[key]
		self.log.write('ocr result: ' + final)	
		return final
if __name__ == "__main__":
	Captchas = captchas('test')
	Captchas.train("test.xml")
	Captchas.ocr('captcha_5a5f51b7db2b6.jpg')