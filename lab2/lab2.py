# author: blow_away
# 基于内容(特征) 的图像检索

# 对 CIFAR-10 的一个子集 90%为database 10%位query
# 通过颜色直方图 HOG 来检索并返回database中 TOP20 的图像
# 性能评价指标采用mAP 相似性度量 -> 欧式距离
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageFeature (object):
    def __init__(self,database_path,query_path):
        self.database_path = database_path
        self.query_path = query_path
        self.database_images = []
        self.query_images = []
        self.database_images_gray = []
        self.query_images_gray = []
        # cat-0 dog-1
        self.database_label = [0]*90+[1]*90
        self.query_label = [0]*10+[1]*10

        self.bin = 8
    def getImages(self):
        #获得database图像
        for i in range(180):
            #读hsv 不读rgb
            arr1 =  np.array(cv2.cvtColor(cv2.imread(self.database_path+"database ("+str(i+1)+").png"),cv2.COLOR_BGR2HSV))
            dis1 = np.max(arr1) - np.min(arr1)
            arr1 = (arr1 - np.min(arr1))/dis1
            self.database_images.append(arr1)
            
            arr1 = np.array(cv2.imread(self.database_path+"database ("+str(i+1)+").png",cv2.IMREAD_GRAYSCALE))
            dis1 = np.max(arr1) - np.min(arr1)
            arr1 = (arr1 - np.min(arr1))/dis1
            self.database_images_gray.append(arr1)
        #获得query图像
        for i in range(20):
            #读hsv 不读rgb
            arr1 = np.array(cv2.cvtColor(cv2.imread(self.query_path+"query ("+str(i+1)+").png"),cv2.COLOR_BGR2HSV))
            dis1 = np.max(arr1) - np.min(arr1)
            arr1 = (arr1 - np.min(arr1))/dis1
            
            self.query_images.append(arr1)
            
            arr1 = np.array(cv2.imread(self.query_path+"query ("+str(i+1)+").png",cv2.IMREAD_GRAYSCALE))
            dis1 = np.max(arr1) - np.min(arr1)
            arr1 = (arr1 - np.min(arr1))/dis1
            self.query_images_gray.append(arr1)
        return
    
    #直方图的欧式距离
    def dis(self,x,y):
        Item1 = np.array(x)
        Item2 = np.array(y)
        dis = []
        #归一化
        Item1=(Item1 - np.min(x)) / (np.max(x)-np.min(x))
        Item2=(Item2 - np.min(y)) / (np.max(y)-np.min(y))
        #一维数组
        Item1 -= Item2
        for i in np.nditer(Item1):
            dis.append(i**2)
        return np.sqrt(sum(dis))
    
    #颜色直方图
    def colorHistogram(self):
        print("颜色直方图:")
        #query的颜色直方图
        self.query_HSV = []
        Item = [0]*(self.bin+1)
        Index = np.zeros([20,20])
        #获取 query的直方图
        for each in self.query_images:
            Item1 = []
            Item2 = []
            Item3 = []
            for i in range(3):
                Item = [0]*(self.bin+1)
                for c in each[:,:,i]:
                    for x in c:
                        Item[int(x*self.bin)]+=1                        
                if i==0:
                    Item1 = Item
                elif i==1:
                    Item2 = Item
                else:
                    Item3 = Item
            self.query_HSV.append(Item1+Item2+Item3) 
        #end for
        # 对每一个query求180个误差
        # database的颜色直方图
        self.data_HSV = []
        for count in range(20):
            # 误差 (通过emd来求)
            exp = []
            j = -1
            for each in self.database_images:
                j+=1
                Item1 = []
                for i in range(3):
                    Item = [0]*(self.bin+1)
                    for c in each[:,:,i]:
                        for x in c:
                            Item[int(x*self.bin)]+=1                            
                    if i==0:
                        if count == 0:
                            Item1 = Item
                    elif i==1:
                        if count == 0:
                            Item1 +=Item    
                    else:
                        if count == 0:
                            Item1+=Item
                            self.data_HSV.append(Item1)
                exp.append(self.dis(self.data_HSV[j],self.query_HSV[count]))
            #输出top20
            for each in range(20):
                index = exp.index(min(exp))
                exp[index] = max(exp)
                Index[count][each] = index
            print(count+1,end=" ") 
            print(np.array(Index[count],int))
        
        print("mAp: ",self.mAp(Index))                  
        return
    
    #HOG
    def HOG(self):
        print("方向梯度直方图(HOG)")
        query = np.array(self.query_images_gray)
        data = np.array(self.database_images_gray)
        gamma = 0.5
        # gamma矫正
        query = np.power(query,gamma)
        data = np.power(data,gamma)
        query = np.array(query*255,int)
        data = np.array(data*255,int)
        #水平,垂直梯度
        query_gx = []
        query_gy = []
        data_gx = []
        data_gy = []
        #梯度幅度和方向
        query_g_magnitude = []
        data_g_magnitude = []
        query_g_direction = []       
        data_g_direction = []
        #求梯度
        for each in query:
            Itemx = np.zeros([32,32])
            Itemy = np.zeros([32,32])
            for i in range(32):
                for j in range(32):
                    left = i-1
                    right = i+1
                    if left<0:
                        x = each[right][j]
                    elif right > 31:
                        x = -each[left][j]
                    else:
                        x = each[right][j]-each[left][j]
                    top = j-1
                    bottom = j+1
                    if top<0:
                        y = -each[i][bottom]
                    elif bottom > 31:
                        y = each[i][top]
                    else:
                        y = each[i][top]-each[i][bottom]
                    Itemx[i][j] = int(x) if abs(int(x))!=0.0 else 0.00000001
                    Itemy[i][j] = int(y)
            query_gx.append(Itemx)
            query_gy.append(Itemy)
            query_g_magnitude.append(np.sqrt(Itemx**2+Itemy**2))
            query_g_direction.append(np.arctan(Itemy/Itemx))        
        for each in data:
            Itemx = np.zeros([32,32])
            Itemy = np.zeros([32,32])
            for i in range(32):
                for j in range(32):
                    left = i-1
                    right = i+1
                    if left<0:
                        x = each[right][j]
                    elif right > 31:
                        x = -each[left][j]
                    else:
                        x = each[right][j]-each[left][j]
                    top = j-1
                    bottom = j+1
                    if top<0:
                        y = -each[i][bottom]
                    elif bottom > 31:
                        y = each[i][top]
                    else:
                        y = each[i][top]-each[i][bottom]
                    #防止除零
                    Itemx[i][j] = int(x) if abs(int(x))!=0 else 0.00000001
                    Itemy[i][j] = int(y)
            data_gx.append(Itemx)
            data_gy.append(Itemy)
            data_g_magnitude.append(np.sqrt(Itemx**2+Itemy**2))
            data_g_direction.append(np.arctan(Itemy/Itemx)) 
        #计算梯度直方图
        # 2*2 cell/block 8*8 像素/cell ->16*16
        # 每次移动一个cell 总共算9个block
        bin = 9
        # 坐标分界
        cell_len = 8
        bolck_len = 2
        query_cell = []
        data_cell = []
        pi = 2*1.5707963267948966
        # 每个cell的hog特征 ->Item
        for count in range(len(query)):
            Item = []
            for i in range(4):
                for j in range(4):
                    item = np.zeros([9])
                    left_top = [cell_len*i,cell_len*j]
                    for ii in range(cell_len):
                        for jj in range(cell_len):
                            m = left_top[0]+ii
                            n = left_top[1]+jj
                            item[int((query_g_direction[count][m][n]+pi/2)*9/pi)]+=query_g_magnitude[count][m][n]
                    Item.append(item)
            query_cell.append(Item)
        for count in range(len(data)):
            Item = []
            for i in range(4):
                for j in range(4):
                    item = np.zeros([9])
                    left_top = [cell_len*i,cell_len*j]
                    for ii in range(cell_len):
                        for jj in range(cell_len):
                            m = left_top[0]+ii
                            n = left_top[1]+jj
                            item[int((data_g_direction[count][m][n]+pi/2)*9/pi)]+=data_g_magnitude[count][m][n]
                    Item.append(item)
            data_cell.append(Item)
        #归一化cell
        dis = np.max(query_cell) - np.min(query_cell)
        query_cell = (query_cell - np.min(query_cell))/dis
        dis = np.max(data_cell) - np.min(data_cell)
        data_cell = (data_cell - np.min(data_cell))/dis
        #通过block计算总hog
        query_Hog = []
        data_Hog = []
        for count in range(len(query)):
            Item = []
            for i in range(3):
                #9 个block
                k,m,n= 0+i,4+i,8+i
                Item1 = np.append(query_cell[count][k],query_cell[count][k+1])
                Item1 = np.append(Item1,query_cell[count][k+4])
                Item1 = np.append(Item1,query_cell[count][k+5])
                Item2 = np.append(query_cell[count][m],query_cell[count][m+1])
                Item2 = np.append(Item2,query_cell[count][m+4])
                Item2 = np.append(Item2,query_cell[count][m+5])
                Item3 = np.append(query_cell[count][n],query_cell[count][n+1],)
                Item3 = np.append(Item3,query_cell[count][n+4])
                Item3 = np.append(Item3,query_cell[count][n+5])
                Item = np.append(Item,Item1)
                Item = np.append(Item,Item2)
                Item = np.append(Item,Item3)
            query_Hog.append(Item)
        for count in range(len(data)):
            Item = []
            for i in range(3):
                #9 个block
                k,m,n= 0+i,4+i,8+i
                Item1 = np.append(data_cell[count][k],data_cell[count][k+1])
                Item1 = np.append(Item1,data_cell[count][k+4])
                Item1 = np.append(Item1,data_cell[count][k+5])
                Item2 = np.append(data_cell[count][m],data_cell[count][m+1])
                Item2 = np.append(Item2,data_cell[count][m+4])
                Item2 = np.append(Item2,data_cell[count][m+5])
                Item3 = np.append(data_cell[count][n],data_cell[count][n+1],)
                Item3 = np.append(Item3,data_cell[count][n+4])
                Item3 = np.append(Item3,data_cell[count][n+5])
                Item = np.append(Item,Item1)
                Item = np.append(Item,Item2)
                Item = np.append(Item,Item3)
            data_Hog.append(Item)
        #比较 得到结果
        Index = []
        for i in range(20):
            index = np.zeros([20])
            Exp = []
            for j in range(180):
                Exp.append(self.dis(query_Hog[i],data_Hog[j]))
            for j in range(20):
                index[j] = Exp.index(min(Exp))
                Exp[int(index[j])] = max(Exp)        
            Index.append(index)
            print(i+1,end=" ") 
            print(np.array(index,int))
        print("mAp: ",self.mAp(Index))
        return
    def mAp(self,Index):
        map =0.0
        for i in range(20):
            ap = 0.0
            num = 0
            arr = Index[i]
            cri = self.query_label[i]
            for j in range(len(arr)):
                count = arr[j]
                if self.database_label[int(count)]==cri:
                    num+=1
                    ap+=num/(j+1)
            ap/=len(arr)
            # print(i+1,end = " ")
            # print(ap)
            map +=ap    
        return map/20

if __name__ == '__main__':
    database_path = "CIFAR-10subset/database/"
    query_path = "CIFAR-10subset/query/"
    imageFeature = ImageFeature(database_path,query_path)
    imageFeature.getImages()
    
    imageFeature.colorHistogram()
    
    imageFeature.HOG()