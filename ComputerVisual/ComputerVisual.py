import numpy as np
import matplotlib.pyplot as plt
import random
# 链码
Chaincode = [0,0,0,0,7,0,0,7,7,7,7,7,7,6,6,7,6,6,6,6,6,6,6,6,5,7,6,7,6,6,5,6,4,5,4,4,3,4,3,6,6,6,6,5,6,5,5,4,5,4,4,4,4,3,4,3,3,2,3,2,2,2,2,5,4,5,4,4,3,4,2,3,2,2,1,2,1,3,2,2,2,2,2,2,2,2,1,2,2,1,1,1,1,1,1,0,0,1,0,0,0,0]

class SplitAndMerge_1(object):
    def __init__(self,Chaincode):
        self.len = np.power(2,0.5)
        #轮廓点
        self.C = []
        #多边形点
        self.P = []
        #链码平移距离
        self.cal = [[self.len,0],[1,1],[0,self.len],[-1,1],[-self.len,0],[-1,-1],[0,-self.len],[1,-1]]
        #8方向链码
        self.cCode = list(Chaincode)
        #多边形起始点
        self.x = 5.0;
        self.y = 5.0;
        self.latelyX = self.x
        self.latelyY = self.y
        return
    #通过链码找到轮廓点
    def findC(self):
        self.C.append((self.x,self.y))
        count = 0
        for each in self.cCode[:]:
            count += 1
            self.C.append(np.array(self.cal[each])+np.array([self.latelyX,self.latelyY]))
            self.latelyX = self.C[count][0]
            self.latelyY = self.C[count][1]
        return
    #初始化P
    def makeP(self):
        while(len(self.P)<27):
            self.P.append(random.randint(0,len(self.C)-1))
            #避免重复
            self.P = list(set(self.P))
            self.P.sort()
        return
    def __exp(self,P):
        exp = 0
        #y = kx + b
        K = 0.0
        B = 0.0
        dis = []
        deno = 0.0
        nume = 0.0
        left = 26
        right = 0
        for i in range(len(self.C)):
            if(i in P):
                left+=1
                left%=27
                right+=1
                right%=27
                dis.append(0)
            else:
                 #y1 = y2 水平
                if(self.C[P[left]][1]-self.C[P[right]][1]==0):
                    dis.append(np.fabs(self.C[i][1]-self.C[P[left]][1]))
                #x1 = x2 垂直
                elif(self.C[P[left]][0]-self.C[P[right]][0]==0):
                    dis.append(np.fabs(self.C[i][0]-self.C[P[left]][0]))
                else:
                    K = (self.C[P[left]][1]-self.C[P[right]][1])/(self.C[P[left]][0]-self.C[P[right]][0])
                    B = self.C[P[left]][1] - K*self.C[P[left]][0]
                    #距离公式分母分子
                    deno = np.sqrt(1+K*K)
                    nume = np.fabs(K*self.C[i][0]+B-self.C[i][1])
                    disnow=np.fabs(nume/deno)
                    dis.append(disnow)
        dis = list(map(lambda x:x**2,dis))
        exp = sum(dis)
        #exp /= len(dis)
        return exp
    def __split(self,P):
        #y = kx + b
        K = 0.0
        B = 0.0
        dis = []
        max_index = -1
        left = 26
        right = 0
        for i in range(len(self.C)):
            if(i in P):
                left+=1
                left%=27
                right+=1
                right%=27
                dis.append(0)
            else:
                #y1 = y2 水平
                if(self.C[P[left]][1]-self.C[P[right]][1]==0):
                    dis.append(np.fabs(self.C[i][1]-self.C[P[left]][1]))
                #x1 = x2 垂直
                elif(self.C[P[left]][0]-self.C[P[right]][0]==0):
                    dis.append(np.fabs(self.C[i][0]-self.C[P[left]][0]))
                else:
                    K = (self.C[P[left]][1]-self.C[P[right]][1])/(self.C[P[left]][0]-self.C[P[right]][0])
                    B = self.C[P[left]][1] - K*self.C[P[left]][0]
                    #距离公式分母分子
                    deno = np.sqrt(1+K**2)
                    nume = np.fabs(K*self.C[i][0]+B-self.C[i][1])
                    disnow =np.fabs(nume/deno)
                    dis.append(disnow)               
        max_index = dis.index(max(dis))
        P.append(max_index)
        P.sort()
        return P
    #减点函数
    def __merge(self,P):
        K = 0.0
        B = 0.0
        left = -1
        right = -1
        dis = []
        mix_index = -1
        for i in range(len(P)):
            left = (i-1)%len(P)
            right = (i+1)%len(P)
            #y1 = y2 水平
            if(self.C[P[left]][1]-self.C[P[right]][1]==0):
                dis.append(np.fabs(self.C[P[i]][1]-self.C[P[left]][1]))
            #x1 = x2 垂直
            elif(self.C[P[left]][0]-self.C[P[right]][0]==0):
                dis.append(np.fabs(self.C[P[i]][0]-self.C[P[left]][0]))
            else:
                K = (self.C[P[left]][1]-self.C[P[right]][1])/(self.C[P[left]][0]-self.C[P[right]][0])
                B = self.C[P[left]][1] - K*self.C[P[left]][0]
                #距离公式分母分子
                deno = np.sqrt(1+K*K)
                nume = np.fabs(K*self.C[P[i]][0]+B-self.C[P[i]][1])
                dis.append(nume/deno)
        min_index = dis.index(min(dis))
        P.remove(P[min_index])
        P.sort()
        return P
    
    
    #找答案
    def __findAnswer(self):
        latelyExp = 100001.0
        nowExp = 100000.0
        newP = list(self.P)
        count =0
        while(nowExp != latelyExp ):
            count+=1
            self.P = list(newP)
            newP = self.__split(newP)
            newP = self.__merge(newP)
            #更新exp
            latelyExp = nowExp
            nowExp = self.__exp(newP) 
            print(nowExp)
            #self.draw()
        print('\n')
        print("最小误差 (局部最优): %f"%self.__exp(self.P))
        return
    #求总误差
   
    def draw(self):
        for each in self.C:
            plt.scatter(each[0],each[1],marker = '.',color = 'black')
        self.__findAnswer()
        for each in range(len(self.P)):
            plt.scatter(self.C[self.P[each]][0],self.C[self.P[each]][1],marker = '.',color = 'r')
            #plt.scatter(self.C[self.P[(each+1)%27]][0],self.C[self.P[(each+1)%27]][1],marker = '.',color = 'r')
            plt.plot([self.C[self.P[each]][0],self.C[self.P[(each+1)%27]][0]],[self.C[self.P[each]][1],self.C[self.P[(each+1)%27]][1]],color = 'green')
          
        plt.show()
        for each in range(len(self.P)):
            plt.scatter(self.C[self.P[each]][0],self.C[self.P[each]][1],marker = '.',color = 'r')
            #plt.scatter(self.C[self.P[(each+1)%27]][0],self.C[self.P[(each+1)%27]][1],marker = '.',color = 'r')
            plt.plot([self.C[self.P[each]][0],self.C[self.P[(each+1)%27]][0]],[self.C[self.P[each]][1],self.C[self.P[(each+1)%27]][1]],color = 'green')
          
        plt.show()
    

if __name__ == '__main__':
    SAM = SplitAndMerge_1(Chaincode)
    SAM.findC()
    SAM.makeP()
    SAM.draw()