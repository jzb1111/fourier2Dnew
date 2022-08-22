# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:25:36 2022

@author: asus
"""

import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n=10
mat_len=n*4

'''def gen_star_map(n):
    num=n*4
    print(num)
    tmp=np.ones((num,num))    
    #rowlis=[]
    for i in range(n):
        for j in range(num-(2*(i+1)-1)*2):
            tmp[i][(i+1)*2-1+j]=0
            tmp[num-i-1][(i+1)*2-1+j]=0
            #print(i,(i+1)*2-1+j)
            tmp[(i+1)*2-1+j][i]=0
            tmp[(i+1)*2-1+j][num-i-1]=0
    return tmp

def gen_hole_num(n):

    split=1/((n*2)+1)
    #tmp=np.ones((num,num)) 
    res=[]
    for i in range(n):
        short=0.5-(i+1)*split
        res.append((0.5**2-short**2)**0.5)
    res.extend(sorted(res,reverse=True))
    #print(res)
    reslis=[]
    for i in res:
        reslis.append(int(np.round(i/res[0])*4))
    return reslis

def gen_ratio_lis(n):
    res=[]
    split=1/(n*2+1)
    tmplis=[]
    for i in range(n):
        xb=0.5-split*(i+1)
        #print(xb)
        tmp=(0.5**2-xb**2)**0.5
        tmplis.append(tmp)
    for i in range(len(tmplis)):
        res.append(tmplis[i]/tmplis[0])
    res+=sorted(res,reverse=True)
    #print(res)
    res=np.floor(np.array(res)*3)
    return res

def gen_star_map_ratio(n):
    res=[]
    for i in range(1,n+1):
        res.append(i*4+(i*2-2)*2)
    res.extend(sorted(res,reverse=True))
    return res

def get_n_match(max_n):
    res=[]
    for i in range(1,max_n+1):
        grl=gen_ratio_lis(i)
        gsmr=gen_star_map_ratio(i)
        diff=sum(((grl-gsmr)**2)**0.5)
        res.append([diff,i])
    return res

def gen_row_distribution(diatribution_a,row_b,num_b):
    split=row_b/num_b
    tmp=np.zeros((row_b))
    for i in range(num_b):
        tmp[i*split]=1
        
    return

def distribution2peak(distribution,resolution):
    tmp=np.zeros((resolution))
    onelis=[]
    splitlis=[]
    for i in range(len(tmp)):
        if distribution[i]==1:
            tmp[int(resolution/len(distribution)*i)]=1
            onelis.append(int(resolution/len(distribution)*i))
    for i in range(len(onelis)):
        pass
    return

def gen_use_map(n):
    circle_lis=[i*2*2+(i*2-2)*2 for i in range(1,n+1)]
    circle_lis.extend(sorted(circle_lis,reverse=True))
    ratio_lis=gen_ratio_lis(n)
    
    return 

def add_ax(ax,mat):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            for k in range(len(mat[i][j])):
                if mat[i][j][k]==1:
                    ax.scatter(i,j,k,c='r',marker='o')
    return ax'''

def gen_center_mat(mat_len):
    res=np.zeros((mat_len,mat_len,2))
    for i in range(mat_len):
        for j in range(mat_len):
            res[i][j][0]=-(i-50)
            res[i][j][1]=j-50
    return res

def gen_radius_mat(center_mat):
    r_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(center_mat)):
        for j in range(len(center_mat[i])):
            if center_mat[i][j][0]==0 or center_mat[i][j][1]==0:
                r=1
            else:
                big_v=abs(center_mat[i][j][0]) if abs(center_mat[i][j][0])>abs(center_mat[i][j][1]) else abs(center_mat[i][j][1])
                small_v=abs(center_mat[i][j][1]) if abs(center_mat[i][j][0])>abs(center_mat[i][j][1]) else abs(center_mat[i][j][0])
                r=np.cos(abs(np.arctan(small_v/big_v)))
            r_mat[i][j]=r
    return r_mat

def gen_rotate_mat(center_mat):
    #生成旋转单位向量
    vector_mat=np.zeros((len(center_mat),len(center_mat[0]),3))
    for i in range(len(vector_mat)):
        for j in range(len(vector_mat[i])):
            #v=[]
            if center_mat[i][j][0]==0:
                v=[0,1,0]
            if center_mat[i][j][1]==0:
                v=[1,0,0]
            
            #vk=np.tan((np.arctan(center_mat[i][j][0]/center_mat[i][j][1])/np.pi+0.5)*np.pi)
            if center_mat[i][j][0]!=0 and center_mat[i][j][1]!=0:
                vk=[center_mat[i][j][0],center_mat[i][j][1]]/(center_mat[i][j][0]**2+center_mat[i][j][1]**2)**0.5
            else:
                vk=[0,0]
            #print(vk,i,j)
            if center_mat[i][j][0]>0 and center_mat[i][j][1]>0:
                v=[abs(vk[1]),-abs(vk[0]),0]
            if center_mat[i][j][0]>0 and center_mat[i][j][1]<0: 
                v=[-abs(vk[1]),-abs(vk[0]),0]
            if center_mat[i][j][0]<0 and center_mat[i][j][1]<0:
                v=[-abs(vk[1]),abs(vk[0]),0]
            if center_mat[i][j][0]<0 and center_mat[i][j][1]>0: 
                v=[abs(vk[1]),abs(vk[0]),0]
            vector_mat[i][j][0]=v[0]
            vector_mat[i][j][1]=v[1]
            vector_mat[i][j][2]=v[2]
    return vector_mat

def gen_angle_mat(center_mat,r_mat):
    #生成旋转角度
    angle_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(center_mat)):
        for j in range(len(center_mat[i])):
            l=((center_mat[i][j][0]**2+center_mat[i][j][1]**2)**0.5*r_mat[i][j])/50.5
            angle_mat[i][j]=l
    return angle_mat

def split(n):
    if n>100:
        return 100
    elif n<0:
        return 0
    else:
        return n

def gen_twining_mat(plane,angle_mat,vector_mat,r_mat,f_num):
    #生成旋转后的坐标
    res=np.zeros((len(angle_mat),len(angle_mat[0]),3))
    for i in range(len(plane)):
        for j in range(len(plane[i])):
            #p_high=test_plane[i][j]
            #p_high=plane[i][j]
            p=np.quaternion(0,0,0,-1)
            angle=angle_mat[i][j]*np.pi*f_num#0.5
            v=vector_mat[i][j]
            a=np.cos(angle/2)*np.quaternion(1,0,0,0)
            b=np.sin(angle/2)*v[0]*np.quaternion(0,1,0,0)
            c=np.sin(angle/2)*v[1]*np.quaternion(0,0,1,0)
            d=np.sin(angle/2)*v[2]*np.quaternion(0,0,0,1)
            h=a+b+c+d
            h_=a-b-c-d            
            p_new=h*p*h_
            p_new=p_new*plane[i][j]
            res[i][j][0]=p_new.x
            res[i][j][1]=p_new.y
            res[i][j][2]=p_new.z
    return res

def get_f_series_by_plane(plane,f_num):
    #生成傅里叶级数
    f_lis=[]
    for i in range(f_num):
        center_mat=gen_center_mat(len(plane))#不需要输入卷绕数
        r_mat=gen_radius_mat(center_mat)#不需要输入卷绕数
        vector_mat=gen_rotate_mat(center_mat)#需要输入卷绕数
        angle_mat=gen_angle_mat(center_mat, r_mat)#需要
        twining_mat=gen_twining_mat(plane,angle_mat,vector_mat,r_mat,i)
        weight_mat=gen_weight_mat(center_mat,r_mat)
        weight_wrap=twining_mat*weight_mat
        f_lis.append(np.mean(weight_wrap))
    return f_lis
    
def gen_weight_mat(center_mat,r_mat,f_num):
    #生成权重
    w_mat=np.zeros((len(center_mat),len(center_mat[0])))
    round_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(center_mat)):
        for j in range(len(center_mat[i])):
            turn_num=(center_mat[i][j][0]**2+center_mat[i][j][1]**2)**0.5
            round_order=np.max([abs(center_mat[i][j])])
            round_mat[i][j]=get_round_num(round_order)
            tmp=turn_num*r_mat[i][j]/50
            w_mat[i][j]=1-tmp
            #radius=1/51*turn_num
    tmp=np.zeros((len(center_mat),len(center_mat[0])))
    radius_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(w_mat)):
        for j in range(len(w_mat[i])):
            tmp[i][j]=2*w_mat[i][j] if w_mat[i][j]<=0.5 else 2-w_mat[i][j]*2
    for i in range(len(radius_mat)):
        for j in range(len(radius_mat[i])):
            radius_mat[i][j]=get_radius_num(tmp[i][j])
    max_line=np.max(round_mat)
    round_ratio_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(round_mat)):
        for j in range(len(round_mat[i])):
            round_ratio_mat[i][j]=round_mat[i][j]/max_line
    res=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(res)):
        for j in range(len(res[i])):
            if i==50 and j==50:
                res[i][j]=1
            else:
                res[i][j]=radius_mat[i][j]/round_ratio_mat[i][j]
    return res

def gen_weight_mat_by_f_num(center_mat,r_mat,f_num):
    #生成权重
    
    f_len=len(center_mat)//f_num
    round_half=50//f_num
    
    w_mat=np.zeros((len(center_mat),len(center_mat[0])))
    round_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(center_mat)):
        for j in range(len(center_mat[i])):
            turn_num=(center_mat[i][j][0]**2+center_mat[i][j][1]**2)**0.5
            round_order=np.max([abs(center_mat[i][j])])
            round_mat[i][j]=get_round_num(round_order)
            tmp=turn_num*r_mat[i][j]/50#修改这里使f_num发挥作用
            
            w_mat[i][j]=1-tmp
            #radius=1/51*turn_num
    #round_mat:每个点所在的圈数
    #turn_num:每个点所在的长度（伸展的长度）
    #tmp:每个点向边缘的伸展程度()
    #w_mat:1-每个点向边缘的伸展程度
    
    tmp=np.zeros((len(center_mat),len(center_mat[0])))
    radius_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(w_mat)):
        for j in range(len(w_mat[i])):
            tmp[i][j]=2*w_mat[i][j] if w_mat[i][j]<=0.5 else 2-w_mat[i][j]*2
    for i in range(len(radius_mat)):
        for j in range(len(radius_mat[i])):
            radius_mat[i][j]=get_radius_num(tmp[i][j])
    
    #tmp:每个点的对应半球高度 ❤需要对应的f_num
    #radius_mat:每个点的对应半径 ❤需要对应的f_num
    
    max_line=np.max(round_mat)
    round_ratio_mat=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(round_mat)):
        for j in range(len(round_mat[i])):
            round_ratio_mat[i][j]=round_mat[i][j]/max_line
            
    #max_line:最大的圈数
    #round_ratio_mat:每个点对应的圈数/最大圈数
    
    res=np.zeros((len(center_mat),len(center_mat[0])))
    for i in range(len(res)):
        for j in range(len(res[i])):
            if i==50 and j==50:
                res[i][j]=1
            else:
                res[i][j]=radius_mat[i][j]/round_ratio_mat[i][j]
    
    #res:每个点对应的半径/（每个点对应的圈数/最大圈数） ❤需要对应的f_num
    
    return res

def get_round_num(round_ind):
    base=round_ind*2-1
    res=base*4+4
    return res

def get_radius_num(tmp):
    if tmp<0.000000001:
        return 0
    else:
        return (1-(1-tmp)**2)**0.5
    
