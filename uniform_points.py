# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 20:04:57 2022

@author: asus
"""

import numpy as np

def generate_uniform_points(num):
    points_theta=[]
    points_phi=[]
    a=4*np.pi*1/num
    d=a**0.5
    num_phi=round(np.pi/d)
    d_phi=np.pi/num_phi
    d_theta=a/d_phi
    #print(d_phi,d_theta)
    for i in range(num_phi):
        phi=np.pi*(i+0.5)/num_phi
        num_theta=round(2*np.pi*np.sin(phi)/d_theta)
        for j in range(int(num_theta)):
            theta=2*np.pi*j/num_theta
            #points_phi.append((phi-np.pi/2)/(np.pi*2/360))
            points_phi.append((phi)/(np.pi*2/360))
            points_theta.append(theta/(np.pi*2/360)+4)
    points=np.stack([points_theta,points_phi],axis=1)
    return points

def coor2angle(coor):
    x=coor[0]
    y=coor[1]
    z=coor[2]
    xy = np.sqrt(x**2 + y**2) # sqrt(x² + y²)
    
    x_2 = x**2
    y_2 = y**2
    z_2 = z**2
 
    r = np.sqrt(x_2 + y_2 + z_2) # r = sqrt(x² + y² + z²)
 
    theta = np.arctan2(y, x) 
 
    phi = np.arctan2(xy, z) 
 
    return r, theta, phi

def angle2coor(angle,r=1):
    theta=angle[0]
    phi=angle[1]
    x=r*np.sin(phi)*np.cos(theta)
    y=r*np.sin(phi)*np.sin(theta)
    z=r*np.cos(phi)
    return [x,y,z]

def sphere2plane(sphere,plane_len=100):
    coor_lis=[]
    plane=np.zeros((plane_len,plane_len))
    for i in range(len(sphere)):
        theta=sphere[i][0]/180*np.pi
        phi=(sphere[i][1]+90)/180*np.pi
        yxratio=np.tan(theta)
        x=(1/(np.tan(theta)**2+1))**0.5
        y=(1-x**2)**0.5
        y,x=(y*(1-phi))*plane_len*0.5,(x*(1-phi))*plane_len*0.5
        coor_lis.append([y,x])
    for i in range(1):
        pass
    return coor_lis