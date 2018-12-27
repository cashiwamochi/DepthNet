#!/bin/bash

wget http://perso.ensta-paristech.fr/~pinard/depthnet/pretrained/DepthNet_elu_bn_512.pth.tar
mkdir pretrained
mv DepthNet_elu_bn_512.pth.tar pretrained/
