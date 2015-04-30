#!/bin/bash


wget https://www.dropbox.com/s/179hboknexmcdm5/best.net?dl=0
mv best.net?dl=0 lmodel_best_3.net
/home/pb1672/torch/install/bin/th main.lua

