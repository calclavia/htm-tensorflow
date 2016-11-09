#!/bin/bash
TF_INC=$(python3.5 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ --std=c++11 --shared inhibit.cc -o inhibit.so -fPIC -I $TF_INC -O2 #-undefined dynamic_lookup
