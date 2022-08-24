#!/bin/sh

cd develop/include
awk 1 hspp_develop.h > ../../include/hspp.h
