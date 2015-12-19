#!/bin/bash

ORIG_DATA_URL="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
PICKLE_URL="http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
WGET=$(which wget)
CURL=$(which curl)
GZIP=$(which gzip)

if [ -n ${WGET} ]; then
    DL_CMD="wget -c"
elif [ -n ${CURL} ]; then
    DL_CMD="curl -C - -O"
else
    echo "please install curl or wget to download mnist dataset"
fi;

if [ -n ${GZIP} ]; then
    GZIP_CMD="gzip -d"
else
    echo "please install tar for decompress mnist data"
fi;

echo "Download MNIST Dataset from http://yann.lecun.com/exdb/mnist/"
if [ ! -f "mnist.pkl" ]; then
    if [ ! -f "mnist.pkl.gz" ]; then
        ${DL_CMD} ${PICKLE_URL}
    fi;
    ${GZIP_CMD} "mnist.pkl.gz"
else
    echo "MNIST Dataset already exist at mnist.pkl"
fi;
