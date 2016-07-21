#!/bin/bash

PWD_=$PWD
PROJECT=opencv
CORES=${CORES:-`grep -c ^processor /proc/cpuinfo`}

rm -rf $PWD_/build/$PROJECT
rm -rf $PWD_/build_rel/$PROJECT

echo "BUILDING $PROJECT"
mkdir -p $PWD_/build/$PROJECT
cd $PWD_/build/$PROJECT

cmake ../../$PROJECT -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DBUILD_DOCS=false \
	-DBUILD_SHARED_LIBS=true \
	-DWITH_CUDA=false \
	-DWITH_QT=true \
	-DWITH_EIGEN=true \
	-DWITH_1394=false \
	-DWITH_OPENEXR=false \
	-DWITH_V4L=true \
	-DWITH_FFMPEG=true \
	-DWITH_GSTREAMER=false \
	-DBUILD_NEW_PYTHON_SUPPORT=false \
	-DBUILD_EXAMPLES=true \
	-DBUILD_TESTS=false \
	-DWITH_OPENGL=true \
	-DCMAKE_INSTALL_PREFIX=$PWD_/build_rel/$PROJECT
make -j$CORES | tee $PWD_/$PROJECT.log
make install

cd $PWD_
