#!/bin/bash

TOOL_NAME=RPM
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(dirname $(realpath $0)))

apt-get update &&

# install python stuff needed for onnx -> nnet
apt-get install -y python3 python3-pip &&
apt-get install -y psmisc && # for killall, used in prepare_instance.sh script
pip3 install -r "$DIR/requirements_py.txt"

# install julia stuff
wget https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.1-linux-x86_64.tar.gz
tar zxvf julia-1.4.1-linux-x86_64.tar.gz

# run Julia by calling /julia-1.4.1/bin/julia


