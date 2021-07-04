# #!/bin/bash

# install julia
apt-get update -y
apt-get install sudo -y
sudo apt-get update -y
sudo apt-get install wget -y
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.1-linux-x86_64.tar.gz
sudo tar -xvzf julia-1.4.1-linux-x86_64.tar.gz
sudo cp -r julia-1.4.1 /opt/
sudo ln -s /opt/julia-1.4.1/bin/julia /usr/local/bin/julia
sudo apt-get install build-essential -y
sudo apt-get install git -y
sudo apt-get install python3 -y
sudo apt-get install python3-pip -y
sudo apt-get install psmisc
source ~/.bashrc

# get path names
script_name=$0
script_path=$(dirname "$0")
project_path=$(dirname "$script_path")

# install julia project environment
cd $project_path
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
' | julia

# install python packages
#pip3 install -r "/requirements_py.txt"




