#!/bin/bash

sudo apt-get update
sleep 0.1
echo "Installing required packages..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
echo "... to allow using a repository over HTTPS"
echo "Adding Docker's official GPG key"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
echo "Verifying key fingerprint"
sudo apt-key fingerprint 0EBFCD88
echo "Setting up stable docker repository"
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
echo "Installing Docker engine"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
