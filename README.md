<h1 align="center">TotemYOLO</h1>
<h3 align="center">Vision system for covid control totems inside the UFSM.</h3>

<p align="center"> 
  <img src="https://img.shields.io/badge/PyTorch-v1.6.0-blue"/>
  <img src="https://img.shields.io/badge/PyTorch_Lightning-v1.0.6-blue"/>
  <img src="https://img.shields.io/badge/OpenCV-v4.4.0.42-blue"/>
  <img src="https://img.shields.io/badge/Torchvision-v0.8.1-blue"/>
  <img src="https://img.shields.io/badge/Pandas-v1.1.4-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.19.2-blue"/>
</p>
<br/>

## Objective
<p align="justify"> 
  <a>Implementation of vision system for covid-19 totems control through real-time facial mask detection. This approach seeks to work effectively on low-cost hardware, such as the Jetson Nano B01, 4gb RAM memory version.</a>
</p>
  

## Setup

<p align="justify"> 
 <a>To setup your system just use git clone command, all the files are included in this repository. To do it, use:</a>
</p>

```shell
git clone https://github.com/victorkich/TotemUFSM/
```

<a>All of requirements is show in the badgets above, but if you want to install all of them, enter the repository and execute the following line of code:</a>
</p>

```shell
pip3 install -r requirements.txt
```

<a>If you want the program run at the system startup, use the following code to setup the startup settings using pm2:</a>
</p>

```shell
sudo ./setup.sh
```

<p align="justify"> 
 <a>For test in your own computer, just type the following line of code:</a>
</p>

```shell
python3 totem.py
```

## Example

<p align="center"> 
  <img src="media/example.gif" alt="TotemUFSM"/>
</p>  

<p align="center"> 
  <i>If you liked this repository, please don't forget to starred it!</i>
  <img src="https://img.shields.io/github/stars/victorkich/TotemUFSM?style=social"/>
</p>
