sudo apt install npm
sudo npm install pm2@latest -g
pm2 startup
sudo env PATH=$PATH:/usr/bin /usr/local/lib/node_modules/pm2/bin/pm2 startup systemd -u totem --hp /home/totem
pm2 start totem.sh
pm2 save
