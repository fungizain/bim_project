# bim_project

git clone https://github.com/fungizain/bim_project.git
cd bim_project

sudo ln -s $(pwd)/bim-app.service /etc/systemd/system/bim-app.service
sudo ln -s $(pwd)/cloudflared.service /etc/systemd/system/cloudflared.service

sudo systemctl daemon-reload

# 啟動 BIM App

sudo systemctl start bim-app
sudo systemctl enable bim-app

# 啟動 Cloudflare Tunnel

sudo systemctl start cloudflared
sudo systemctl enable cloudflared
