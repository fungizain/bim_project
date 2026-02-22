# bim_project

git clone https://github.com/fungizain/bim_project.git
cd bim_project

# poetry

python3 -m venv .venv
source .venv/bin/activate
pip install poetry

# docker

docker build -t bim-app .
docker run -d --name bim-app -p 8080:8080 bim-app
sudo ln -s $(pwd)/docker-bim-app.service /etc/systemd/system/docker-bim-app.service

sudo systemctl start docker-bim-app
sudo systemctl enable docker-bim-app

# normal

sudo ln -s $(pwd)/bim-app.service /etc/systemd/system/bim-app.service
sudo ln -s $(pwd)/cloudflared.service /etc/systemd/system/cloudflared.service
sudo ln -s $(pwd)/celery.service /etc/systemd/system/celery.service
sudo systemctl daemon-reload

# 啟動

sudo systemctl start bim-app
sudo systemctl enable bim-app
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
sudo systemctl start celery
sudo systemctl enable celery

# utils

apt install poppler-utils tesseract-ocr redis-server -y

# openai MXFP4

need accelerate, kernels and triton>=3.4
