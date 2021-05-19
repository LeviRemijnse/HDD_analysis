rm -rf DFNDataReleases
git clone https://github.com/cltl/DFNDataReleases.git
cd DFNDataReleases
git checkout 5209544e4951778ff746094fdb482898fd9326d2
pip install -r requirements.txt
bash install.sh
