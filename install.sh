python3 -m venv env
source env/bin/activate
git clone https://c4science.ch/source/pynoza.git
cd pynoza || exit
python3 -m pip install --upgrade pip
pip install --use-feature=in-tree-build .