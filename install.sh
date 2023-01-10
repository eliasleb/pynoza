python3 -m venv env
source env/bin/activate
git clone https://github.com/eliasleb/pynoza
cd pynoza || exit
python3 -m pip install --upgrade pip
pip install .
python -c "import pynoza"
