module load python
virtualenv venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements_cc.txt